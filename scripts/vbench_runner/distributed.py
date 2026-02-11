"""
Multi-GPU coordination: rank management, subtask splitting, and auto-launch.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from collections.abc import Callable

import pandas as pd

try:
    from .env import PROJECT_ROOT, get_vbench_subtasks, logger
except ImportError:
    from vbench_runner.env import PROJECT_ROOT, get_vbench_subtasks, logger


def init_distributed_if_needed() -> tuple[int, int, Callable[[], None]]:
    """
    Read torchrun rank/world_size without initializing torch.distributed.

    We intentionally avoid process-group init because this pipeline parallelizes
    by dimensions (different ranks run different dimensions). VBench internals
    use all_gather/barrier and can deadlock if a global process group is active.

    Returns:
        (rank, world_size, barrier_fn)
    """
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env <= 1:
        return 0, 1, (lambda: None)
    rank = int(os.environ.get("RANK", "0"))
    return rank, world_size_env, (lambda: None)


def _parse_visible_devices() -> list[str]:
    """Parse visible CUDA device identifiers."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        visible = visible.strip()
        if not visible or visible == "-1":
            return []
        return [item.strip() for item in visible.split(",") if item.strip()]

    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        count = 0
    return [str(i) for i in range(max(0, count))]


def split_subtasks_for_rank(subtasks: list[str], rank: int, world_size: int) -> list[str]:
    """Round-robin split: 16 dims, 4 ranks => each rank 4 dims (all videos)."""
    if world_size <= 1:
        return list(subtasks)
    return [subtask for idx, subtask in enumerate(subtasks) if idx % world_size == rank]


def merge_rank_partial_results(partial_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge rank-local wide tables into one final wide table on CPU."""
    long_rows = []
    for frame in partial_frames:
        if frame is None or frame.empty or "video_id" not in frame.columns:
            continue
        metric_cols = [
            col
            for col in frame.columns
            if col not in {"video_id", "group", "vbench_temporal_score"}
        ]
        for col in metric_cols:
            if frame[col].notna().any():
                temp = frame[["video_id", col]].copy()
                temp["subtask"] = col
                temp = temp.rename(columns={col: "score"})
                long_rows.append(temp)

    if not long_rows:
        return pd.DataFrame(columns=["video_id", "vbench_temporal_score"])

    df_long = pd.concat(long_rows, ignore_index=True)
    df_long = (
        df_long.groupby(["video_id", "subtask"], as_index=False)["score"]
        .mean()
        .reset_index(drop=True)
    )
    df_wide = df_long.pivot(index="video_id", columns="subtask", values="score").reset_index()
    metric_cols = [col for col in df_wide.columns if col != "video_id"]
    if metric_cols:
        df_wide["vbench_temporal_score"] = df_wide[metric_cols].mean(axis=1)
    return df_wide


def make_file_barrier(sync_dir: Path, rank: int, world_size: int) -> Callable[[], None]:
    """
    Build a filesystem-based barrier for multi-process coordination.

    This avoids torch.distributed dependency while still synchronizing stages
    (e.g., one-time preprocess completion, partial result collection).
    """
    sync_dir.mkdir(parents=True, exist_ok=True)
    counter_lock = threading.Lock()
    stage_counter = {"value": 0}

    def _barrier(timeout_sec: float = 7200.0, poll_sec: float = 0.2) -> None:
        if world_size <= 1:
            return
        with counter_lock:
            stage_counter["value"] += 1
            stage = stage_counter["value"]
        stage_dir = sync_dir / f"stage_{stage:03d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        marker = stage_dir / f"rank_{rank}.done"
        marker.write_text(str(time.time()), encoding="utf-8")

        deadline = time.time() + timeout_sec
        while True:
            ready = len(list(stage_dir.glob("rank_*.done")))
            if ready >= world_size:
                return
            if time.time() > deadline:
                raise TimeoutError(
                    f"File barrier timeout at {stage_dir} ({ready}/{world_size} ready)"
                )
            time.sleep(poll_sec)

    return _barrier


def maybe_auto_launch_multi_gpu(
    args: argparse.Namespace,
    config: dict,
    rank: int,
    world_size: int,
    output_dir: Path,
) -> bool:
    """
    Auto-launch torch distributed when multiple GPUs are visible.

    When launched, the parent process manages a live progress board on the
    terminal while torchrun worker output is captured to a log file.
    """
    if rank != 0 or world_size > 1:
        return False
    if getattr(args, "no_auto_multi_gpu", False):
        return False
    if os.environ.get("VBENCH_AUTO_MULTI_GPU_LAUNCHED") == "1":
        return False

    runtime_config = config.get("runtime", {})
    device = str(runtime_config.get("device", "cuda")).lower()
    if not device.startswith("cuda"):
        return False

    subtasks = get_vbench_subtasks(config)
    if len(subtasks) <= 1:
        return False

    visible_devices = _parse_visible_devices()
    if len(visible_devices) <= 1:
        return False

    worker_count = min(len(visible_devices), len(subtasks))
    if worker_count <= 1:
        return False

    logger.info(
        "Auto multi-GPU enabled: %d visible GPUs, %d subtasks -> launching %d workers.",
        len(visible_devices),
        len(subtasks),
        worker_count,
    )

    entry_script = PROJECT_ROOT / "scripts" / "run_vbench.py"
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(worker_count),
        str(entry_script),
        "--config",
        str(args.config),
    ]
    if args.force:
        cmd.append("--force")
    if args.skip_on_error:
        cmd.append("--skip-on-error")
    if getattr(args, "no_auto_multi_gpu", False):
        cmd.append("--no-auto-multi-gpu")

    env = os.environ.copy()
    env["VBENCH_AUTO_MULTI_GPU_LAUNCHED"] = "1"
    env["VBENCH_PROGRESS_PARENT_MANAGED"] = "1"

    # --- Prepare parent-managed progress board ---
    assignment_map = {
        r: split_subtasks_for_rank(subtasks, r, worker_count) for r in range(worker_count)
    }
    gpu_map = {
        r: visible_devices[r] if r < len(visible_devices) else str(r)
        for r in range(worker_count)
    }
    progress_dir = output_dir / "vbench_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    for stale in progress_dir.glob("rank_*.json"):
        try:
            stale.unlink()
        except OSError:
            pass

    log_path = output_dir / "vbench_workers.log"

    try:
        from .progress import MultiGpuProgressBoard
    except ImportError:
        from vbench_runner.progress import MultiGpuProgressBoard

    board = MultiGpuProgressBoard(
        progress_dir=progress_dir,
        assignment_map=assignment_map,
        gpu_map=gpu_map,
        refresh_sec=1.0,
        log_path=log_path,
    )
    board.start()

    try:
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
    finally:
        board.stop()

    # --- Post-run summary ---
    _print_run_summary(log_path)

    if result.returncode != 0:
        logger.error("Auto multi-GPU torchrun failed with code %d.", result.returncode)
        logger.error("Full worker log: %s", log_path)
        try:
            with open(log_path, errors="replace") as f:
                lines = f.readlines()
            tail = lines[-30:] if len(lines) > 30 else lines
            for line in tail:
                logger.error("  | %s", line.rstrip())
        except Exception:
            pass

        if args.skip_on_error:
            logger.warning("Skipping VBench evaluation due to auto multi-GPU failure.")
            pd.DataFrame(columns=["video_id", "vbench_temporal_score"]).to_csv(
                output_dir / "vbench_per_video.csv", index=False
            )
            return True
        raise RuntimeError(f"torchrun failed with exit code {result.returncode}")

    return True


def _print_run_summary(log_path: Path) -> None:
    """Print post-run summary extracted from worker log file."""
    try:
        with open(log_path, errors="replace") as f:
            lines = f.readlines()
    except Exception:
        return

    summary_keywords = [
        "results saved to",
        "score summary",
        "coverage summary",
        "error",
        "failed",
        "merged",
        "copied",
    ]
    important = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(kw in lower for kw in summary_keywords):
            important.append(stripped)

    if important:
        logger.info("=== Worker Summary ===")
        for line in important[-20:]:
            logger.info("  %s", line)
    logger.info("Full worker log: %s", log_path)
