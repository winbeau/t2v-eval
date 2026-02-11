#!/usr/bin/env python3
"""
run_vbench.py
Run VBench evaluation using the official VBench implementation.

This script wraps the official VBench repository (https://github.com/Vchitect/VBench)
to evaluate temporal quality of generated videos.
It can run from:
1) `processed_metadata.csv` (preferred), or
2) `metadata.csv`, or
3) direct local video folders configured in `dataset.local_video_dir`.

Usage:
    python scripts/run_vbench.py --config configs/eval.yaml
"""

import argparse
from collections import Counter
import contextlib
from copy import deepcopy
import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml
try:
    from .dimensions.registry import (
        CLIP_REQUIRED_DIMENSIONS,
        LONG_DIMENSION_SET,
        PYIQA_REQUIRED_DIMENSIONS,
        default_long_subtasks,
        normalize_subtasks,
        supported_long_subtasks,
    )
except ImportError:
    from vbench_runner.dimensions.registry import (
        CLIP_REQUIRED_DIMENSIONS,
        LONG_DIMENSION_SET,
        PYIQA_REQUIRED_DIMENSIONS,
        default_long_subtasks,
        normalize_subtasks,
        supported_long_subtasks,
    )

# =============================================================================
# Setup paths to use official VBench from submodule
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VBENCH_ROOT = PROJECT_ROOT / "third_party" / "VBench"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

AUXILIARY_REQUIRED_LONG_DIMS: set[str] = {
    "object_class",
    "multiple_objects",
    "scene",
    "appearance_style",
    "color",
    "spatial_relationship",
}

COLOR_WORDS = (
    "white",
    "red",
    "pink",
    "blue",
    "silver",
    "purple",
    "orange",
    "green",
    "gray",
    "grey",
    "yellow",
    "black",
    "brown",
)

SPATIAL_RELATIONS = (
    "on the left of",
    "on the right of",
    "on the top of",
    "on the bottom of",
)

PROMPT_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "with",
    "for",
    "and",
    "or",
    "from",
    "by",
    "style",
    "view",
    "shot",
    "camera",
    "front",
    "back",
    "side",
    "left",
    "right",
    "top",
    "bottom",
}


def check_vbench_installation() -> bool:
    """Check if VBench submodule is properly initialized."""
    vbench_init_file = VBENCH_ROOT / "vbench" / "__init__.py"
    if not vbench_init_file.exists():
        logger.error("=" * 70)
        logger.error("VBench submodule not found or not initialized!")
        logger.error("")
        logger.error("Please run the following commands from project root:")
        logger.error("  git submodule update --init --recursive")
        logger.error("")
        logger.error("Or if cloning fresh:")
        logger.error("  git clone --recurse-submodules <repo_url>")
        logger.error("=" * 70)
        return False
    return True


def setup_vbench_path():
    """Add VBench to Python path."""
    vbench_path = str(VBENCH_ROOT)
    if vbench_path not in sys.path:
        sys.path.insert(0, vbench_path)


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


def _shorten_text(text: str, max_len: int) -> str:
    """Shorten text for compact terminal table display."""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


class RankProgressReporter:
    """Per-rank progress reporter writing JSON status for controller display."""

    def __init__(
        self,
        progress_dir: Path,
        rank: int,
        local_rank: int,
        assigned_subtasks: list[str],
        visible_devices: list[str] | None = None,
    ):
        self.progress_dir = progress_dir
        self.rank = rank
        self.local_rank = local_rank
        self.assigned_subtasks = list(assigned_subtasks)
        self.visible_devices = visible_devices or []
        self._lock = threading.Lock()
        self._completed_subtasks = 0
        self._current_subtask: str | None = None
        self._current_percent: int | None = None
        self._status_text = "idle"
        self._elapsed_sec = 0
        self._done = False
        self._last_error: str | None = None
        self._write_status()

    @property
    def _status_file(self) -> Path:
        return self.progress_dir / f"rank_{self.rank}.json"

    def _gpu_label(self) -> str:
        if self.visible_devices and 0 <= self.local_rank < len(self.visible_devices):
            return self.visible_devices[self.local_rank]
        return str(self.local_rank)

    def _next_subtask(self) -> str:
        if self._current_subtask:
            return self._current_subtask
        if self._completed_subtasks < len(self.assigned_subtasks):
            return self.assigned_subtasks[self._completed_subtasks]
        return "-"

    def _write_status(self) -> None:
        payload = {
            "rank": self.rank,
            "gpu": self._gpu_label(),
            "assigned_total": len(self.assigned_subtasks),
            "assigned_subtasks": self.assigned_subtasks,
            "completed_subtasks": self._completed_subtasks,
            "current_subtask": self._current_subtask,
            "next_subtask": self._next_subtask(),
            "percent": self._current_percent,
            "status_text": self._status_text,
            "elapsed_sec": int(self._elapsed_sec),
            "done": self._done,
            "error": self._last_error,
            "updated_at": time.time(),
        }
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._status_file.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp_path.replace(self._status_file)

    def start_task(self, task_name: str, status_text: str = "running") -> None:
        with self._lock:
            self._current_subtask = task_name
            self._current_percent = None
            self._status_text = status_text
            self._elapsed_sec = 0
            self._write_status()

    def update_live(self, percent: int | None, status_text: str, elapsed_sec: int) -> None:
        with self._lock:
            if percent is not None:
                self._current_percent = max(0, min(99, int(percent)))
            else:
                self._current_percent = None
            self._status_text = status_text
            self._elapsed_sec = int(elapsed_sec)
            self._write_status()

    def finish_task(
        self,
        success: bool = True,
        error: str | None = None,
        count_completion: bool = True,
    ) -> None:
        with self._lock:
            if count_completion and self._current_subtask is not None:
                self._completed_subtasks += 1
            self._current_percent = 100
            self._status_text = "done" if success else "failed"
            self._last_error = _shorten_text(str(error), 120) if error else None
            self._current_subtask = None
            self._elapsed_sec = 0
            self._write_status()

    def mark_done(self) -> None:
        with self._lock:
            self._done = True
            self._current_subtask = None
            self._current_percent = 100 if self.assigned_subtasks else 0
            self._status_text = "done"
            self._elapsed_sec = 0
            self._write_status()


class MultiGpuProgressBoard:
    """Rank-0 terminal board rendering all rank progress lines."""

    def __init__(
        self,
        progress_dir: Path,
        assignment_map: dict[int, list[str]],
        gpu_map: dict[int, str] | None = None,
        refresh_sec: float = 1.0,
        non_tty_snapshot_sec: float = 10.0,
    ):
        self.progress_dir = progress_dir
        self.assignment_map = assignment_map
        self.gpu_map = gpu_map or {}
        self.world_size = len(assignment_map)
        self.refresh_sec = refresh_sec
        self.non_tty_snapshot_sec = max(1.0, float(non_tty_snapshot_sec))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._printed_live_block = False
        self._use_ansi = bool(getattr(sys.__stdout__, "isatty", lambda: False)())
        self._last_non_tty_emit = 0.0

    def _status_file(self, rank: int) -> Path:
        return self.progress_dir / f"rank_{rank}.json"

    def _read_rank_status(self, rank: int) -> dict:
        path = self._status_file(rank)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        assigned = self.assignment_map.get(rank, [])
        return {
            "rank": rank,
            "gpu": self.gpu_map.get(rank, str(rank)),
            "assigned_total": len(assigned),
            "assigned_subtasks": assigned,
            "completed_subtasks": 0,
            "current_subtask": None,
            "next_subtask": assigned[0] if assigned else "-",
            "percent": 0,
            "status_text": "waiting",
            "elapsed_sec": 0,
            "done": False,
            "error": None,
        }

    def _print_assignment_table(self) -> None:
        rows = []
        for rank in range(self.world_size):
            tasks = self.assignment_map.get(rank, [])
            rows.append(
                {
                    "gpu": f"gpu{self.gpu_map.get(rank, str(rank))}",
                    "tasks": _shorten_text(", ".join(tasks) if tasks else "-", 52),
                    "done": f"0/{len(tasks)}",
                    "next": _shorten_text(tasks[0] if tasks else "-", 20),
                }
            )

        headers = {"gpu": "GPU", "tasks": "Assigned Tasks", "done": "Done", "next": "Next Task"}
        widths = {
            key: max(len(headers[key]), max(len(str(row[key])) for row in rows) if rows else 0)
            for key in headers
        }

        sep = (
            "+"
            + "+".join("-" * (widths[key] + 2) for key in ["gpu", "tasks", "done", "next"])
            + "+"
        )
        print("\n[Multi-GPU Progress Board]", file=sys.__stdout__)
        print(sep, file=sys.__stdout__)
        print(
            f"| {headers['gpu']:<{widths['gpu']}} "
            f"| {headers['tasks']:<{widths['tasks']}} "
            f"| {headers['done']:<{widths['done']}} "
            f"| {headers['next']:<{widths['next']}} |",
            file=sys.__stdout__,
        )
        print(sep, file=sys.__stdout__)
        for row in rows:
            print(
                f"| {row['gpu']:<{widths['gpu']}} "
                f"| {row['tasks']:<{widths['tasks']}} "
                f"| {row['done']:<{widths['done']}} "
                f"| {row['next']:<{widths['next']}} |",
                file=sys.__stdout__,
            )
        print(sep, file=sys.__stdout__)
        print("Live Progress:", file=sys.__stdout__)
        for _ in range(self.world_size):
            print("", file=sys.__stdout__)
        sys.__stdout__.flush()
        self._printed_live_block = True

    def _format_live_line(self, status: dict) -> str:
        completed = int(status.get("completed_subtasks", 0))
        total = int(status.get("assigned_total", 0))
        percent = status.get("percent")
        pct_text = "--%" if percent is None else f"{int(percent):>3d}%"
        current = status.get("current_subtask") or "-"
        next_task = status.get("next_subtask") or "-"
        state_text = status.get("status_text", "running")
        elapsed = int(status.get("elapsed_sec", 0))
        gpu_label = status.get("gpu", "?")
        base = (
            f"GPU{gpu_label} | done {completed}/{total} | {pct_text} | "
            f"cur: {_shorten_text(str(current), 24):<24} | "
            f"next: {_shorten_text(str(next_task), 24):<24} | "
            f"{_shorten_text(str(state_text), 24):<24} | {elapsed:>4}s"
        )
        if status.get("done", False):
            base += " [DONE]"
        return base

    def _render_live_lines(self, force: bool = False) -> None:
        statuses = [self._read_rank_status(rank) for rank in range(self.world_size)]
        lines = [self._format_live_line(status) for status in statuses]

        if self._use_ansi and self._printed_live_block:
            sys.__stdout__.write(f"\x1b[{self.world_size}A")
            for line in lines:
                sys.__stdout__.write("\r\x1b[2K" + line + "\n")
            sys.__stdout__.flush()
        else:
            now = time.time()
            if not force and (now - self._last_non_tty_emit) < self.non_tty_snapshot_sec:
                return
            self._last_non_tty_emit = now
            print("[Live Progress Snapshot]", file=sys.__stdout__)
            print("\n".join(lines), file=sys.__stdout__)
            sys.__stdout__.flush()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._render_live_lines(force=False)
            if self._stop.wait(self.refresh_sec):
                break
        self._render_live_lines(force=True)

    def start(self) -> None:
        self._print_assignment_table()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._use_ansi:
            print("", file=sys.__stdout__)


def maybe_auto_launch_multi_gpu(
    args: argparse.Namespace,
    config: dict,
    rank: int,
    world_size: int,
    output_dir: Path,
) -> bool:
    """
    Auto-launch torch distributed when multiple GPUs are visible.

    Default behavior:
    - single visible GPU -> current process only
    - multiple visible GPUs -> re-launch with torchrun and split by dimensions
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

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        logger.error("Auto multi-GPU torchrun failed with code %d.", result.returncode)
        if args.skip_on_error:
            logger.warning("Skipping VBench evaluation due to auto multi-GPU failure.")
            pd.DataFrame(columns=["video_id", "vbench_temporal_score"]).to_csv(
                output_dir / "vbench_per_video.csv", index=False
            )
            return True
        raise RuntimeError(f"torchrun failed with exit code {result.returncode}")

    return True


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_video_list(metadata_path: Path) -> list:
    """Get list of video paths from metadata."""
    df = pd.read_csv(metadata_path)
    return df.to_dict("records")


def resolve_path(path_str: str | None) -> Path | None:
    """Resolve path against project root when relative."""
    if not path_str:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def infer_group_from_path(video_path: Path, group_names: list[str]) -> str | None:
    """Infer group by folder name or filename."""
    if group_names:
        parts = set(video_path.parts)
        for name in group_names:
            if name in parts:
                return name
        filename = video_path.name
        for name in group_names:
            if name in filename:
                return name
    if len(video_path.parts) > 1:
        return video_path.parent.name
    return None


def load_prompt_map(prompt_path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Load prompt mapping by video_id/stem and ordered fallback list."""
    prompt_df = pd.read_csv(prompt_path, sep=None, engine="python")
    prompt_df.columns = [str(c).strip().lower() for c in prompt_df.columns]

    rename = {}
    if "caption" in prompt_df.columns:
        rename["caption"] = "prompt"
    if "text" in prompt_df.columns:
        rename["text"] = "prompt"
    if "id" in prompt_df.columns:
        rename["id"] = "video_id"
    if "uid" in prompt_df.columns:
        rename["uid"] = "video_id"
    if "filename" in prompt_df.columns:
        rename["filename"] = "video_id"
    if "path" in prompt_df.columns:
        rename["path"] = "video_path"
    if rename:
        prompt_df = prompt_df.rename(columns=rename)

    if "prompt" not in prompt_df.columns and len(prompt_df.columns) >= 1:
        prompt_df = prompt_df.rename(columns={prompt_df.columns[-1]: "prompt"})

    prompt_by_id = {}
    prompt_by_stem = {}
    ordered_prompts = []

    if "prompt" in prompt_df.columns:
        ordered_prompts = [str(v) for v in prompt_df["prompt"].fillna("").tolist()]
        if "video_id" in prompt_df.columns:
            for _, row in prompt_df.iterrows():
                video_id = str(row.get("video_id", "")).strip()
                prompt = str(row.get("prompt", "")).strip()
                if video_id and prompt:
                    prompt_by_id[video_id] = prompt
        if "video_path" in prompt_df.columns:
            for _, row in prompt_df.iterrows():
                video_path = str(row.get("video_path", "")).strip()
                prompt = str(row.get("prompt", "")).strip()
                if video_path and prompt:
                    prompt_by_stem[Path(video_path).stem] = prompt

    return prompt_by_id, prompt_by_stem, ordered_prompts


def build_video_list_from_local_dataset(config: dict) -> list:
    """Build video records from local dataset folders (no preprocess required)."""
    dataset_config = config.get("dataset", {})
    local_video_dir = dataset_config.get("local_video_dir") or dataset_config.get("video_dir")
    local_dir = resolve_path(local_video_dir)
    if local_dir is None or not local_dir.exists():
        raise FileNotFoundError(f"Local video dir not found: {local_video_dir}")

    groups = config.get("groups", [])
    group_names = [g["name"] for g in groups if isinstance(g, dict) and "name" in g]
    default_group = dataset_config.get("default_group")

    prompt_by_id = {}
    prompt_by_stem = {}
    ordered_prompts = []
    prompt_file = dataset_config.get("prompt_file")
    if prompt_file:
        prompt_path = resolve_path(prompt_file)
        if prompt_path and prompt_path.exists():
            prompt_by_id, prompt_by_stem, ordered_prompts = load_prompt_map(prompt_path)
            logger.info(f"Loaded prompts from: {prompt_path} ({len(ordered_prompts)} rows)")
        else:
            logger.warning(f"Prompt file not found, fallback to filename prompt: {prompt_file}")

    video_files = sorted(local_dir.rglob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No mp4 videos found under: {local_dir}")

    records = []
    for idx, video_path in enumerate(video_files):
        candidate_id = video_path.stem

        relative = video_path.relative_to(local_dir)
        group = relative.parts[0] if len(relative.parts) > 1 else None
        if not group or (group_names and group not in group_names):
            group = infer_group_from_path(video_path, group_names)
        if group_names and group not in group_names:
            continue
        if group is None:
            group = default_group or "default"

        prompt = (
            prompt_by_id.get(candidate_id)
            or prompt_by_stem.get(candidate_id)
            or (ordered_prompts[idx] if idx < len(ordered_prompts) else None)
            or candidate_id
        )

        records.append(
            {
                "video_id": candidate_id,
                "group": group,
                "prompt": prompt,
                "video_path": str(video_path),
            }
        )

    if not records:
        raise RuntimeError("No usable videos matched configured groups under local dataset directory.")
    return records


def load_video_records_for_vbench(config: dict, output_dir: Path, paths_config: dict) -> list:
    """Load records from processed metadata / raw metadata / local dataset."""
    processed_metadata = output_dir / paths_config.get("processed_metadata", "processed_metadata.csv")
    metadata_file = output_dir / paths_config.get("metadata_file", "metadata.csv")

    if processed_metadata.exists():
        logger.info(f"Using processed metadata: {processed_metadata}")
        return get_video_list(processed_metadata)

    if metadata_file.exists():
        logger.info(f"Using raw metadata: {metadata_file}")
        return get_video_list(metadata_file)

    logger.info("Metadata not found, building video list from local dataset paths...")
    return build_video_list_from_local_dataset(config)


def ensure_unique_video_ids(video_records: list[dict], config: dict) -> list[dict]:
    """
    Ensure video_id is globally unique across groups.

    Some experiments store videos as {group}/video_000.mp4, causing duplicate
    video_id values across groups. VBench parsing/merge expects unique ids.
    """
    if not video_records:
        return video_records

    groups = config.get("groups", [])
    group_names = [g["name"] for g in groups if isinstance(g, dict) and "name" in g]

    normalized: list[dict] = []
    for record in video_records:
        rec = dict(record)
        video_id = str(rec.get("video_id", "")).strip()
        video_path = str(rec.get("video_path", "")).strip()
        group = str(rec.get("group", "")).strip()

        if not video_id and video_path:
            video_id = Path(video_path).stem
        if not group and video_path:
            group = infer_group_from_path(Path(video_path), group_names) or ""

        rec["video_id"] = video_id
        rec["video_path"] = video_path
        rec["group"] = group
        normalized.append(rec)

    counts = Counter(rec["video_id"] for rec in normalized if rec["video_id"])
    duplicated_ids = [video_id for video_id, count in counts.items() if count > 1]
    if not duplicated_ids:
        return normalized

    logger.warning(
        "Detected duplicate video_id values across groups (%d duplicated ids). "
        "Applying group-prefixed remap for VBench run.",
        len(duplicated_ids),
    )

    ordered_groups: list[str] = []
    for name in group_names:
        if name and name not in ordered_groups:
            ordered_groups.append(name)
    for rec in normalized:
        group_name = str(rec.get("group", "")).strip()
        if group_name and group_name not in ordered_groups:
            ordered_groups.append(group_name)
    group_alias = {name: f"g{idx + 1}" for idx, name in enumerate(ordered_groups)}

    used_ids: set[str] = set()
    remapped_records = 0
    for rec in normalized:
        original_id = rec["video_id"]
        if not original_id:
            original_id = Path(rec["video_path"]).stem if rec.get("video_path") else "video"

        unique_id = original_id
        if counts.get(original_id, 0) > 1:
            group_name = str(rec.get("group") or "").strip()
            alias = group_alias.get(group_name, "g0")
            unique_id = f"{alias}_{original_id}"

        if unique_id in used_ids:
            suffix = 2
            candidate = f"{unique_id}_{suffix}"
            while candidate in used_ids:
                suffix += 1
                candidate = f"{unique_id}_{suffix}"
            unique_id = candidate

        if unique_id != rec.get("video_id"):
            remapped_records += 1
        rec["video_id"] = unique_id
        used_ids.add(unique_id)

    logger.warning("Remapped %d records to unique video_id.", remapped_records)
    if group_alias:
        logger.info("Group alias mapping for duplicate IDs: %s", group_alias)
    return normalized


def copy_outputs_to_frontend(output_dir: Path, paths_config: dict, vbench_output: Path) -> bool:
    """Copy outputs to frontend/public/data and update manifest."""
    try:
        frontend_data_dir = PROJECT_ROOT / "frontend" / "public" / "data"
        frontend_data_dir.mkdir(parents=True, exist_ok=True)

        copied_files = []

        if vbench_output.exists():
            dst = frontend_data_dir / vbench_output.name
            import shutil

            shutil.copy2(vbench_output, dst)
            copied_files.append(vbench_output.name)
            logger.info(f"  Copied: {vbench_output.name}")

        for key, default_name in [
            ("group_summary", "group_summary.csv"),
            ("per_video_metrics", "per_video_metrics.csv"),
            ("experiment_output", None),
        ]:
            file_name = paths_config.get(key, default_name)
            if not file_name:
                continue
            src = output_dir / file_name
            if src.exists():
                dst = frontend_data_dir / src.name
                import shutil

                shutil.copy2(src, dst)
                copied_files.append(src.name)
                logger.info(f"  Copied: {src.name}")

        manifest_path = frontend_data_dir / "manifest.json"
        existing_files = set()
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    existing_files = set(manifest.get("files", []))
            except (json.JSONDecodeError, KeyError):
                existing_files = set()

        discovered_csv_files = {
            p.name for p in frontend_data_dir.glob("*.csv") if p.is_file()
        }
        all_files = sorted(
            {
                file_name
                for file_name in existing_files.union(set(copied_files)).union(discovered_csv_files)
                if str(file_name).endswith(".csv")
            }
        )
        with open(manifest_path, "w") as f:
            json.dump({"files": all_files}, f, indent=2)
        logger.info(f"  Updated manifest.json with {len(all_files)} files")
        return True
    except Exception as e:
        logger.warning(f"Failed to copy outputs to frontend: {e}")
        return False


def ensure_moviepy_editor_compat() -> None:
    """
    Ensure `moviepy.editor` import path works for VBench-Long.

    Some MoviePy versions expose `VideoFileClip` without the legacy
    `moviepy.editor` module path. VBench-Long imports `moviepy.editor`.
    """
    try:
        import moviepy.editor  # noqa: F401
        return
    except ModuleNotFoundError as e:
        if e.name == "moviepy":
            raise ModuleNotFoundError(
                "moviepy is not installed in current Python env. "
                "Please run: python -m pip install \"moviepy<2\""
            ) from e
        if e.name != "moviepy.editor":
            raise

    try:
        import moviepy  # noqa: F401
    except Exception as e:
        raise ModuleNotFoundError(
            "moviepy is not installed in current Python env. "
            "Please run: python -m pip install \"moviepy<2\""
        ) from e

    try:
        from moviepy import VideoFileClip
    except Exception:
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
        except Exception as e:
            raise ModuleNotFoundError(
                "moviepy is installed but VideoFileClip is unavailable. "
                "Try: python -m pip install --upgrade \"moviepy<2\""
            ) from e

    import types

    editor_module = types.ModuleType("moviepy.editor")
    editor_module.VideoFileClip = VideoFileClip
    sys.modules["moviepy.editor"] = editor_module
    logger.info(
        "Applied compatibility shim for `moviepy.editor` "
        "(VBench-Long expects legacy import path)."
    )


def configure_warning_filters() -> None:
    """Suppress noisy but non-actionable warnings from third-party libs."""
    warnings.filterwarnings(
        "ignore",
        message=r"The video decoding and encoding capabilities of torchvision are deprecated.*",
        category=UserWarning,
        module=r"torchvision\.io\._video_deprecation_warning",
    )


def configure_third_party_loggers() -> None:
    """Reduce verbose INFO logs from VBench internals."""
    noisy_loggers = [
        "vbench2_beta_long",
        "vbench2_beta_long.subject_consistency",
        "vbench2_beta_long.background_consistency",
        "vbench2_beta_long.utils",
        "vbench",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def summarize_vbench_stdout(stdout_text: str, subtask: str) -> None:
    """Summarize verbose stdout from VBench/VBench-Long evaluate()."""
    if not stdout_text:
        return

    normalized = stdout_text.replace("\r", "\n")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return

    saved_lines = [line for line in lines if line.startswith("Saved ")]
    other_lines = [line for line in lines if not line.startswith("Saved ")]
    other_lines = [
        line
        for line in other_lines
        if "████" not in line
        and "clips=" not in line
        and not line.startswith("[")
        and not ("%" in line and "|" in line and "/" in line)
    ]

    if saved_lines:
        logger.info(f"[{subtask}] Saved split clips: {len(saved_lines)}")

    max_preview = 4
    for line in other_lines[:max_preview]:
        logger.info(f"[{subtask}] {line}")

    hidden = len(other_lines) - max_preview
    if hidden > 0:
        logger.info(f"[{subtask}] ... {hidden} more log lines suppressed")


class StdoutMonitor:
    """Thread-safe stdout monitor for VBench evaluation output."""

    def __init__(self):
        self._lock = threading.Lock()
        self._parts: list[str] = []
        self._line_buffer = ""
        self._saved_count = 0
        self._saved_video_ids: set[str] = set()
        self._line_count = 0
        self._chars = 0
        self._latest_percent: int | None = None
        self._latest_done: int | None = None
        self._latest_total: int | None = None

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            self._parts.append(text)
            self._chars += len(text)
            for match in re.finditer(r"(\d{1,3})%\|", text):
                percent = int(match.group(1))
                if 0 <= percent <= 100:
                    self._latest_percent = percent
            if "%|" in text:
                for match in re.finditer(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)", text):
                    done = int(match.group(1))
                    total = int(match.group(2))
                    if total > 0 and 0 <= done <= total:
                        self._latest_done = done
                        self._latest_total = total
                        self._latest_percent = int(done * 100 / total)
            self._line_buffer += text
            lines = self._line_buffer.splitlines(keepends=True)
            remainder = ""
            if lines and not lines[-1].endswith(("\n", "\r")):
                remainder = lines.pop()
            self._line_buffer = remainder

            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                self._line_count += 1
                if line_stripped.startswith("Saved "):
                    self._saved_count += 1
                    saved_video_id = _extract_saved_video_id(line_stripped)
                    if saved_video_id:
                        self._saved_video_ids.add(saved_video_id)
                match = re.search(r"(\d{1,3})%\|", line_stripped)
                if match:
                    percent = int(match.group(1))
                    if 0 <= percent <= 100:
                        self._latest_percent = percent
                if "%|" in line_stripped:
                    for ratio_match in re.finditer(r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)", line_stripped):
                        done = int(ratio_match.group(1))
                        total = int(ratio_match.group(2))
                        if total > 0 and 0 <= done <= total:
                            self._latest_done = done
                            self._latest_total = total
                            self._latest_percent = int(done * 100 / total)
        return len(text)

    def flush(self) -> None:
        return

    def snapshot(self) -> tuple[int, int, int, int | None, int, int | None, int | None]:
        with self._lock:
            return (
                self._saved_count,
                self._line_count,
                self._chars,
                self._latest_percent,
                len(self._saved_video_ids),
                self._latest_done,
                self._latest_total,
            )

    def getvalue(self) -> str:
        with self._lock:
            tail = self._line_buffer
            return "".join(self._parts) + tail


def _render_bar(step: int, width: int = 18) -> str:
    filled = step % (width + 1)
    return f"{'█' * filled}{'·' * (width - filled)}"


def _render_percent_bar(percent: int, width: int = 18) -> str:
    clamped = max(0, min(100, percent))
    filled = int(round(clamped * width / 100))
    return f"{'█' * filled}{'·' * (width - filled)}"


def _extract_saved_video_id(saved_line: str) -> str | None:
    """
    Extract video id from VBench split-clip log line, e.g.:
    Saved .../split_clip/<video_id>/<video_id>_000.mp4
    """
    match = re.search(r"split_clip/([^/]+)/", saved_line)
    if match:
        return match.group(1)
    return None


def run_callable_with_progress(
    task_fn,
    title: str,
    prefix: str = "",
    refresh_sec: float = 1.0,
    status_mode: str = "events",
    enable_live: bool = True,
    expected_units: int | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[str, int]:
    """
    Run a callable with one-line progress display and captured output.

    Returns:
        Tuple of (captured_stdout_and_stderr, saved_clip_count)
    """
    monitor = StdoutMonitor()
    worker_error: list[Exception] = []

    def _worker():
        try:
            with contextlib.redirect_stdout(monitor), contextlib.redirect_stderr(monitor):
                task_fn()
        except Exception as e:
            worker_error.append(e)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    start = time.time()
    tick = 0

    while thread.is_alive():
        elapsed = int(time.time() - start)
        (
            saved_count,
            line_count,
            _,
            latest_percent,
            saved_video_count,
            latest_done,
            latest_total,
        ) = monitor.snapshot()
        display_percent = latest_percent
        if status_mode == "clips" and expected_units is not None and expected_units > 0:
            # For preprocessing, use observed completed videos as authoritative progress.
            # This avoids premature 100% from third-party tqdm outputs.
            display_percent = min(99, int(saved_video_count * 100 / expected_units))

        # Never show 100% until the worker thread has actually finished.
        if display_percent is not None and display_percent >= 100:
            display_percent = 99

        if status_mode == "clips":
            if expected_units is not None and expected_units > 0:
                status_text = f"videos={saved_video_count}/{expected_units}, clips={saved_count}"
            else:
                status_text = f"clips={saved_count}"
        else:
            if latest_done is not None and latest_total is not None and latest_total > 0:
                status_text = f"progress={latest_done}/{latest_total}"
            else:
                status_text = f"events={line_count}"

        if progress_callback is not None:
            progress_callback(
                {
                    "title": title,
                    "percent": display_percent,
                    "status_text": status_text,
                    "elapsed_sec": elapsed,
                    "done": False,
                }
            )

        if enable_live:
            if display_percent is not None:
                bar = _render_percent_bar(display_percent)
                pct_text = f"{display_percent:>3d}%"
            else:
                bar = _render_bar(tick)
                pct_text = " --%"
            line = f"\r{prefix}{title:<24} | {bar} | {pct_text} | {status_text} | {elapsed:>4}s"
            print(line, end="", flush=True, file=sys.__stdout__)
            tick += 1
        thread.join(timeout=refresh_sec)

    thread.join()
    elapsed = int(time.time() - start)
    (
        saved_count,
        line_count,
        _,
        latest_percent,
        saved_video_count,
        latest_done,
        latest_total,
    ) = monitor.snapshot()
    final_percent = 100 if latest_percent is None else max(100, latest_percent)
    final_bar = _render_percent_bar(final_percent)
    if status_mode == "clips":
        if expected_units is not None and expected_units > 0:
            final_status = f"videos={saved_video_count}/{expected_units}, clips={saved_count}"
        else:
            final_status = f"clips={saved_count}"
    else:
        if latest_done is not None and latest_total is not None and latest_total > 0:
            final_status = f"progress={latest_done}/{latest_total}"
        else:
            final_status = f"events={line_count}"
    if progress_callback is not None:
        progress_callback(
            {
                "title": title,
                "percent": 100,
                "status_text": final_status,
                "elapsed_sec": elapsed,
                "done": True,
            }
        )
    if enable_live:
        print(
            f"\r{prefix}{title:<24} | {final_bar} | 100% | {final_status} | {elapsed:>4}s done",
            file=sys.__stdout__,
        )

    if worker_error:
        raise worker_error[0]
    return monitor.getvalue(), saved_count


def run_evaluate_with_progress(
    vbench: Any,
    eval_kwargs: dict,
    subtask: str,
    subtask_index: int,
    subtask_total: int,
    refresh_sec: float = 1.0,
    enable_live: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[str, int]:
    """
    Run one VBench subtask with a lightweight manual progress bar.
    """
    return run_callable_with_progress(
        task_fn=lambda: vbench.evaluate(**eval_kwargs),
        title=subtask,
        prefix=f"[{subtask_index}/{subtask_total}] ",
        refresh_sec=refresh_sec,
        status_mode="events",
        enable_live=enable_live,
        progress_callback=progress_callback,
    )


def get_input_video_files(video_dir: Path) -> list[Path]:
    """List direct input videos under the VBench input directory."""
    valid_suffixes = {".mp4", ".avi", ".mov"}
    return sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_suffixes
    )


def are_split_clips_ready(video_dir: Path, input_videos: list[Path]) -> bool:
    """
    Check whether split clips are already prepared for all input videos.
    """
    split_dir = video_dir / "split_clip"
    if not split_dir.exists() or not split_dir.is_dir():
        return False
    if not input_videos:
        return False

    valid_suffixes = {".mp4", ".avi", ".mov"}
    for video_path in input_videos:
        clip_dir = split_dir / video_path.stem
        if not clip_dir.exists() or not clip_dir.is_dir():
            return False
        has_clip = any(
            clip_file.is_file() and clip_file.suffix.lower() in valid_suffixes
            for clip_file in clip_dir.iterdir()
        )
        if not has_clip:
            return False
    return True


def use_vbench_long(config: dict) -> bool:
    """Whether to run VBench-Long backend."""
    vbench_config = config.get("metrics", {}).get("vbench", {})
    backend = str(vbench_config.get("backend", "vbench")).lower()
    return bool(vbench_config.get("use_long", False) or backend in {"long", "vbench_long"})


def ensure_clip_dependency(subtasks: list[str]) -> None:
    """Ensure CLIP Python package is available for CLIP-based dimensions."""
    extra_clip_dims = {
        "appearance_style",
        "human_action",
        "overall_consistency",
        "temporal_style",
    }
    required = sorted(set(subtasks).intersection(CLIP_REQUIRED_DIMENSIONS.union(extra_clip_dims)))
    if not required:
        return

    try:
        import clip  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name == "pkg_resources":
            raise ModuleNotFoundError(
                "OpenAI CLIP requires `pkg_resources` from `setuptools`. "
                "Install it in current env: `uv pip install -U setuptools` "
                "or `python -m pip install -U setuptools`."
            ) from e
        if e.name != "clip":
            raise ModuleNotFoundError(
                f"`clip` import failed due to missing dependency `{e.name}`. "
                "Please install/repair CLIP environment first. "
                "Suggested order: `uv pip install -U setuptools` then "
                "`uv pip install openai-clip`."
            ) from e
        dims = ", ".join(required)
        raise ModuleNotFoundError(
            "`clip` module is required for VBench subtasks: "
            f"{dims}. Install it in the same environment, e.g. "
            "`uv pip install openai-clip` or "
            "`python -m pip install openai-clip`. "
            "If your mirror does not provide it, use "
            "`python -m pip install git+https://github.com/openai/CLIP.git`."
        ) from e


def ensure_pyiqa_dependency(subtasks: list[str]) -> None:
    """Ensure pyiqa is installed when imaging-quality subtask is enabled."""
    required = sorted(set(subtasks).intersection(PYIQA_REQUIRED_DIMENSIONS))
    if not required:
        return
    try:
        import pyiqa  # noqa: F401
    except Exception as e:
        dims = ", ".join(required)
        raise ModuleNotFoundError(
            "`pyiqa` is required for VBench subtasks: "
            f"{dims}. Install it in current env, e.g. `uv pip install pyiqa`, "
            "and ensure NumPy compatibility (recommend numpy<2 for older pyiqa stacks)."
        ) from e


def ensure_extended_dimension_dependencies(subtasks: list[str]) -> None:
    """Ensure optional heavy dependencies exist for full 16-dimension run."""
    detectron_dims = {
        "object_class",
        "multiple_objects",
        "spatial_relationship",
        "color",
    }
    if set(subtasks) & detectron_dims:
        try:
            import detectron2  # noqa: F401
        except Exception as e:
            missing = ", ".join(sorted(set(subtasks) & detectron_dims))
            raise ModuleNotFoundError(
                "detectron2 is required for VBench subtasks: "
                f"{missing}. Install with:\n"
                "  uv pip install --no-build-isolation "
                "\"detectron2 @ git+https://github.com/facebookresearch/detectron2.git\""
            ) from e

    if "scene" in set(subtasks):
        try:
            import fairscale  # noqa: F401
        except Exception as e:
            raise ModuleNotFoundError(
                "fairscale is required for VBench `scene` subtask. Install with:\n"
                "  uv pip install fairscale"
            ) from e


def ensure_pyav_dependency(long_mode: bool) -> None:
    """Ensure PyAV is available for VBench-Long preprocessing video writes."""
    if not long_mode:
        return
    try:
        import av  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name != "av":
            raise
        raise ModuleNotFoundError(
            "PyAV is required by torchvision video IO in VBench-Long. "
            "Install in current env: `uv pip install av` "
            "or `python -m pip install av`."
        ) from e


def get_vbench_subtasks(config: dict) -> list:
    """Get configured subtasks with mode-specific defaults."""
    vbench_config = config.get("metrics", {}).get("vbench", {})
    if "subtasks" in vbench_config:
        configured = normalize_subtasks(vbench_config["subtasks"])
        if configured:
            unknown = sorted(set(configured) - LONG_DIMENSION_SET)
            if unknown:
                logger.warning(
                    "Configured subtasks include unregistered dimensions: %s",
                    unknown,
                )
            return configured

    if use_vbench_long(config):
        profile = vbench_config.get("dimension_profile", "long_6")
        subtasks = default_long_subtasks(profile=profile)
        if str(profile).strip().lower() in {"long_16", "16", "16d", "full", "full_16"}:
            logger.info("Using VBench-Long full 16-dimension profile.")
            logger.debug("Supported long subtasks: %s", supported_long_subtasks())
        else:
            logger.info("Using VBench-Long recommended 6-dimension profile.")
        return subtasks

    return [
        "temporal_flickering",
        "motion_smoothness",
    ]


def normalize_prompt_text(text: str) -> str:
    """Normalize prompt for robust key lookup."""
    return " ".join(str(text or "").strip().lower().split())


def simplify_prompt_text(text: str) -> str:
    """Normalize prompt and remove punctuation for fallback key lookup."""
    normalized = normalize_prompt_text(text)
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    return " ".join(normalized.split())


def tokenize_prompt_words(text: str) -> list[str]:
    """Tokenize lowercase words from prompt."""
    return re.findall(r"[a-z]+", str(text or "").lower())


def extract_object_token(text: str, default: str = "object") -> str:
    """Extract one object-like token from free-form text."""
    tokens = [
        token
        for token in tokenize_prompt_words(text)
        if token not in PROMPT_STOPWORDS and token not in COLOR_WORDS
    ]
    return tokens[-1] if tokens else default


def infer_auxiliary_from_prompt(dimension: str, prompt_text: str) -> dict | None:
    """Infer missing auxiliary info for long custom input dimensions."""
    prompt_text = str(prompt_text or "")
    prompt_simple = simplify_prompt_text(prompt_text)

    if dimension == "appearance_style":
        style_match = re.search(
            r"(?:in the style of|style of)\s+([a-z0-9\s-]+)",
            prompt_simple,
        )
        if style_match:
            style = style_match.group(1).strip()
        else:
            style_tail = re.search(r"([a-z0-9\s-]+ style)\b", prompt_simple)
            style = style_tail.group(1).strip() if style_tail else ""
        if not style:
            style = "realistic style"
        return {"appearance_style": style}

    if dimension == "scene":
        first_clause = re.split(r"[,.;]", prompt_simple, maxsplit=1)[0]
        return {"scene": extract_object_token(first_clause, default="outdoor")}

    if dimension == "object_class":
        first_clause = re.split(r"[,.;]", prompt_simple, maxsplit=1)[0]
        return {"object": extract_object_token(first_clause, default="person")}

    if dimension == "multiple_objects":
        pattern = re.search(
            r"(?:^|\b)(.+?)\s+and\s+(.+?)(?:$|,|;|\.)",
            prompt_simple,
        )
        if pattern:
            obj_a = extract_object_token(pattern.group(1), default="person")
            obj_b = extract_object_token(pattern.group(2), default="object")
        else:
            words = [
                token
                for token in tokenize_prompt_words(prompt_simple)
                if token not in PROMPT_STOPWORDS and token not in COLOR_WORDS
            ]
            obj_a = words[0] if len(words) >= 1 else "person"
            obj_b = words[1] if len(words) >= 2 else "object"
        return {"object": f"{obj_a} and {obj_b}"}

    if dimension == "spatial_relationship":
        for relation in SPATIAL_RELATIONS:
            if relation in prompt_simple:
                left, right = prompt_simple.split(relation, 1)
                obj_a = extract_object_token(left, default="object")
                obj_b = extract_object_token(right, default="object")
                return {
                    "spatial_relationship": {
                        "object_a": obj_a,
                        "object_b": obj_b,
                        "relationship": relation,
                    }
                }
        return {
            "spatial_relationship": {
                "object_a": extract_object_token(prompt_simple, default="object"),
                "object_b": "object",
                "relationship": "on the left of",
            }
        }

    if dimension == "color":
        for color in COLOR_WORDS:
            if re.search(rf"\b{re.escape(color)}\b", prompt_simple):
                normalized_color = "gray" if color == "grey" else color
                return {"color": normalized_color}
        return {"color": "red"}

    return None


def build_auxiliary_prompt_lookup(
    full_info_path: Path,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    """
    Build auxiliary lookup maps from official VBench full-info.

    Returns:
        (exact_prompt_lookup, simplified_prompt_lookup)
    """
    if not full_info_path.exists():
        return {}, {}

    with open(full_info_path, "r", encoding="utf-8") as f:
        full_info_data = json.load(f)

    exact: dict[str, dict[str, dict]] = {dim: {} for dim in AUXILIARY_REQUIRED_LONG_DIMS}
    simple: dict[str, dict[str, dict]] = {dim: {} for dim in AUXILIARY_REQUIRED_LONG_DIMS}

    for item in full_info_data:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt_en", "")).strip()
        aux_all = item.get("auxiliary_info")
        if not prompt or not isinstance(aux_all, dict):
            continue

        norm_key = normalize_prompt_text(prompt)
        simple_key = simplify_prompt_text(prompt)
        for dim in AUXILIARY_REQUIRED_LONG_DIMS:
            payload = aux_all.get(dim)
            if not isinstance(payload, dict):
                continue
            exact[dim].setdefault(norm_key, deepcopy(payload))
            simple[dim].setdefault(simple_key, deepcopy(payload))

    return exact, simple


def resolve_auxiliary_payload(
    dimension: str,
    prompt_text: str,
    exact_lookup: dict[str, dict[str, dict]],
    simple_lookup: dict[str, dict[str, dict]],
) -> tuple[dict | None, str]:
    """Resolve auxiliary payload by official lookup first, then heuristic fallback."""
    norm_key = normalize_prompt_text(prompt_text)
    simple_key = simplify_prompt_text(prompt_text)

    by_exact = exact_lookup.get(dimension, {})
    by_simple = simple_lookup.get(dimension, {})
    if norm_key in by_exact:
        return deepcopy(by_exact[norm_key]), "exact"
    if simple_key in by_simple:
        return deepcopy(by_simple[simple_key]), "simplified"

    inferred = infer_auxiliary_from_prompt(dimension, prompt_text)
    if inferred is not None:
        return inferred, "heuristic"
    return None, "missing"


def patch_long_custom_full_info_builder(
    vbench_runner: Any,
    video_records: list[dict],
    exact_lookup: dict[str, dict[str, dict]],
    simple_lookup: dict[str, dict[str, dict]],
) -> None:
    """
    Patch VBench-Long `build_full_info_json` for long_custom_input to support all 16 dimensions.

    Upstream `long_custom_input` uses video-id as prompt and omits auxiliary_info, which makes
    6 dimensions fail. This patch keeps custom mode but injects:
      - true `prompt_en` from metadata
      - `auxiliary_info` for required dimensions
    """
    original_build = vbench_runner.build_full_info_json
    prompt_by_video_id = {
        str(record.get("video_id", "")): str(record.get("prompt", "")).strip()
        for record in video_records
    }

    def _patched_build_full_info_json(
        self,
        videos_path,
        name,
        dimension_list,
        prompt_list=[],
        special_str="",
        verbose=False,
        mode="vbench_standard",
        **kwargs,
    ):
        if str(mode).strip().lower() != "long_custom_input":
            return original_build(
                videos_path,
                name,
                dimension_list,
                prompt_list,
                special_str,
                verbose,
                mode,
                **kwargs,
            )

        split_root = Path(videos_path) / "split_clip"
        if not split_root.exists():
            return original_build(
                videos_path,
                name,
                dimension_list,
                prompt_list,
                special_str,
                verbose,
                mode,
                **kwargs,
            )

        valid_suffixes = {".mp4", ".avi", ".mov"}
        clips_by_video: dict[str, list[str]] = {}
        for folder in sorted(split_root.iterdir()):
            if not folder.is_dir():
                continue
            base_video_id = folder.name.split("-Scene")[0]
            clip_files = sorted(
                str(path)
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in valid_suffixes
            )
            if clip_files:
                clips_by_video.setdefault(base_video_id, []).extend(clip_files)

        if not clips_by_video:
            raise RuntimeError(f"No split clips found under: {split_root}")

        aux_dims = sorted(set(dimension_list) & AUXILIARY_REQUIRED_LONG_DIMS)
        source_counter: Counter[str] = Counter()
        full_info_payload = []
        for video_id in sorted(clips_by_video.keys()):
            prompt_text = prompt_by_video_id.get(video_id) or video_id
            item = {
                "prompt_en": prompt_text,
                "dimension": list(dimension_list),
                "video_list": clips_by_video[video_id],
            }
            if aux_dims:
                aux_map: dict[str, dict] = {}
                for dim in aux_dims:
                    aux_payload, source = resolve_auxiliary_payload(
                        dimension=dim,
                        prompt_text=prompt_text,
                        exact_lookup=exact_lookup,
                        simple_lookup=simple_lookup,
                    )
                    if aux_payload is None:
                        raise ValueError(
                            f"Cannot build auxiliary_info for dimension `{dim}` and prompt: "
                            f"{prompt_text!r} (video_id={video_id})."
                        )
                    aux_map[dim] = aux_payload
                    source_counter[source] += 1
                item["auxiliary_info"] = aux_map
            full_info_payload.append(item)

        output_path = Path(self.output_path) / f"{name}_full_info.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_info_payload, f, ensure_ascii=False, indent=2)

        if aux_dims:
            logger.info(
                "[%s] auxiliary_info ready for %d videos (%s)",
                name,
                len(full_info_payload),
                ", ".join(f"{k}={v}" for k, v in sorted(source_counter.items())),
            )

        print(f"Evaluation meta data saved to {output_path}")
        return str(output_path)

    vbench_runner.build_full_info_json = _patched_build_full_info_json.__get__(
        vbench_runner, type(vbench_runner)
    )


def resolve_video_id(video_path: str, valid_video_ids: set[str]) -> str:
    """
    Resolve a VBench/VBench-Long output path to original video_id.

    Long-mode outputs are often clip-level paths, so we attempt multiple candidates.
    """
    path = Path(video_path)
    stem = path.stem
    candidates = [stem, path.parent.name]

    if stem.rsplit("_", 1)[-1].isdigit():
        candidates.append(stem.rsplit("_", 1)[0])
    if "-Scene" in stem:
        candidates.append(stem.split("-Scene")[0])
    if "-Scene" in path.parent.name:
        candidates.append(path.parent.name.split("-Scene")[0])

    for candidate in candidates:
        if candidate in valid_video_ids:
            return candidate

    return stem


def apply_long_consistency_prefix_fallback(
    parsed_items: list[dict],
    unresolved_items: list[tuple[str, float]],
    subtask: str,
    valid_video_ids: set[str],
) -> list[dict]:
    """
    Fallback for VBench-Long subject/background id truncation.

    Some upstream outputs collapse video ids to coarse prefixes
    (e.g. frame/head/osc/stable). When this happens, expand prefix
    scores to all matching video_ids in current dataset.
    """
    if subtask not in {"subject_consistency", "background_consistency"}:
        return parsed_items
    if not unresolved_items:
        return parsed_items

    prefix_to_video_ids: dict[str, list[str]] = {}
    for video_id in sorted(valid_video_ids):
        prefix = video_id.split("_", 1)[0].lower()
        prefix_to_video_ids.setdefault(prefix, []).append(video_id)

    prefix_scores: dict[str, list[float]] = {}
    for video_path, score in unresolved_items:
        stem = Path(str(video_path)).stem.lower()
        prefix = stem.split("_", 1)[0]
        if prefix in prefix_to_video_ids:
            prefix_scores.setdefault(prefix, []).append(float(score))

    if not prefix_scores:
        return parsed_items

    existing_video_ids = {item["video_id"] for item in parsed_items}
    added = 0
    for prefix, scores in prefix_scores.items():
        mean_score = float(sum(scores) / len(scores))
        for video_id in prefix_to_video_ids[prefix]:
            if video_id in existing_video_ids:
                continue
            parsed_items.append(
                {
                    "video_id": video_id,
                    "subtask": subtask,
                    "score": mean_score,
                }
            )
            existing_video_ids.add(video_id)
            added += 1

    logger.warning(
        f"[{subtask}] fallback prefix mapping applied: added {added} videos "
        f"from prefixes {sorted(prefix_scores.keys())}"
    )
    return parsed_items


def extract_subtask_scores(
    dimension_data,
    subtask: str,
    valid_video_ids: set[str],
    long_mode: bool = False,
) -> list[dict]:
    """Extract per-video scores from one subtask result blob."""
    if not isinstance(dimension_data, list):
        return []

    per_video_items = []
    if len(dimension_data) >= 2 and isinstance(dimension_data[1], list):
        per_video_items = dimension_data[1]
    elif isinstance(dimension_data, list):
        per_video_items = [item for item in dimension_data if isinstance(item, dict)]

    parsed_items = []
    unresolved_count = 0
    unresolved_items: list[tuple[str, float]] = []
    for item in per_video_items:
        if not isinstance(item, dict):
            continue
        video_path = item.get("video_path", item.get("video_name", ""))
        score = item.get("video_results", item.get("score"))
        if not video_path or score is None:
            continue
        resolved_video_id = resolve_video_id(str(video_path), valid_video_ids)
        if resolved_video_id not in valid_video_ids:
            unresolved_count += 1
            unresolved_items.append((str(video_path), float(score)))
            continue
        parsed_items.append(
            {
                "video_id": resolved_video_id,
                "subtask": subtask,
                "score": float(score),
            }
        )

    if long_mode:
        parsed_items = apply_long_consistency_prefix_fallback(
            parsed_items=parsed_items,
            unresolved_items=unresolved_items,
            subtask=subtask,
            valid_video_ids=valid_video_ids,
        )

    if unresolved_count > 0:
        logger.info(
            f"[{subtask}] skipped {unresolved_count} unresolved video ids from raw VBench output"
        )

    if not long_mode:
        return parsed_items

    # VBench-Long can emit clip-level results; aggregate to per-original-video.
    if not parsed_items:
        return parsed_items

    df_long = pd.DataFrame(parsed_items)
    agg = (
        df_long.groupby(["video_id", "subtask"], as_index=False)["score"]
        .mean()
        .round(6)
    )
    return agg.to_dict("records")


def run_vbench_evaluation(
    video_records: list,
    output_dir: Path,
    config: dict,
    device: str = "cuda",
    rank: int = 0,
    world_size: int = 1,
    barrier_fn: Callable[[], None] | None = None,
    subtasks_override: list[str] | None = None,
    progress_total_subtasks: int | None = None,
    progress_reporter: RankProgressReporter | None = None,
) -> pd.DataFrame:
    """
    Run VBench evaluation on videos using official implementation.

    Args:
        video_records: List of dicts with video_id, group, video_path, prompt
        output_dir: Directory to save results
        config: VBench configuration from eval.yaml
        device: Device to run on

    Returns:
        DataFrame with per-video VBench scores
    """
    setup_vbench_path()
    if barrier_fn is None:
        barrier_fn = lambda: None

    long_mode = use_vbench_long(config)

    # VBench configuration
    vbench_config = config.get("metrics", {}).get("vbench", {})
    subtasks = list(subtasks_override) if subtasks_override is not None else get_vbench_subtasks(config)
    eval_mode = vbench_config.get("mode", "long_custom_input" if long_mode else "custom_input")
    total_subtasks = len(subtasks)
    display_total_subtasks = progress_total_subtasks or total_subtasks

    ensure_pyav_dependency(long_mode=long_mode)
    ensure_clip_dependency(subtasks=subtasks)
    ensure_pyiqa_dependency(subtasks=subtasks)
    ensure_extended_dimension_dependencies(subtasks=subtasks)

    try:
        if long_mode:
            ensure_moviepy_editor_compat()
            from vbench2_beta_long import VBenchLong as VBenchRunner
        else:
            from vbench import VBench as VBenchRunner
    except ImportError as e:
        logger.error(f"Failed to import VBench backend: {e}")
        logger.error("Please ensure VBench dependencies are installed:")
        logger.error("  pip install -r third_party/VBench/requirements.txt")
        raise

    long_kwargs = {
        "use_semantic_splitting": bool(vbench_config.get("use_semantic_splitting", False)),
        "clip_length_config": vbench_config.get("clip_length_config", "clip_length_mix.yaml"),
        "threshold": float(vbench_config.get("scene_threshold", 35.0)),
        "static_filter_flag": bool(vbench_config.get("static_filter_flag", False)),
        "sb_clip2clip_feat_extractor": vbench_config.get("sb_clip2clip_feat_extractor", "dino"),
        "bg_clip2clip_feat_extractor": vbench_config.get("bg_clip2clip_feat_extractor", "clip"),
        "w_inclip": float(vbench_config.get("w_inclip", 1.0)),
        "w_clip2clip": float(vbench_config.get("w_clip2clip", 0.0)),
        "imaging_quality_preprocessing_mode": vbench_config.get(
            "imaging_quality_preprocessing_mode", "longer"
        ),
        "dev_flag": bool(vbench_config.get("dev_flag", False)),
        "num_of_samples_per_prompt": int(vbench_config.get("num_of_samples_per_prompt", 5)),
        "slow_fast_eval_config": str(
            vbench_config.get(
                "slow_fast_eval_config",
                VBENCH_ROOT / "vbench2_beta_long" / "configs" / "slow_fast_params.yaml",
            )
        ),
        "sb_mapping_file_path": str(
            vbench_config.get(
                "sb_mapping_file_path",
                VBENCH_ROOT / "vbench2_beta_long" / "configs" / "subject_mapping_table.yaml",
            )
        ),
        "bg_mapping_file_path": str(
            vbench_config.get(
                "bg_mapping_file_path",
                VBENCH_ROOT / "vbench2_beta_long" / "configs" / "background_mapping_table.yaml",
            )
        ),
    }

    # Results directory for VBench output (must be absolute path)
    results_dir = (output_dir / "vbench_results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # VBench requires full_info_dir to be a JSON file path (not directory)
    if long_mode:
        vbench_full_info = VBENCH_ROOT / "vbench2_beta_long" / "VBench_full_info.json"
    else:
        vbench_full_info = VBENCH_ROOT / "vbench" / "VBench_full_info.json"

    if not vbench_full_info.exists():
        logger.warning(f"VBench full info not found: {vbench_full_info}")
        logger.warning("Will use custom_input mode without full_info")
        vbench_full_info_str = str(results_dir / "dummy_info.json")
        # Create a dummy empty JSON for custom_input mode
        with open(vbench_full_info_str, "w") as f:
            f.write("[]")
    else:
        vbench_full_info_str = str(vbench_full_info.resolve())
    vbench_full_info_path = Path(vbench_full_info_str)

    # Prepare video paths for VBench
    # VBench expects a specific format: list of video paths or a directory
    video_map = {r["video_id"]: r["video_path"] for r in video_records}
    valid_video_ids = set(video_map.keys())

    # Create a directory with symlinks to all videos for VBench
    # VBench custom_input mode expects a directory of videos
    video_dir = results_dir / "input_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for video_id, vp in video_map.items():
        src = Path(vp)
        suffix = src.suffix if src.suffix else ".mp4"
        dst = video_dir / f"{video_id}{suffix}"
        if not dst.exists() and src.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                # If symlink fails, try copying
                import shutil
                shutil.copy2(src, dst)
    video_dir_str = str(video_dir.resolve())

    results = []

    # Initialize VBench
    try:
        # VBench requires full_info_dir (JSON file path) and output_path (directory)
        vbench = VBenchRunner(
            device=device,
            full_info_dir=vbench_full_info_str,
            output_path=str(results_dir),
        )
    except Exception as e:
        logger.error(f"Failed to initialize VBench: {e}")
        logger.error("This may be due to missing model weights.")
        logger.error("Please check VBench documentation for weight download instructions.")
        raise

    if long_mode and str(eval_mode).strip().lower() == "long_custom_input":
        exact_lookup, simple_lookup = build_auxiliary_prompt_lookup(vbench_full_info_path)
        patch_long_custom_full_info_builder(
            vbench_runner=vbench,
            video_records=video_records,
            exact_lookup=exact_lookup,
            simple_lookup=simple_lookup,
        )

    if long_mode:
        input_videos = get_input_video_files(video_dir)
        split_ready = are_split_clips_ready(video_dir, input_videos)
        if split_ready:
            if rank == 0:
                logger.info(
                    "Detected existing split clips for all videos; skip preprocessing and reuse cache."
                )
        else:
            if rank == 0:
                logger.info("Preprocessing videos into long clips once before subtasks...")
                preprocess_kwargs = {
                    "videos_path": video_dir_str,
                    "mode": eval_mode,
                    **long_kwargs,
                }
                if progress_reporter is not None:
                    progress_reporter.start_task("preprocess_clips")
                try:
                    preprocess_stdout, saved_count = run_callable_with_progress(
                        task_fn=lambda: vbench.preprocess(**preprocess_kwargs),
                        title="preprocess_clips",
                        prefix=f"[0/{display_total_subtasks}] ",
                        refresh_sec=1.0,
                        status_mode="clips",
                        enable_live=(world_size <= 1),
                        expected_units=len(video_records),
                        progress_callback=(
                            (lambda payload: progress_reporter.update_live(
                                percent=payload.get("percent"),
                                status_text=str(payload.get("status_text", "running")),
                                elapsed_sec=int(payload.get("elapsed_sec", 0)),
                            ))
                            if progress_reporter is not None
                            else None
                        ),
                    )
                    if progress_reporter is not None:
                        progress_reporter.finish_task(success=True, count_completion=False)
                except Exception as exc:
                    if progress_reporter is not None:
                        progress_reporter.finish_task(
                            success=False,
                            error=str(exc),
                            count_completion=False,
                        )
                    raise
                if saved_count > 0:
                    logger.info(f"[preprocess] Saved split clips: {saved_count}")
                summarize_vbench_stdout(preprocess_stdout, "preprocess")
            else:
                if progress_reporter is not None:
                    progress_reporter.start_task(
                        "preprocess_clips",
                        status_text="waiting_rank0_preprocess",
                    )
            if world_size > 1:
                barrier_fn()
                if rank != 0 and progress_reporter is not None:
                    progress_reporter.finish_task(success=True, count_completion=False)

        # VBench-Long evaluate() always calls preprocess(). We already did one-time
        # preprocessing above, so disable repeated preprocessing for each subtask.
        def _skip_preprocess(*args, **kwargs):
            return None

        vbench.preprocess = _skip_preprocess

    # Run evaluation for each subtask
    backend_name = "VBench-Long" if long_mode else "VBench"
    if rank == 0:
        logger.info(f"Running {backend_name} evaluation with subtasks: {subtasks}")

    for idx, subtask in enumerate(subtasks, start=1):
        if rank == 0:
            logger.info(f"Evaluating subtask: {subtask}")
        try:
            if progress_reporter is not None:
                progress_reporter.start_task(subtask)
            eval_kwargs = {
                "videos_path": video_dir_str,
                "name": subtask,
                "dimension_list": [subtask],
                "local": True,
                "read_frame": bool(vbench_config.get("read_frame", False)),
                "mode": eval_mode,
            }
            if long_mode:
                eval_kwargs.update(long_kwargs)
            stdout_text, saved_count = run_evaluate_with_progress(
                vbench=vbench,
                eval_kwargs=eval_kwargs,
                subtask=subtask,
                subtask_index=idx,
                subtask_total=display_total_subtasks,
                refresh_sec=1.0,
                enable_live=(rank == 0 and world_size <= 1),
                progress_callback=(
                    (lambda payload: progress_reporter.update_live(
                        percent=payload.get("percent"),
                        status_text=str(payload.get("status_text", "running")),
                        elapsed_sec=int(payload.get("elapsed_sec", 0)),
                    ))
                    if progress_reporter is not None
                    else None
                ),
            )
            if rank == 0 and saved_count > 0:
                logger.info(f"[{subtask}] Saved split clips: {saved_count}")
            if rank == 0:
                summarize_vbench_stdout(stdout_text, subtask)

            # VBench saves results to {output_path}/{name}_eval_results.json
            result_file = results_dir / f"{subtask}_eval_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    subtask_data = json.load(f)
                # VBench output format: {dimension: [avg_score, [{"video_path": ..., "video_results": ...}, ...]]}
                if subtask in subtask_data:
                    count_before = len(results)
                    dimension_data = subtask_data[subtask]
                    extracted = extract_subtask_scores(
                        dimension_data=dimension_data,
                        subtask=subtask,
                        valid_video_ids=valid_video_ids,
                        long_mode=long_mode,
                    )
                    results.extend(extracted)

                    if rank == 0:
                        logger.info(f"Parsed {len(results) - count_before} results for {subtask}")
            else:
                logger.warning(f"[rank {rank}] Result file not found: {result_file}")
            if progress_reporter is not None:
                progress_reporter.finish_task(success=True)
        except Exception as e:
            if progress_reporter is not None:
                progress_reporter.finish_task(success=False, error=str(e))
            logger.warning(f"[rank {rank}] Failed to run subtask {subtask}: {e}")
            logger.debug("Subtask traceback:", exc_info=True)
            continue

    # Convert to DataFrame and pivot
    if results:
        df_results = pd.DataFrame(results)
        # Pivot to get one row per video with subtask scores as columns
        df_pivot = df_results.pivot(
            index="video_id",
            columns="subtask",
            values="score"
        ).reset_index()

        # Compute aggregate temporal score (mean of all subtasks)
        subtask_cols = [c for c in df_pivot.columns if c != "video_id"]
        if subtask_cols:
            df_pivot["vbench_temporal_score"] = df_pivot[subtask_cols].mean(axis=1)

        return df_pivot
    else:
        logger.warning(f"[rank {rank}] No VBench results obtained")
        return pd.DataFrame()


def run_vbench_cli_fallback(
    video_records: list,
    output_dir: Path,
    config: dict,
) -> pd.DataFrame:
    """
    Fallback: Run VBench via CLI if Python API fails.

    This uses VBench's command-line interface which is more stable.
    """
    import subprocess

    vbench_config = config.get("metrics", {}).get("vbench", {})
    subtasks = vbench_config.get("subtasks", [
        "temporal_flickering",
        "motion_smoothness",
    ])

    # Create video directory structure expected by VBench CLI
    videos_dir = output_dir / "vbench_input"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Symlink videos to expected location
    for record in video_records:
        src = Path(record["video_path"])
        dst = videos_dir / src.name
        if not dst.exists() and src.exists():
            dst.symlink_to(src.resolve())

    results_dir = output_dir / "vbench_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run VBench CLI
    vbench_script = VBENCH_ROOT / "evaluate.py"

    for subtask in subtasks:
        cmd = [
            sys.executable,
            str(vbench_script),
            "--videos_path", str(videos_dir.resolve()),
            "--dimension", subtask,
            "--output_path", str(results_dir.resolve()),
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            if result.returncode != 0:
                logger.warning(f"VBench CLI failed for {subtask}: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"VBench CLI timeout for {subtask}")
        except Exception as e:
            logger.warning(f"VBench CLI error for {subtask}: {e}")

    # Parse results from output directory
    results = []
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

                # VBench outputs different formats depending on version/mode
                # Handle list format: [{"video_name": "...", "dimension": "...", "score": ...}, ...]
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            video_name = item.get("video_name", item.get("video_path", ""))
                            dimension = item.get("dimension", item.get("name", ""))
                            score = item.get("score", item.get("value", None))
                            if video_name and score is not None:
                                results.append({
                                    "video_id": Path(video_name).stem,
                                    "subtask": dimension,
                                    "score": score,
                                })
                # Handle dict format: {"video_name": {"dimension": score, ...}, ...}
                elif isinstance(data, dict):
                    for video_name, scores in data.items():
                        if isinstance(scores, dict):
                            for dim, score in scores.items():
                                results.append({
                                    "video_id": Path(video_name).stem,
                                    "subtask": dim,
                                    "score": score,
                                })
                        elif isinstance(scores, (int, float)):
                            # Single score format
                            dim = result_file.stem  # Use filename as dimension
                            results.append({
                                "video_id": Path(video_name).stem,
                                "subtask": dim,
                                "score": scores,
                            })
        except Exception as e:
            logger.warning(f"Failed to parse {result_file}: {e}")

    if results:
        df_results = pd.DataFrame(results)
        df_pivot = df_results.pivot(
            index="video_id",
            columns="subtask",
            values="score"
        ).reset_index()

        subtask_cols = [c for c in df_pivot.columns if c != "video_id"]
        if subtask_cols:
            df_pivot["vbench_temporal_score"] = df_pivot[subtask_cols].mean(axis=1)

        return df_pivot

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Run VBench evaluation using official implementation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--skip-on-error", action="store_true",
        help="Skip VBench if errors occur (don't fail pipeline)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing results"
    )
    parser.add_argument(
        "--no-auto-multi-gpu",
        action="store_true",
        help="Disable auto-launch to multi-GPU torchrun",
    )
    args = parser.parse_args()
    configure_warning_filters()
    configure_third_party_loggers()

    # Check VBench installation
    if not check_vbench_installation():
        if args.skip_on_error:
            logger.warning("Skipping VBench evaluation due to missing submodule")
            sys.exit(0)
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    paths_config = config["paths"]
    runtime_config = config.get("runtime", {})
    rank, world_size, barrier_fn = init_distributed_if_needed()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank == 0 and world_size > 1:
        logger.info(
            "Multi-process dimension-parallel mode enabled: world_size=%d (no torch.distributed init)",
            world_size,
        )

    output_dir = Path(paths_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        sync_run_id = os.environ.get("TORCHELASTIC_RUN_ID") or str(int(time.time()))
        barrier_fn = make_file_barrier(
            sync_dir=output_dir / "vbench_sync" / sync_run_id,
            rank=rank,
            world_size=world_size,
        )

    vbench_output = output_dir / "vbench_per_video.csv"
    vbench_config = config.get("metrics", {}).get("vbench", {})

    # Check if already exists
    if vbench_output.exists() and not args.force:
        if rank == 0:
            logger.info(f"VBench results already exist: {vbench_output}")
            logger.info("Use --force to recompute")
        return
    elif vbench_output.exists() and args.force:
        if rank == 0:
            logger.info(f"Force recomputing: {vbench_output}")

    if maybe_auto_launch_multi_gpu(
        args=args,
        config=config,
        rank=rank,
        world_size=world_size,
        output_dir=output_dir,
    ):
        return

    # Load video metadata / build from local dataset
    try:
        video_records = load_video_records_for_vbench(
            config=config,
            output_dir=output_dir,
            paths_config=paths_config,
        )
        video_records = ensure_unique_video_ids(video_records, config)
    except Exception as e:
        logger.error(f"Failed to prepare video records: {e}")
        logger.error(
            "Need one of: processed_metadata.csv / metadata.csv / "
            "dataset.local_video_dir(with optional prompt_file)"
        )
        if args.skip_on_error:
            sys.exit(0)
        sys.exit(1)

    if rank == 0:
        logger.info(f"Loaded {len(video_records)} videos for VBench evaluation")

    all_subtasks = get_vbench_subtasks(config)
    assigned_subtasks = split_subtasks_for_rank(all_subtasks, rank=rank, world_size=world_size)
    visible_devices = _parse_visible_devices()
    if rank == 0:
        logger.info(
            "Subtask distribution: total=%d, world_size=%d",
            len(all_subtasks),
            world_size,
        )
        if world_size > 1:
            for worker_rank in range(world_size):
                worker_subtasks = split_subtasks_for_rank(
                    all_subtasks, rank=worker_rank, world_size=world_size
                )
                logger.info("  rank %d -> %s", worker_rank, worker_subtasks)
    if not assigned_subtasks:
        logger.warning("Rank %d has no assigned subtasks; producing empty partial.", rank)

    progress_reporter: RankProgressReporter | None = None
    progress_board: MultiGpuProgressBoard | None = None
    if world_size > 1:
        progress_dir = output_dir / "vbench_progress"
        if rank == 0:
            progress_dir.mkdir(parents=True, exist_ok=True)
            for stale_file in progress_dir.glob("rank_*.json"):
                try:
                    stale_file.unlink()
                except OSError:
                    pass
        barrier_fn()
        progress_reporter = RankProgressReporter(
            progress_dir=progress_dir,
            rank=rank,
            local_rank=local_rank,
            assigned_subtasks=assigned_subtasks,
            visible_devices=visible_devices,
        )
        if rank == 0:
            assignment_map = {
                worker_rank: split_subtasks_for_rank(
                    all_subtasks, rank=worker_rank, world_size=world_size
                )
                for worker_rank in range(world_size)
            }
            gpu_map = {
                worker_rank: (
                    visible_devices[worker_rank]
                    if worker_rank < len(visible_devices)
                    else str(worker_rank)
                )
                for worker_rank in range(world_size)
            }
            progress_board = MultiGpuProgressBoard(
                progress_dir=progress_dir,
                assignment_map=assignment_map,
                gpu_map=gpu_map,
                refresh_sec=1.0,
            )
            progress_board.start()

    try:
        device = runtime_config.get("device", "cuda")
        if world_size > 1 and isinstance(device, str) and device.startswith("cuda"):
            device = f"cuda:{local_rank}"

        # Try Python API first, fall back to CLI
        long_mode = use_vbench_long(config)

        try:
            if rank == 0:
                logger.info("Attempting VBench evaluation via Python API...")
            df_results = run_vbench_evaluation(
                video_records=video_records,
                output_dir=output_dir,
                config=config,
                device=device,
                rank=rank,
                world_size=world_size,
                barrier_fn=barrier_fn,
                subtasks_override=assigned_subtasks,
                progress_total_subtasks=len(all_subtasks),
                progress_reporter=progress_reporter,
            )
        except Exception as e:
            if rank == 0:
                logger.warning(f"Python API failed: {e}")
            if long_mode:
                if rank == 0:
                    logger.error("VBench-Long currently supports Python API path only in this pipeline.")
                if args.skip_on_error:
                    if rank == 0:
                        logger.warning("Skipping VBench evaluation")
                        pd.DataFrame(columns=["video_id", "vbench_temporal_score"]).to_csv(
                            vbench_output, index=False
                        )
                    sys.exit(0)
                raise

            if rank == 0:
                logger.info("Falling back to CLI-based evaluation...")
            try:
                if rank == 0:
                    df_results = run_vbench_cli_fallback(
                        video_records=video_records,
                        output_dir=output_dir,
                        config=config,
                    )
                else:
                    df_results = pd.DataFrame()
            except Exception as e2:
                if rank == 0:
                    logger.error(f"CLI fallback also failed: {e2}")
                if args.skip_on_error:
                    if rank == 0:
                        logger.warning("Skipping VBench evaluation")
                        # Create empty result file
                        pd.DataFrame(columns=["video_id", "vbench_temporal_score"]).to_csv(
                            vbench_output, index=False
                        )
                    sys.exit(0)
                raise

        if world_size > 1:
            partial_dir = output_dir / "vbench_partials"
            partial_dir.mkdir(parents=True, exist_ok=True)
            partial_file = partial_dir / f"rank_{rank}.csv"
            df_results.to_csv(partial_file, index=False)
            logger.info("Rank %d wrote partial results: %s", rank, partial_file)
            barrier_fn()
            if rank != 0:
                logger.info(f"Rank {rank} finished distributed VBench worker.")
                return

            partial_frames: list[pd.DataFrame] = []
            for worker_rank in range(world_size):
                rank_file = partial_dir / f"rank_{worker_rank}.csv"
                if rank_file.exists():
                    try:
                        partial_frames.append(pd.read_csv(rank_file))
                    except Exception as exc:
                        logger.warning("Failed to read %s: %s", rank_file, exc)
                else:
                    logger.warning("Missing partial results for rank %d: %s", worker_rank, rank_file)
            df_results = merge_rank_partial_results(partial_frames)
            logger.info("Merged %d partial result files on CPU.", len(partial_frames))

        # Merge with metadata to get group info
        df_meta = pd.DataFrame(video_records)[["video_id", "group"]].drop_duplicates(
            subset=["video_id"]
        )
        if not df_results.empty:
            df_results = df_results.merge(df_meta, on="video_id", how="left")
        expected_count = len(video_records)
        coverage_rows: list[tuple[str, int, int]] = []
        for subtask in all_subtasks:
            if subtask not in df_results.columns:
                covered = 0
                logger.warning("Missing subtask column in merged output: %s", subtask)
            else:
                covered = int(df_results[subtask].notna().sum())
            coverage_rows.append((subtask, covered, expected_count))
            if covered != expected_count:
                logger.warning(
                    "Subtask coverage mismatch for %s: %d/%d",
                    subtask,
                    covered,
                    expected_count,
                )

        missing_coverage = [
            (name, covered, total)
            for name, covered, total in coverage_rows
            if covered != total
        ]

        # Save results
        df_results.to_csv(vbench_output, index=False)
        logger.info(f"VBench results saved to: {vbench_output}")

        # Print summary
        if "vbench_temporal_score" in df_results.columns:
            logger.info("\nVBench Temporal Score Summary by Group:")
            summary = df_results.groupby("group")["vbench_temporal_score"].agg(["mean", "std"])
            logger.info(f"\n{summary}")

        # Copy outputs to frontend/public/data
        logger.info("\nCopying VBench outputs to frontend/public/data ...")
        copy_outputs_to_frontend(
            output_dir=output_dir,
            paths_config=paths_config,
            vbench_output=vbench_output,
        )

        if coverage_rows:
            name_width = max(len("subtask"), max(len(name) for name, _, _ in coverage_rows))
            logger.info("\nVBench Coverage Summary:")
            logger.info(f"{'subtask':<{name_width}} | coverage | status")
            logger.info(f"{'-' * name_width}-+----------+-------")
            for name, covered, total in coverage_rows:
                status = "OK" if covered == total else "MISS"
                logger.info(f"{name:<{name_width}} | {covered:>4d}/{total:<4d} | {status}")

        profile = str(vbench_config.get("dimension_profile", "")).strip().lower()
        strict_full_coverage = bool(vbench_config.get("require_full_coverage", False))
        if not strict_full_coverage and profile in {"long_16", "16", "16d", "full", "full_16"}:
            strict_full_coverage = True

        if strict_full_coverage and missing_coverage:
            short_msg = ", ".join(
                f"{name}:{covered}/{total}" for name, covered, total in missing_coverage
            )
            raise RuntimeError(
                "VBench coverage check failed under strict mode: "
                f"{short_msg}. "
                "Install missing dependencies / inspect failed subtasks, then rerun."
            )
    finally:
        if progress_reporter is not None:
            progress_reporter.mark_done()
        if progress_board is not None:
            progress_board.stop()


if __name__ == "__main__":
    main()
