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
import contextlib
import io
import json
import logging
import os
import re
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

# =============================================================================
# Setup paths to use official VBench from submodule
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
VBENCH_ROOT = PROJECT_ROOT / "third_party" / "VBench"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    Initialize torch distributed when launched via torchrun.

    Returns:
        (rank, world_size, barrier_fn)
    """
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env <= 1:
        return 0, 1, (lambda: None)

    setup_vbench_path()
    from vbench.distributed import dist_init, get_rank, get_world_size, barrier

    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist_init()
    except Exception:
        # Fallback to VBench helper init in unusual runtimes
        dist_init()

    return int(get_rank()), int(get_world_size()), barrier


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

        all_files = sorted(existing_files.union(set(copied_files)))
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
        self._line_count = 0
        self._chars = 0
        self._latest_percent: int | None = None

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
                match = re.search(r"(\d{1,3})%\|", line_stripped)
                if match:
                    percent = int(match.group(1))
                    if 0 <= percent <= 100:
                        self._latest_percent = percent
        return len(text)

    def flush(self) -> None:
        return

    def snapshot(self) -> tuple[int, int, int, int | None]:
        with self._lock:
            return self._saved_count, self._line_count, self._chars, self._latest_percent

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


def run_callable_with_progress(
    task_fn,
    title: str,
    prefix: str = "",
    refresh_sec: float = 1.0,
    status_mode: str = "events",
    enable_live: bool = True,
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
        if enable_live:
            elapsed = int(time.time() - start)
            saved_count, line_count, _, latest_percent = monitor.snapshot()
            if latest_percent is not None:
                bar = _render_percent_bar(latest_percent)
                pct_text = f"{latest_percent:>3d}%"
            else:
                bar = _render_bar(tick)
                pct_text = " --%"

            if status_mode == "clips":
                status_text = f"clips={saved_count}"
            else:
                status_text = f"events={line_count}"

            line = f"\r{prefix}{title:<24} | {bar} | {pct_text} | {status_text} | {elapsed:>4}s"
            print(line, end="", flush=True, file=sys.__stdout__)
            tick += 1
        thread.join(timeout=refresh_sec)

    thread.join()
    elapsed = int(time.time() - start)
    saved_count, line_count, _, latest_percent = monitor.snapshot()
    final_percent = 100 if latest_percent is None else max(100, latest_percent)
    final_bar = _render_percent_bar(final_percent)
    if status_mode == "clips":
        final_status = f"clips={saved_count}"
    else:
        final_status = f"events={line_count}"
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
    clip_required_subtasks = {
        "background_consistency",
        "aesthetic_quality",
        "appearance_style",
        "human_action",
        "overall_consistency",
        "temporal_style",
    }
    required = sorted(set(subtasks).intersection(clip_required_subtasks))
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
        return vbench_config["subtasks"]

    if use_vbench_long(config):
        return [
            "subject_consistency",
            "background_consistency",
            "motion_smoothness",
            "dynamic_degree",
            "imaging_quality",
            "aesthetic_quality",
        ]

    return [
        "temporal_flickering",
        "motion_smoothness",
    ]


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
    for item in per_video_items:
        if not isinstance(item, dict):
            continue
        video_path = item.get("video_path", item.get("video_name", ""))
        score = item.get("video_results", item.get("score"))
        if not video_path or score is None:
            continue
        parsed_items.append(
            {
                "video_id": resolve_video_id(str(video_path), valid_video_ids),
                "subtask": subtask,
                "score": float(score),
            }
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
    subtasks = get_vbench_subtasks(config)
    eval_mode = vbench_config.get("mode", "long_custom_input" if long_mode else "custom_input")

    ensure_pyav_dependency(long_mode=long_mode)
    ensure_clip_dependency(subtasks=subtasks)

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
                preprocess_stdout, saved_count = run_callable_with_progress(
                    task_fn=lambda: vbench.preprocess(**preprocess_kwargs),
                    title="preprocess_clips",
                    prefix="[0/6] ",
                    refresh_sec=1.0,
                    status_mode="clips",
                    enable_live=True,
                )
                if saved_count > 0:
                    logger.info(f"[preprocess] Saved split clips: {saved_count}")
                summarize_vbench_stdout(preprocess_stdout, "preprocess")
            if world_size > 1:
                barrier_fn()

        # VBench-Long evaluate() always calls preprocess(). We already did one-time
        # preprocessing above, so disable repeated preprocessing for each subtask.
        def _skip_preprocess(*args, **kwargs):
            return None

        vbench.preprocess = _skip_preprocess

    # Run evaluation for each subtask
    backend_name = "VBench-Long" if long_mode else "VBench"
    if rank == 0:
        logger.info(f"Running {backend_name} evaluation with subtasks: {subtasks}")

    total_subtasks = len(subtasks)
    for idx, subtask in enumerate(subtasks, start=1):
        if rank == 0:
            logger.info(f"Evaluating subtask: {subtask}")
        try:
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
                subtask_total=total_subtasks,
                refresh_sec=1.0,
                enable_live=(rank == 0),
            )
            if rank == 0 and saved_count > 0:
                logger.info(f"[{subtask}] Saved split clips: {saved_count}")
            if rank == 0:
                summarize_vbench_stdout(stdout_text, subtask)

            if rank != 0:
                continue
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

                    logger.info(f"Parsed {len(results) - count_before} results for {subtask}")
            else:
                logger.warning(f"Result file not found: {result_file}")
        except Exception as e:
            if rank == 0:
                logger.warning(f"Failed to run subtask {subtask}: {e}")
                logger.debug("Subtask traceback:", exc_info=True)
            continue

    if rank != 0:
        return pd.DataFrame()

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
        logger.warning("No VBench results obtained")
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
        logger.info(f"Distributed mode enabled: world_size={world_size}")

    output_dir = Path(paths_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    vbench_output = output_dir / "vbench_per_video.csv"

    # Check if already exists
    if vbench_output.exists() and not args.force:
        if rank == 0:
            logger.info(f"VBench results already exist: {vbench_output}")
            logger.info("Use --force to recompute")
        return
    elif vbench_output.exists() and args.force:
        if rank == 0:
            logger.info(f"Force recomputing: {vbench_output}")

    # Load video metadata / build from local dataset
    try:
        video_records = load_video_records_for_vbench(
            config=config,
            output_dir=output_dir,
            paths_config=paths_config,
        )
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
        barrier_fn()
    if rank != 0:
        logger.info(f"Rank {rank} finished distributed VBench worker.")
        return

    # Merge with metadata to get group info
    df_meta = pd.DataFrame(video_records)[["video_id", "group"]].drop_duplicates(subset=["video_id"])
    if not df_results.empty:
        df_results = df_results.merge(df_meta, on="video_id", how="left")

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


if __name__ == "__main__":
    main()
