"""
Parallel clip preprocessing helpers for VBench-Long.
"""

import contextlib
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

try:
    from .env import PROJECT_ROOT, logger
except ImportError:
    from vbench_runner.env import PROJECT_ROOT, logger


_VALID_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}


def _ensure_vbench_path() -> None:
    """Ensure official VBench python modules are importable."""
    vbench_root = PROJECT_ROOT / "third_party" / "VBench"
    vbench_root_str = str(vbench_root)
    if vbench_root_str not in sys.path:
        sys.path.insert(0, vbench_root_str)


def _count_clip_files(clip_dir: Path) -> int:
    """Count generated clip files for one source video."""
    if not clip_dir.exists() or not clip_dir.is_dir():
        return 0
    return sum(
        1
        for path in clip_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _VALID_VIDEO_SUFFIXES
    )


def _split_single_video(
    video_path_str: str,
    split_root_str: str,
    duration: int,
    fps: int,
) -> dict[str, Any]:
    """Split one source video into clips using official VBench utility."""
    video_path = Path(video_path_str)
    split_root = Path(split_root_str)
    video_id = video_path.stem
    clip_dir = split_root / video_id
    try:
        existing_clips = _count_clip_files(clip_dir)
        if existing_clips > 0:
            return {
                "ok": True,
                "video_id": video_id,
                "clip_count": existing_clips,
                "skipped": True,
            }

        _ensure_vbench_path()
        from vbench2_beta_long.utils import split_video_into_clips

        # Third-party splitter prints one line per clip; silence in worker process.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            split_video_into_clips(
                str(video_path),
                str(split_root),
                duration=duration,
                fps=fps,
            )

        clip_count = _count_clip_files(clip_dir)
        if clip_count <= 0:
            raise RuntimeError("No output clips generated")

        return {
            "ok": True,
            "video_id": video_id,
            "clip_count": clip_count,
            "skipped": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "video_id": video_id,
            "error": str(exc),
        }


def parallel_split_long_clips(
    video_dir: Path,
    input_videos: list[Path],
    duration: int,
    fps: int,
    workers: int,
    show_progress: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Split videos into VBench-Long clips with process-level parallelism.

    Returns:
        Summary dict with counts and elapsed time.
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")

    split_root = video_dir / "split_clip"
    split_root.mkdir(parents=True, exist_ok=True)

    tasks = [
        str(path)
        for path in input_videos
        if path.is_file() and path.suffix.lower() in _VALID_VIDEO_SUFFIXES
    ]
    total = len(tasks)
    if total == 0:
        return {
            "total_videos": 0,
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "clips": 0,
            "elapsed_sec": 0,
            "errors": [],
        }

    start = time.time()
    completed = 0
    failed = 0
    skipped = 0
    clips = 0
    errors: list[tuple[str, str]] = []

    def report_progress() -> None:
        if progress_callback is None:
            return
        percent = int(completed * 100 / total) if total > 0 else 100
        # Let caller mark 100% on finish_task().
        if completed < total:
            percent = min(percent, 99)
        else:
            percent = 99
        progress_callback(
            {
                "percent": percent,
                "status_text": f"videos={completed}/{total}, clips={clips}, failed={failed}",
                "elapsed_sec": int(time.time() - start),
            }
        )

    if workers == 1:
        iterator = tasks
        if show_progress:
            iterator = tqdm(tasks, total=total, desc="Preprocessing clips")
        for video_path_str in iterator:
            result = _split_single_video(
                video_path_str=video_path_str,
                split_root_str=str(split_root),
                duration=duration,
                fps=fps,
            )
            completed += 1
            if result.get("ok"):
                clips += int(result.get("clip_count", 0))
                skipped += int(bool(result.get("skipped")))
            else:
                failed += 1
                errors.append(
                    (str(result.get("video_id", "unknown")), str(result.get("error", "")))
                )
            report_progress()
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _split_single_video,
                    video_path_str,
                    str(split_root),
                    duration,
                    fps,
                )
                for video_path_str in tasks
            ]

            completed_futures = as_completed(futures)
            if show_progress:
                completed_futures = tqdm(
                    completed_futures,
                    total=total,
                    desc="Preprocessing clips",
                )

            for future in completed_futures:
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"ok": False, "video_id": "unknown", "error": str(exc)}

                completed += 1
                if result.get("ok"):
                    clips += int(result.get("clip_count", 0))
                    skipped += int(bool(result.get("skipped")))
                else:
                    failed += 1
                    errors.append(
                        (str(result.get("video_id", "unknown")), str(result.get("error", "")))
                    )
                report_progress()

    elapsed_sec = int(time.time() - start)
    succeeded = completed - failed

    if errors:
        max_report = 10
        for video_id, err in errors[:max_report]:
            logger.error("[preprocess] %s failed: %s", video_id, err)
        if len(errors) > max_report:
            logger.error("[preprocess] ... %d more failures omitted", len(errors) - max_report)

    return {
        "total_videos": total,
        "processed": succeeded,
        "failed": failed,
        "skipped": skipped,
        "clips": clips,
        "elapsed_sec": elapsed_sec,
        "errors": errors,
    }
