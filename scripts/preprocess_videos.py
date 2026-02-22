#!/usr/bin/env python3
"""
preprocess_videos.py
Preprocess videos to unified format for evaluation.

- Resample to target FPS
- Resize to target resolution
- Extract fixed number of frames
- Handle videos with insufficient frames (loop/repeat/truncate)

Output: eval_cache/{group}/{video_id}.mp4 + processed_metadata.csv
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:
    from vbench_runner.video_records import build_video_list_from_local_dataset
except ImportError:
    from scripts.vbench_runner.video_records import build_video_list_from_local_dataset

try:
    import decord
    decord.bridge.set_bridge("native")
    USE_DECORD = True
except ImportError:
    USE_DECORD = False
    import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_video_info(video_path: str) -> Tuple[int, float, int, int]:
    """
    Get video information: num_frames, fps, width, height.
    """
    if USE_DECORD:
        vr = decord.VideoReader(video_path)
        num_frames = len(vr)
        fps = vr.get_avg_fps()
        h, w, _ = vr[0].shape
        return num_frames, fps, w, h
    else:
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return num_frames, fps, w, h


def read_video_frames(video_path: str, indices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Read video frames at specified indices.
    Returns: (N, H, W, C) numpy array in RGB format.
    """
    if USE_DECORD:
        vr = decord.VideoReader(video_path)
        if indices is None:
            indices = np.arange(len(vr))
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C) RGB
        return frames
    else:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if indices is None:
            indices = np.arange(total_frames)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        return np.array(frames)


def compute_frame_indices(
    total_frames: int,
    target_frames: int,
    sampling: str = "uniform",
    padding: str = "loop",
) -> np.ndarray:
    """
    Compute frame indices for sampling.

    Args:
        total_frames: Number of frames in source video
        target_frames: Desired number of output frames
        sampling: Sampling strategy (uniform)
        padding: How to handle insufficient frames (loop, repeat_last, truncate)

    Returns:
        Array of frame indices to extract
    """
    if total_frames >= target_frames:
        # Uniform sampling when we have enough frames
        if sampling == "uniform":
            indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        else:
            indices = np.arange(target_frames)
    else:
        # Handle insufficient frames
        base_indices = np.arange(total_frames)

        if padding == "loop":
            # Cycle through frames
            repetitions = int(np.ceil(target_frames / total_frames))
            indices = np.tile(base_indices, repetitions)[:target_frames]
        elif padding == "repeat_last":
            # Repeat the last frame
            padding_needed = target_frames - total_frames
            indices = np.concatenate([base_indices, np.full(padding_needed, total_frames - 1)])
        elif padding == "truncate":
            # Just use what we have
            indices = base_indices
        else:
            raise ValueError(f"Unknown padding strategy: {padding}")

    return indices


def write_video_ffmpeg(
    frames: np.ndarray,
    output_path: str,
    fps: int,
    size: Tuple[int, int],
    ffmpeg_threads: int = 1,
) -> bool:
    """
    Write frames to video using ffmpeg.

    Args:
        frames: (N, H, W, C) numpy array in RGB format
        output_path: Output video path
        fps: Target FPS
        size: (width, height) tuple
    """
    _, h, w, _ = frames.shape

    # Resize if needed
    if (w, h) != size:
        from PIL import Image
        resized_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            img = img.resize(size, Image.LANCZOS)
            resized_frames.append(np.array(img))
        frames = np.array(resized_frames)

    # Use ffmpeg to write video
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{size[0]}x{size[1]}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",  # Read from stdin
        "-c:v", "libx264",
        "-threads", str(ffmpeg_threads),
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        output_path,
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        process.communicate(input=frames.tobytes())
        return process.returncode == 0
    except Exception as e:
        logger.error(f"FFmpeg error: {e}")
        return False


def preprocess_single_video(
    input_path: str,
    output_path: str,
    target_fps: int,
    target_frames: int,
    target_size: int,
    sampling: str,
    padding: str,
    ffmpeg_threads: int,
) -> dict:
    """
    Preprocess a single video.

    Returns:
        Dict with video info (original_frames, duration_sec, etc.)
    """
    # Get source video info
    orig_frames, orig_fps, orig_w, orig_h = get_video_info(input_path)
    duration_sec = orig_frames / orig_fps if orig_fps > 0 else 0

    # Compute frame indices
    indices = compute_frame_indices(
        total_frames=orig_frames,
        target_frames=target_frames,
        sampling=sampling,
        padding=padding,
    )

    # Read frames
    frames = read_video_frames(input_path, indices)

    # Write processed video
    size = (target_size, target_size)
    success = write_video_ffmpeg(
        frames,
        output_path,
        target_fps,
        size,
        ffmpeg_threads=ffmpeg_threads,
    )

    return {
        "original_frames": orig_frames,
        "original_fps": orig_fps,
        "original_width": orig_w,
        "original_height": orig_h,
        "duration_sec": round(duration_sec, 3),
        "processed_frames": len(indices),
        "processed": success,
    }


def _safe_prompt_value(prompt: Any) -> str:
    """Normalize prompt value loaded from metadata CSV."""
    if pd.isna(prompt):
        return ""
    return str(prompt)


def _video_info_for_existing_output(output_path: str, target_frames: int) -> Dict[str, Any]:
    """Read info for an already-processed output file."""
    try:
        info = get_video_info(output_path)
        return {
            "original_frames": info[0],
            "original_fps": info[1],
            "original_width": info[2],
            "original_height": info[3],
            "duration_sec": round(info[0] / info[1], 3) if info[1] > 0 else 0,
            "processed_frames": target_frames,
            "processed": True,
        }
    except Exception:
        return {
            "original_frames": 0,
            "original_fps": 0,
            "original_width": 0,
            "original_height": 0,
            "duration_sec": 0,
            "processed_frames": target_frames,
            "processed": True,
        }


def _build_processed_record(
    video_id: str,
    group: str,
    prompt: str,
    output_path: Path,
    video_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one row for processed metadata."""
    return {
        "video_id": video_id,
        "group": group,
        "prompt": prompt,
        "video_path": str(output_path),
        "num_frames": video_info["processed_frames"],
        "duration_sec": video_info["duration_sec"],
        "original_frames": video_info["original_frames"],
        "original_fps": video_info["original_fps"],
    }


def load_input_records(config: dict, output_dir: Path, paths_config: dict) -> pd.DataFrame:
    """
    Load preprocess input records.

    Priority:
    1) outputs/<metadata_file>
    2) build from dataset.local_video_dir (same fallback idea as run_vbench)
    """
    metadata_path = output_dir / paths_config["metadata_file"]
    if metadata_path.exists():
        df = pd.read_csv(metadata_path)
        logger.info("Loaded %d videos from metadata: %s", len(df), metadata_path)
    else:
        logger.warning("Metadata file not found: %s", metadata_path)
        logger.info("Falling back to local dataset scan (run_vbench-style record loading)")
        records = build_video_list_from_local_dataset(config)
        df = pd.DataFrame(records)
        logger.info("Built %d videos from local dataset directory", len(df))

    required_columns = {"video_id", "group", "video_path"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Input records missing required columns: {missing_cols}")

    if "prompt" not in df.columns:
        df["prompt"] = ""

    return df


def preprocess_video_task(
    row: Dict[str, Any],
    processed_dir: str,
    target_fps: int,
    target_frames: int,
    target_size: int,
    sampling: str,
    padding: str,
    force: bool,
    ffmpeg_threads: int,
) -> Dict[str, Any]:
    """Process one video task and return structured result."""
    video_id = str(row["video_id"])
    group = str(row["group"])
    prompt = _safe_prompt_value(row.get("prompt", ""))
    input_path = str(row["video_path"])

    group_dir = Path(processed_dir) / group
    group_dir.mkdir(parents=True, exist_ok=True)
    output_path = group_dir / f"{video_id}.mp4"

    skipped = False
    if output_path.exists() and not force:
        video_info = _video_info_for_existing_output(str(output_path), target_frames)
        skipped = True
    else:
        if not os.path.exists(input_path):
            return {
                "ok": False,
                "reason": "missing_input",
                "video_id": video_id,
                "message": f"Input video not found: {input_path}",
            }
        try:
            video_info = preprocess_single_video(
                input_path=input_path,
                output_path=str(output_path),
                target_fps=target_fps,
                target_frames=target_frames,
                target_size=target_size,
                sampling=sampling,
                padding=padding,
                ffmpeg_threads=ffmpeg_threads,
            )
        except Exception as exc:
            return {
                "ok": False,
                "reason": "preprocess_exception",
                "video_id": video_id,
                "message": f"Failed to preprocess video: {exc}",
            }

    if not video_info.get("processed", False):
        return {
            "ok": False,
            "reason": "write_failed",
            "video_id": video_id,
            "message": "Failed to write processed video",
        }

    return {
        "ok": True,
        "video_id": video_id,
        "skipped": skipped,
        "record": _build_processed_record(video_id, group, prompt, output_path, video_info),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for evaluation")
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing processed videos"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of videos (for testing)"
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help="Number of worker processes for video preprocessing",
    )
    parser.add_argument(
        "--ffmpeg-threads",
        type=int,
        default=1,
        help="Number of threads per ffmpeg process",
    )
    args = parser.parse_args()
    if args.preprocess_workers < 1:
        parser.error("--preprocess-workers must be >= 1")
    if args.ffmpeg_threads < 1:
        parser.error("--ffmpeg-threads must be >= 1")

    # Load configuration
    config = load_config(args.config)
    protocol = config["protocol"]
    paths_config = config["paths"]

    target_fps = protocol["fps_eval"]
    target_frames = protocol["num_frames"]
    target_size = protocol["resize"]
    sampling = protocol["frame_sampling"]
    padding = protocol.get("frame_padding", "loop")

    # Setup paths
    output_dir = Path(paths_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(paths_config["cache_dir"])
    processed_dir = cache_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_metadata_path = output_dir / paths_config["processed_metadata"]

    # Load input records (metadata first, then local dataset fallback)
    try:
        df = load_input_records(config=config, output_dir=output_dir, paths_config=paths_config)
    except Exception as exc:
        logger.error("Failed to prepare input records for preprocessing: %s", exc)
        sys.exit(1)

    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to {len(df)} videos")

    # Process videos
    rows = df.to_dict(orient="records")
    processed_records = []
    skipped_count = 0
    failed_count = 0

    task_kwargs = {
        "processed_dir": str(processed_dir),
        "target_fps": target_fps,
        "target_frames": target_frames,
        "target_size": target_size,
        "sampling": sampling,
        "padding": padding,
        "force": args.force,
        "ffmpeg_threads": args.ffmpeg_threads,
    }

    logger.info(
        "Preprocess concurrency: workers=%d, ffmpeg_threads=%d",
        args.preprocess_workers,
        args.ffmpeg_threads,
    )

    if args.preprocess_workers == 1:
        for row in tqdm(rows, total=len(rows), desc="Preprocessing videos"):
            result = preprocess_video_task(row=row, **task_kwargs)
            if result["ok"]:
                if result["skipped"]:
                    skipped_count += 1
                processed_records.append(result["record"])
                continue

            failed_count += 1
            level = logger.warning if result["reason"] == "missing_input" else logger.error
            level("%s: %s", result["video_id"], result["message"])
    else:
        with ProcessPoolExecutor(max_workers=args.preprocess_workers) as executor:
            futures = [
                executor.submit(preprocess_video_task, row=row, **task_kwargs)
                for row in rows
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing videos"):
                try:
                    result = future.result()
                except Exception as exc:
                    failed_count += 1
                    logger.error("Worker crashed during preprocessing: %s", exc)
                    continue

                if result["ok"]:
                    if result["skipped"]:
                        skipped_count += 1
                    processed_records.append(result["record"])
                    continue

                failed_count += 1
                level = logger.warning if result["reason"] == "missing_input" else logger.error
                level("%s: %s", result["video_id"], result["message"])

    # Save processed metadata
    df_processed = pd.DataFrame(processed_records)
    if not df_processed.empty:
        df_processed = df_processed.sort_values(["group", "video_id"]).reset_index(drop=True)
    df_processed.to_csv(processed_metadata_path, index=False)
    logger.info(f"Processed metadata saved to: {processed_metadata_path}")
    logger.info(f"Total videos processed: {len(df_processed)}")
    logger.info(f"Skipped existing videos: {skipped_count}")
    logger.info(f"Failed videos: {failed_count}")

    # Print summary
    logger.info("\nPreprocessing Configuration:")
    logger.info(f"  Target FPS: {target_fps}")
    logger.info(f"  Target Frames: {target_frames}")
    logger.info(f"  Target Size: {target_size}x{target_size}")
    logger.info(f"  Sampling: {sampling}")
    logger.info(f"  Padding: {padding}")
    logger.info(f"  Preprocess Workers: {args.preprocess_workers}")
    logger.info(f"  FFmpeg Threads: {args.ffmpeg_threads}")


if __name__ == "__main__":
    main()
