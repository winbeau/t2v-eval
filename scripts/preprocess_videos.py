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
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

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
) -> bool:
    """
    Write frames to video using ffmpeg.

    Args:
        frames: (N, H, W, C) numpy array in RGB format
        output_path: Output video path
        fps: Target FPS
        size: (width, height) tuple
    """
    n_frames, h, w, c = frames.shape

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
    success = write_video_ffmpeg(frames, output_path, target_fps, size)

    return {
        "original_frames": orig_frames,
        "original_fps": orig_fps,
        "original_width": orig_w,
        "original_height": orig_h,
        "duration_sec": round(duration_sec, 3),
        "processed_frames": len(indices),
        "processed": success,
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
    args = parser.parse_args()

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
    cache_dir = Path(paths_config["cache_dir"])
    processed_dir = cache_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / paths_config["metadata_file"]
    processed_metadata_path = output_dir / paths_config["processed_metadata"]

    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        logger.error("Run export_from_hf.py first to generate metadata.")
        sys.exit(1)

    # Load metadata
    df = pd.read_csv(metadata_path)
    logger.info(f"Loaded {len(df)} videos from metadata")

    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limited to {len(df)} videos")

    # Process videos
    processed_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing videos"):
        video_id = row["video_id"]
        group = row["group"]
        prompt = row["prompt"]
        input_path = row["video_path"]

        # Create group subdirectory
        group_dir = processed_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)

        output_path = group_dir / f"{video_id}.mp4"

        # Skip if exists and not forcing
        if output_path.exists() and not args.force:
            # Still need to get video info for metadata
            try:
                info = get_video_info(str(output_path))
                video_info = {
                    "original_frames": info[0],
                    "original_fps": info[1],
                    "original_width": info[2],
                    "original_height": info[3],
                    "duration_sec": round(info[0] / info[1], 3) if info[1] > 0 else 0,
                    "processed_frames": target_frames,
                    "processed": True,
                }
            except Exception:
                video_info = {
                    "original_frames": 0,
                    "original_fps": 0,
                    "original_width": 0,
                    "original_height": 0,
                    "duration_sec": 0,
                    "processed_frames": target_frames,
                    "processed": True,
                }
            logger.debug(f"Skipping existing: {output_path}")
        else:
            # Process the video
            if not os.path.exists(input_path):
                logger.warning(f"Input video not found: {input_path}")
                continue

            try:
                video_info = preprocess_single_video(
                    input_path=input_path,
                    output_path=str(output_path),
                    target_fps=target_fps,
                    target_frames=target_frames,
                    target_size=target_size,
                    sampling=sampling,
                    padding=padding,
                )
            except Exception as e:
                logger.error(f"Failed to process {video_id}: {e}")
                continue

        if not video_info.get("processed", False):
            logger.warning(f"Failed to write processed video: {video_id}")
            continue

        processed_records.append({
            "video_id": video_id,
            "group": group,
            "prompt": prompt,
            "video_path": str(output_path),
            "num_frames": video_info["processed_frames"],
            "duration_sec": video_info["duration_sec"],
            "original_frames": video_info["original_frames"],
            "original_fps": video_info["original_fps"],
        })

    # Save processed metadata
    df_processed = pd.DataFrame(processed_records)
    df_processed.to_csv(processed_metadata_path, index=False)
    logger.info(f"Processed metadata saved to: {processed_metadata_path}")
    logger.info(f"Total videos processed: {len(df_processed)}")

    # Print summary
    logger.info("\nPreprocessing Configuration:")
    logger.info(f"  Target FPS: {target_fps}")
    logger.info(f"  Target Frames: {target_frames}")
    logger.info(f"  Target Size: {target_size}x{target_size}")
    logger.info(f"  Sampling: {sampling}")
    logger.info(f"  Padding: {padding}")


if __name__ == "__main__":
    main()
