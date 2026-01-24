#!/usr/bin/env python3
"""
run_flicker.py
Compute Temporal Flicker Score for generated videos.

This is a no-reference metric that measures frame-to-frame instability
by computing the mean absolute difference between consecutive frames.

Formula:
    flicker_mean = (1/T-1) * sum_t( mean_pixels( |I_t - I_{t-1}| ) )
    flicker_std = std_t( mean_pixels( |I_t - I_{t-1}| ) )

Lower scores indicate better temporal consistency.

Usage:
    python scripts/run_flicker.py --config configs/eval.yaml
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

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


def read_video_frames(video_path: str) -> np.ndarray:
    """
    Read all frames from video file.
    Returns: (N, H, W, C) numpy array in RGB format, float32 [0, 1]
    """
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        frames = vr.get_batch(range(len(vr))).asnumpy()
        return frames.astype(np.float32) / 255.0
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames, dtype=np.float32) / 255.0


def compute_flicker_score(
    video_path: str,
    method: str = "l1",
    normalize: bool = True,
    grayscale: bool = False,
) -> Tuple[float, float]:
    """
    Compute temporal flicker score for a video.

    Args:
        video_path: Path to video file
        method: Difference method - 'l1' or 'l2'
        normalize: Whether to normalize pixel values to [0, 1]
        grayscale: Whether to convert to grayscale before computing

    Returns:
        Tuple of (flicker_mean, flicker_std)

    The score represents the average per-pixel change between consecutive frames.
    Lower values indicate more temporal stability.
    """
    # Read frames
    frames = read_video_frames(video_path)  # (T, H, W, C), float32 [0, 1]

    if len(frames) < 2:
        logger.warning(f"Video has less than 2 frames: {video_path}")
        return 0.0, 0.0

    if not normalize:
        frames = frames * 255.0

    # Convert to grayscale if requested
    if grayscale:
        # ITU-R BT.601 luma coefficients
        frames = np.dot(frames[..., :3], [0.299, 0.587, 0.114])
        frames = frames[..., np.newaxis]

    # Compute frame-to-frame differences
    frame_diffs = []

    for t in range(1, len(frames)):
        diff = frames[t] - frames[t - 1]

        if method == "l1":
            # Mean absolute difference per pixel
            pixel_diff = np.abs(diff).mean()
        elif method == "l2":
            # Root mean squared difference per pixel
            pixel_diff = np.sqrt((diff ** 2).mean())
        else:
            raise ValueError(f"Unknown method: {method}")

        frame_diffs.append(pixel_diff)

    frame_diffs = np.array(frame_diffs)

    flicker_mean = float(frame_diffs.mean())
    flicker_std = float(frame_diffs.std())

    return flicker_mean, flicker_std


def main():
    parser = argparse.ArgumentParser(
        description="Compute Temporal Flicker Score for videos"
    )
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing results"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    paths_config = config["paths"]
    flicker_config = config.get("metrics", {}).get("flicker", {})

    method = flicker_config.get("method", "l1")
    normalize = flicker_config.get("normalize", True)
    compute_std = flicker_config.get("compute_std", True)
    grayscale = flicker_config.get("grayscale", False)

    logger.info(f"Flicker configuration: method={method}, normalize={normalize}, grayscale={grayscale}")

    output_dir = Path(paths_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_metadata = output_dir / paths_config["processed_metadata"]
    output_file = output_dir / "flicker_per_video.csv"

    # Check if already exists
    if output_file.exists() and not args.force:
        logger.info(f"Results already exist: {output_file}")
        logger.info("Use --force to recompute")
        return

    # Load video metadata
    if not processed_metadata.exists():
        logger.error(f"Processed metadata not found: {processed_metadata}")
        logger.error("Run preprocess_videos.py first")
        return

    df_meta = pd.read_csv(processed_metadata)
    logger.info(f"Loaded {len(df_meta)} videos for flicker evaluation")

    # Compute flicker scores
    results = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Computing Flicker"):
        video_id = row["video_id"]
        video_path = row["video_path"]
        group = row.get("group", "unknown")

        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            continue

        try:
            flicker_mean, flicker_std = compute_flicker_score(
                video_path=video_path,
                method=method,
                normalize=normalize,
                grayscale=grayscale,
            )

            result = {
                "video_id": video_id,
                "group": group,
                "flicker_mean": round(flicker_mean, 6),
            }

            if compute_std:
                result["flicker_std"] = round(flicker_std, 6)

            results.append(result)

        except Exception as e:
            logger.warning(f"Failed to compute flicker for {video_id}: {e}")
            continue

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    logger.info(f"Flicker results saved to: {output_file}")

    # Print summary
    logger.info("\nFlicker Score Summary by Group (lower is better):")
    summary = df_results.groupby("group")["flicker_mean"].agg(["mean", "std"])
    logger.info(f"\n{summary}")


if __name__ == "__main__":
    main()
