#!/usr/bin/env python3
"""
run_niqe.py
Compute NIQE (Natural Image Quality Evaluator) scores for video frames.

NIQE is a no-reference image quality metric that measures how "natural"
an image looks based on statistical regularities of natural scenes.

Lower NIQE scores indicate better perceptual quality.

Usage:
    python scripts/run_niqe.py --config configs/eval.yaml
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:
    from .vbench_runner.group_labels import build_group_alias_map, remap_group_column
except ImportError:
    from vbench_runner.group_labels import build_group_alias_map, remap_group_column

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


def read_video_frames(
    video_path: str,
    num_frames: int = 8,
) -> np.ndarray:
    """
    Read uniformly sampled frames from video file.
    Returns: (N, H, W, C) numpy array in RGB format, uint8
    """
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return frames
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()
        return np.array(frames)


def compute_niqe_pyiqa(frames: np.ndarray, device: str = "cuda") -> float:
    """
    Compute NIQE using pyiqa library (recommended).

    Args:
        frames: (N, H, W, C) numpy array, uint8 RGB
        device: torch device

    Returns:
        Mean NIQE score across frames
    """
    import torch
    import pyiqa

    # Initialize NIQE metric
    niqe_metric = pyiqa.create_metric("niqe", device=device)

    scores = []
    for frame in frames:
        # Convert to tensor: (H, W, C) -> (1, C, H, W), normalized to [0, 1]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frame_tensor = frame_tensor.to(device)

        with torch.no_grad():
            score = niqe_metric(frame_tensor)
            scores.append(score.item())

    return float(np.mean(scores))


def compute_niqe_skvideo(frames: np.ndarray) -> float:
    """
    Compute NIQE using scikit-video (fallback).

    Args:
        frames: (N, H, W, C) numpy array, uint8 RGB

    Returns:
        Mean NIQE score across frames
    """
    from skvideo.measure import niqe as skvideo_niqe
    import cv2

    scores = []
    for frame in frames:
        # Convert to grayscale for NIQE
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        score = skvideo_niqe(gray)
        if np.isfinite(score):
            scores.append(float(score))

    if scores:
        return float(np.mean(scores))
    return float("nan")


def compute_niqe_custom(frames: np.ndarray) -> float:
    """
    Simplified NIQE implementation (fallback if libraries unavailable).

    This is a simplified approximation based on local contrast and sharpness.
    For accurate NIQE, use pyiqa or skvideo.
    """
    import cv2

    scores = []
    for frame in frames:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float64)

        # Compute local mean and variance (simplified NSS features)
        kernel_size = 7
        mu = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 1.16)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(gray * gray, (kernel_size, kernel_size), 1.16)
        sigma = np.sqrt(np.maximum(sigma - mu_sq, 0))

        # Normalized coefficients
        eps = 1e-7
        mscn = (gray - mu) / (sigma + eps)

        # Use variance of MSCN as quality proxy
        # Lower variance in MSCN often indicates distortion
        score = np.var(mscn)

        # Invert so lower is better (like NIQE)
        # This is a rough approximation
        niqe_approx = 10.0 / (score + 0.1)
        scores.append(niqe_approx)

    return float(np.mean(scores))


def compute_niqe(
    frames: np.ndarray,
    device: str = "cuda",
) -> float:
    """
    Compute NIQE score, trying multiple implementations.
    """
    # Try pyiqa first (best)
    try:
        return compute_niqe_pyiqa(frames, device)
    except ImportError:
        logger.debug("pyiqa not available, trying skvideo")
    except Exception as e:
        logger.debug(f"pyiqa failed: {e}")

    # Try skvideo
    try:
        return compute_niqe_skvideo(frames)
    except ImportError:
        logger.debug("skvideo not available, using custom implementation")
    except Exception as e:
        logger.debug(f"skvideo failed: {e}")

    # Fallback to custom
    logger.warning("Using simplified NIQE approximation. Install pyiqa for accurate results.")
    return compute_niqe_custom(frames)


def main():
    parser = argparse.ArgumentParser(
        description="Compute NIQE scores for video frames"
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
    niqe_config = config.get("metrics", {}).get("niqe", {})
    group_alias_map = build_group_alias_map(config)
    runtime_config = config.get("runtime", {})

    num_frames = niqe_config.get("num_frames_for_niqe", 8)
    device = runtime_config.get("device", "cuda")

    # Check CUDA availability
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
    except ImportError:
        device = "cpu"

    logger.info(f"NIQE configuration: num_frames={num_frames}, device={device}")

    output_dir = Path(paths_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_metadata = output_dir / paths_config["processed_metadata"]
    output_file = output_dir / "niqe_per_video.csv"

    # Check if already exists
    if output_file.exists() and not args.force:
        logger.info(f"Results already exist: {output_file}")
        logger.info("Use --force to recompute")
        return
    elif output_file.exists() and args.force:
        logger.info(f"Force recomputing: {output_file}")

    # Load video metadata
    if not processed_metadata.exists():
        logger.error(f"Processed metadata not found: {processed_metadata}")
        logger.error("Run preprocess_videos.py first")
        return

    df_meta = pd.read_csv(processed_metadata)
    df_meta = remap_group_column(df_meta, group_alias_map)
    logger.info(f"Loaded {len(df_meta)} videos for NIQE evaluation")

    # Compute NIQE scores
    results = []

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Computing NIQE"):
        video_id = row["video_id"]
        video_path = row["video_path"]
        group = row.get("group", "unknown")

        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            continue

        try:
            frames = read_video_frames(video_path, num_frames=num_frames)
            niqe_score = compute_niqe(frames, device=device)

            results.append({
                "video_id": video_id,
                "group": group,
                "niqe_mean": round(niqe_score, 4),
            })

        except Exception as e:
            logger.warning(f"Failed to compute NIQE for {video_id}: {e}")
            continue

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    logger.info(f"NIQE results saved to: {output_file}")

    # Print summary
    logger.info("\nNIQE Score Summary by Group (lower is better):")
    summary = df_results.groupby("group")["niqe_mean"].agg(["mean", "std"])
    logger.info(f"\n{summary}")


if __name__ == "__main__":
    main()
