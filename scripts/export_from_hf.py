#!/usr/bin/env python3
"""
export_from_hf.py
Export video dataset from HuggingFace Hub to local storage.

Generates metadata.csv with columns:
  - video_id: unique identifier
  - group: experiment group name
  - prompt: text prompt
  - video_path: local path to downloaded video
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
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


def export_video_bytes(video_data, output_path: Path) -> bool:
    """
    Export video data to file.
    Handles both bytes and file path formats from HF datasets.
    """
    try:
        if isinstance(video_data, bytes):
            output_path.write_bytes(video_data)
        elif isinstance(video_data, dict) and "bytes" in video_data:
            output_path.write_bytes(video_data["bytes"])
        elif isinstance(video_data, dict) and "path" in video_data:
            # Video is stored as a path reference
            import shutil
            shutil.copy(video_data["path"], output_path)
        elif isinstance(video_data, str) and os.path.exists(video_data):
            import shutil
            shutil.copy(video_data, output_path)
        else:
            logger.warning(f"Unknown video format: {type(video_data)}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to export video: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export HF dataset to local storage")
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for raw videos",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples (for testing)"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    dataset_config = config["dataset"]
    paths_config = config["paths"]

    repo_id = dataset_config["repo_id"]
    split = dataset_config.get("split", "test")

    # Setup directories
    cache_dir = Path(args.output_dir or paths_config["cache_dir"])
    raw_videos_dir = cache_dir / "raw"
    raw_videos_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(paths_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / paths_config["metadata_file"]

    logger.info(f"Loading dataset from HuggingFace: {repo_id}")
    logger.info(f"Split: {split}")

    try:
        dataset = load_dataset(repo_id, split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(
            "Please check:\n"
            "  1. Dataset repo_id is correct in configs/eval.yaml\n"
            "  2. You have access to the dataset (may need `huggingface-cli login`)\n"
            "  3. Network connection is available"
        )
        sys.exit(1)

    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Validate required columns
    required_columns = ["video", "prompt", "group", "video_id"]
    available_columns = dataset.column_names
    missing = [c for c in required_columns if c not in available_columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.error(f"Available columns: {available_columns}")
        sys.exit(1)

    # Export videos and collect metadata
    metadata_records = []
    samples = dataset if args.limit is None else dataset.select(range(min(args.limit, len(dataset))))

    for idx, sample in enumerate(tqdm(samples, desc="Exporting videos")):
        video_id = sample["video_id"]
        group = sample["group"]
        prompt = sample["prompt"]
        video_data = sample["video"]

        # Create group subdirectory
        group_dir = raw_videos_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)

        # Export video
        video_filename = f"{video_id}.mp4"
        video_path = group_dir / video_filename

        if video_path.exists() and not args.force:
            logger.debug(f"Skipping existing video: {video_path}")
        else:
            success = export_video_bytes(video_data, video_path)
            if not success:
                logger.warning(f"Failed to export video_id={video_id}, skipping")
                continue

        metadata_records.append(
            {
                "video_id": video_id,
                "group": group,
                "prompt": prompt,
                "video_path": str(video_path),
            }
        )

    # Save metadata
    df = pd.DataFrame(metadata_records)
    df.to_csv(metadata_path, index=False)
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Total videos exported: {len(df)}")

    # Print group distribution
    logger.info("Group distribution:")
    for group, count in df["group"].value_counts().items():
        logger.info(f"  {group}: {count}")


if __name__ == "__main__":
    main()
