#!/usr/bin/env python3
"""
summarize.py
Aggregate per-video metrics into final summary tables.

This script:
1. Reads all per-video metric CSVs (clipvqa, vbench, flicker, niqe)
2. Merges them into a single per_video_metrics.csv
3. Computes group-level statistics (mean ± std)
4. Outputs group_summary.csv
5. Optionally merges runtime/FPS data if available

Usage:
    python scripts/summarize.py --config configs/eval.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

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


def load_metric_csv(path: Path, metric_name: str) -> Optional[pd.DataFrame]:
    """Load a metric CSV file if it exists."""
    if path.exists():
        try:
            df = pd.read_csv(path)
            if df.empty:
                logger.warning(f"{metric_name} file is empty: {path}")
                return None
            logger.info(f"Loaded {metric_name}: {len(df)} records from {path}")
            return df
        except pd.errors.EmptyDataError:
            logger.warning(f"{metric_name} file has no data: {path}")
            return None
    else:
        logger.warning(f"{metric_name} not found: {path}")
        return None


def merge_metrics(
    base_df: pd.DataFrame,
    metric_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all metric DataFrames on video_id.

    Args:
        base_df: Base DataFrame with video_id, group, etc.
        metric_dfs: Dict of metric name -> DataFrame

    Returns:
        Merged DataFrame
    """
    result = base_df.copy()

    for name, df in metric_dfs.items():
        if df is None or df.empty:
            continue

        # Get columns to merge (excluding video_id and group which are in base)
        merge_cols = ["video_id"] + [
            c for c in df.columns
            if c not in ["video_id", "group", "prompt", "video_path"]
        ]

        if len(merge_cols) > 1:
            df_to_merge = df[merge_cols].drop_duplicates(subset=["video_id"])
            result = result.merge(df_to_merge, on="video_id", how="left")
            logger.debug(f"Merged {name}: added {len(merge_cols) - 1} columns")

    return result


def compute_group_summary(
    df: pd.DataFrame,
    metric_cols: List[str],
) -> pd.DataFrame:
    """
    Compute group-level statistics (mean and std) for each metric.

    Args:
        df: DataFrame with video_id, group, and metric columns
        metric_cols: List of metric column names

    Returns:
        DataFrame with group-level summary
    """
    summary_records = []

    for group in df["group"].unique():
        group_df = df[df["group"] == group]

        record = {
            "group": group,
            "n_videos": len(group_df),
        }

        for col in metric_cols:
            if col in group_df.columns:
                values = group_df[col].dropna()
                if len(values) > 0:
                    record[f"{col}_mean"] = round(values.mean(), 4)
                    record[f"{col}_std"] = round(values.std(), 4)
                else:
                    record[f"{col}_mean"] = None
                    record[f"{col}_std"] = None

        summary_records.append(record)

    return pd.DataFrame(summary_records)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize evaluation metrics"
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

    output_dir = Path(paths_config["output_dir"])

    # Output paths
    per_video_output = output_dir / paths_config["per_video_metrics"]
    group_summary_output = output_dir / paths_config["group_summary"]

    # Check if already exists
    if per_video_output.exists() and group_summary_output.exists() and not args.force:
        logger.info("Summary files already exist. Use --force to regenerate.")
        return

    # Load base metadata
    processed_metadata = output_dir / paths_config["processed_metadata"]
    if not processed_metadata.exists():
        logger.error(f"Processed metadata not found: {processed_metadata}")
        return

    base_df = pd.read_csv(processed_metadata)
    logger.info(f"Loaded base metadata: {len(base_df)} videos")

    # Keep relevant columns from base
    base_cols = ["video_id", "group", "prompt", "num_frames", "duration_sec"]
    base_cols = [c for c in base_cols if c in base_df.columns]
    base_df = base_df[base_cols]

    # Load all metric CSVs
    metric_dfs = {
        "clipvqa": load_metric_csv(output_dir / "clipvqa_per_video.csv", "CLIP/VQA"),
        "vbench": load_metric_csv(output_dir / "vbench_per_video.csv", "VBench"),
        "flicker": load_metric_csv(output_dir / "flicker_per_video.csv", "Flicker"),
        "niqe": load_metric_csv(output_dir / "niqe_per_video.csv", "NIQE"),
    }

    # Load optional runtime data
    runtime_csv = output_dir / paths_config.get("runtime_csv", "runtime.csv")
    if runtime_csv.exists():
        metric_dfs["runtime"] = load_metric_csv(runtime_csv, "Runtime")
    else:
        logger.info(f"Runtime CSV not found (optional): {runtime_csv}")

    # Merge all metrics
    merged_df = merge_metrics(base_df, metric_dfs)

    # Save per-video metrics
    merged_df.to_csv(per_video_output, index=False)
    logger.info(f"Per-video metrics saved to: {per_video_output}")
    logger.info(f"Columns: {list(merged_df.columns)}")

    # Identify metric columns for summary
    metric_cols = [
        "clip_or_vqa_score",
        "vbench_temporal_score",
        "flicker_mean",
        "niqe_mean",
        "num_frames",
        "duration_sec",
    ]

    # Add any VBench subtask columns
    vbench_cols = [c for c in merged_df.columns if c.startswith("temporal_") or c.startswith("motion_")]
    metric_cols.extend(vbench_cols)

    # Add runtime columns if present
    if "fps" in merged_df.columns:
        metric_cols.append("fps")
    if "inference_time_sec" in merged_df.columns:
        metric_cols.append("inference_time_sec")

    # Filter to existing columns
    metric_cols = [c for c in metric_cols if c in merged_df.columns]

    # Compute group summary
    summary_df = compute_group_summary(merged_df, metric_cols)

    # Sort by group name
    summary_df = summary_df.sort_values("group").reset_index(drop=True)

    # Save group summary
    summary_df.to_csv(group_summary_output, index=False)
    logger.info(f"Group summary saved to: {group_summary_output}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("GROUP SUMMARY")
    logger.info("=" * 80)

    # Format for display
    display_cols = ["group", "n_videos"]
    for col in ["clip_or_vqa_score", "vbench_temporal_score", "flicker_mean", "niqe_mean"]:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col in summary_df.columns:
            display_cols.extend([mean_col, std_col])

    print(summary_df[display_cols].to_string(index=False))

    # Print metric directions
    logger.info("\nMetric Directions:")
    logger.info("  ↑ clip_or_vqa_score: Higher is better (text-video alignment)")
    logger.info("  ↑ vbench_temporal_score: Higher is better (temporal quality)")
    logger.info("  ↓ flicker_mean: Lower is better (temporal stability)")
    logger.info("  ↓ niqe_mean: Lower is better (visual quality)")


if __name__ == "__main__":
    main()
