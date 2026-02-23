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

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Deep-Forcing paper-aligned 8 dimensions (+ optional throughput/fps).
PROFILE_METRICS: dict[str, list[str]] = {
    "deep_forcing_8d": [
        "dynamic_degree",
        "motion_smoothness",
        "overall_consistency",
        "imaging_quality",
        "aesthetic_quality",
        "subject_consistency",
        "background_consistency",
    ]
}

# For deep-forcing alignment, these 0-1 dimensions should be scaled to 0-100.
DEEP_FORCING_PERCENT_METRICS = [
    "dynamic_degree",
    "motion_smoothness",
    "overall_consistency",
    "aesthetic_quality",
    "subject_consistency",
    "background_consistency",
]


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_metric_csv(path: Path, metric_name: str) -> pd.DataFrame | None:
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
    metric_dfs: dict[str, pd.DataFrame],
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


def resolve_comparison_profile(vbench_config: dict) -> str | None:
    """Normalize configured comparison profile name."""
    raw = vbench_config.get("comparison_profile")
    if raw is None:
        return None
    profile = str(raw).strip().lower()
    if not profile:
        return None
    if profile not in PROFILE_METRICS:
        logger.warning(
            "Unknown metrics.vbench.comparison_profile=%r; supported=%s",
            raw,
            sorted(PROFILE_METRICS.keys()),
        )
        return None
    return profile


def _normalize_metric_name_list(raw) -> list[str]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    normalized: list[str] = []
    for item in values:
        value = str(item).strip()
        if value:
            normalized.append(value)
    return list(dict.fromkeys(normalized))


def resolve_scale_to_percent(vbench_config: dict, comparison_profile: str | None) -> list[str]:
    """
    Resolve which columns should be scaled from [0,1] -> [0,100].

    Priority:
      1) explicit metrics.vbench.scale_to_percent
      2) default by comparison profile (currently deep_forcing_8d)
      3) empty
    """
    explicit = _normalize_metric_name_list(vbench_config.get("scale_to_percent"))
    if explicit:
        return explicit
    if comparison_profile == "deep_forcing_8d":
        return list(DEEP_FORCING_PERCENT_METRICS)
    return []


def apply_percent_scaling(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Scale numeric columns by 100 in-place. Returns actually scaled columns."""
    scaled: list[str] = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        df[col] = series * 100.0
        scaled.append(col)
    return scaled


def resolve_profile_metric_cols(profile: str, df_columns: list[str]) -> list[str]:
    """
    Resolve metric columns for a named comparison profile.

    For deep_forcing_8d, include fps when available for throughput comparison.
    """
    if profile not in PROFILE_METRICS:
        return []
    cols = [col for col in PROFILE_METRICS[profile] if col in df_columns]
    if profile == "deep_forcing_8d" and "fps" in df_columns:
        cols = ["fps"] + cols
    return cols


def cleanup_summary_column_names(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize duplicated suffix artifacts (e.g., flicker_mean_mean -> flicker_mean)."""
    rename_map = {}
    for col in summary_df.columns:
        if col.endswith("_mean_mean"):
            rename_map[col] = col.replace("_mean_mean", "_mean")
        elif col.endswith("_mean_std"):
            rename_map[col] = col.replace("_mean_std", "_std")
    if rename_map:
        summary_df = summary_df.rename(columns=rename_map)
        logger.info(f"Renamed columns: {rename_map}")
    return summary_df


def load_base_metadata_for_summary(output_dir: Path, paths_config: dict) -> pd.DataFrame | None:
    """
    Load base metadata for summary merge.

    Priority:
      1) processed_metadata.csv (full pipeline output)
      2) vbench_per_video.csv
      3) first vbench_*.csv with video_id/group columns
    """
    processed_metadata = output_dir / paths_config["processed_metadata"]
    if processed_metadata.exists():
        df = pd.read_csv(processed_metadata)
        logger.info(f"Loaded base metadata: {len(df)} videos from {processed_metadata}")
        return df

    logger.warning(
        "Processed metadata not found: %s. Trying VBench CSV fallback...",
        processed_metadata,
    )

    fallback_candidates: list[Path] = []
    vbench_per_video = output_dir / "vbench_per_video.csv"
    if vbench_per_video.exists():
        fallback_candidates.append(vbench_per_video)
    fallback_candidates.extend(sorted(output_dir.glob("vbench_*.csv")))

    seen: set[Path] = set()
    deduped_candidates: list[Path] = []
    for path in fallback_candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped_candidates.append(path)

    for candidate in deduped_candidates:
        try:
            df = pd.read_csv(candidate)
        except Exception as exc:
            logger.warning("Failed to read fallback base from %s: %s", candidate, exc)
            continue

        required = {"video_id", "group"}
        if not required.issubset(df.columns):
            logger.warning(
                "Skip fallback base %s because missing required columns: %s",
                candidate,
                sorted(required - set(df.columns)),
            )
            continue

        logger.info("Using fallback base metadata from %s", candidate)
        return df

    logger.error(
        "No usable base metadata found. Need either %s or VBench CSV with video_id/group.",
        processed_metadata,
    )
    return None


def compute_group_summary(
    df: pd.DataFrame,
    metric_cols: list[str],
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

    # Custom experiment output filename (if configured)
    experiment_output = paths_config.get("experiment_output")
    if experiment_output:
        experiment_output_path = output_dir / experiment_output
    else:
        experiment_output_path = None

    # Check if already exists
    if per_video_output.exists() and group_summary_output.exists() and not args.force:
        logger.info("Summary files already exist. Use --force to regenerate.")
        return
    elif (per_video_output.exists() or group_summary_output.exists()) and args.force:
        logger.info("Force regenerating summary files")

    # Load base metadata (supports both full-pipeline and vbench-only outputs)
    base_df = load_base_metadata_for_summary(output_dir, paths_config)
    if base_df is None:
        return

    # Keep relevant columns from base
    base_cols = ["video_id", "group", "prompt", "num_frames", "duration_sec"]
    base_cols = [c for c in base_cols if c in base_df.columns]
    base_df = base_df[base_cols].drop_duplicates(subset=["video_id"])

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

    # Determine score type from clipvqa data
    score_type = "clip"  # default
    if metric_dfs.get("clipvqa") is not None and "score_type" in metric_dfs["clipvqa"].columns:
        score_type = metric_dfs["clipvqa"]["score_type"].iloc[0]

    # Rename score column to be explicit about type (clip_score or vqa_score)
    if "clip_or_vqa_score" in merged_df.columns:
        explicit_score_name = f"{score_type}_score"
        merged_df = merged_df.rename(columns={"clip_or_vqa_score": explicit_score_name})
        logger.info(f"Score column renamed: clip_or_vqa_score -> {explicit_score_name}")

    # Optional VBench comparison profile + scale alignment
    vbench_config = config.get("metrics", {}).get("vbench", {})
    comparison_profile = resolve_comparison_profile(vbench_config)
    scale_to_percent = resolve_scale_to_percent(vbench_config, comparison_profile)
    if scale_to_percent:
        scaled_cols = apply_percent_scaling(merged_df, scale_to_percent)
        if scaled_cols:
            logger.info(
                "Applied percent scaling (x100) for columns: %s",
                scaled_cols,
            )
        else:
            logger.warning(
                "metrics.vbench.scale_to_percent configured but no matching numeric columns found."
            )

    # Identify metric columns for summary
    metric_cols = [
        f"{score_type}_score",  # clip_score or vqa_score
        "vbench_temporal_score",
        "flicker_mean",
        "niqe_mean",
        "num_frames",
        "duration_sec",
    ]

    # Add configured VBench subtask columns
    configured_subtasks = config.get("metrics", {}).get("vbench", {}).get("subtasks", [])
    metric_cols.extend(configured_subtasks)

    # Also include any additional VBench columns found in vbench_per_video.csv
    vbench_df = metric_dfs.get("vbench")
    if vbench_df is not None:
        auto_vbench_cols = [
            c for c in vbench_df.columns
            if c not in ["video_id", "group", "prompt", "video_path", "vbench_temporal_score"]
        ]
        metric_cols.extend(auto_vbench_cols)

    # Add runtime columns if present
    if "fps" in merged_df.columns:
        metric_cols.append("fps")
    if "inference_time_sec" in merged_df.columns:
        metric_cols.append("inference_time_sec")

    # Filter to existing columns and de-duplicate while preserving order
    metric_cols = [c for c in dict.fromkeys(metric_cols) if c in merged_df.columns]

    # Compute group summary
    summary_df = compute_group_summary(merged_df, metric_cols)
    summary_df = cleanup_summary_column_names(summary_df)

    # Sort by group name
    summary_df = summary_df.sort_values("group").reset_index(drop=True)

    # Save group summary
    summary_df.to_csv(group_summary_output, index=False)
    logger.info(f"Group summary saved to: {group_summary_output}")

    # Save to custom experiment output path if configured
    if experiment_output_path:
        summary_df.to_csv(experiment_output_path, index=False)
        logger.info(f"Experiment output saved to: {experiment_output_path}")

    # Optional profile-specific summary export (e.g., deep_forcing_8d).
    if comparison_profile is not None:
        profile_metric_cols = resolve_profile_metric_cols(
            profile=comparison_profile,
            df_columns=list(merged_df.columns),
        )
        if profile_metric_cols:
            profile_summary_df = compute_group_summary(merged_df, profile_metric_cols)
            profile_summary_df = cleanup_summary_column_names(profile_summary_df)
            profile_summary_df = profile_summary_df.sort_values("group").reset_index(drop=True)
            profile_output_name = str(
                vbench_config.get("profile_output", f"group_summary_{comparison_profile}.csv")
            )
            profile_output_path = output_dir / profile_output_name
            profile_summary_df.to_csv(profile_output_path, index=False)
            logger.info(
                "Profile summary (%s) saved to: %s (metrics=%s)",
                comparison_profile,
                profile_output_path,
                profile_metric_cols,
            )
        else:
            logger.warning(
                "Comparison profile %s enabled but no matching columns found in merged data.",
                comparison_profile,
            )

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("GROUP SUMMARY")
    logger.info("=" * 80)

    # Format for display
    display_cols = ["group", "n_videos"]
    for col in [f"{score_type}_score", "vbench_temporal_score", "flicker", "niqe"]:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        if mean_col in summary_df.columns:
            display_cols.extend([mean_col, std_col])

    print(summary_df[display_cols].to_string(index=False))

    # Print metric directions
    score_type_upper = score_type.upper()
    logger.info("\nMetric Directions:")
    logger.info(f"  ↑ {score_type}_score: Higher is better ({score_type_upper} text-video alignment)")
    logger.info("  ↑ vbench_temporal_score: Higher is better (temporal quality)")
    logger.info("  ↓ flicker: Lower is better (temporal stability)")
    logger.info("  ↓ niqe: Lower is better (visual quality)")


if __name__ == "__main__":
    main()
