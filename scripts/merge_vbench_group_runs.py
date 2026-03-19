#!/usr/bin/env python3
"""
Merge deterministic per-group VBench cache files into one final VBench export.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

try:
    from .summarize import cleanup_summary_column_names, compute_group_summary
    from .vbench_runner.group_labels import build_group_alias_map, remap_group_column
    from .vbench_runner.group_runs import load_group_run_cache
    from .vbench_runner.group_subset import validate_output_file_name
    from .vbench_runner.video_records import copy_outputs_to_frontend
except ImportError:
    from summarize import cleanup_summary_column_names, compute_group_summary
    from vbench_runner.group_labels import build_group_alias_map, remap_group_column
    from vbench_runner.group_runs import load_group_run_cache
    from vbench_runner.group_subset import validate_output_file_name
    from vbench_runner.video_records import copy_outputs_to_frontend


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    ignored = {"video_id", "group", "prompt", "video_path"}
    result: list[str] = []
    for column in df.columns:
        if column in ignored:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            result.append(column)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge per-group VBench cache files")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--merged-output",
        type=str,
        default="",
        help="Optional final merged VBench CSV file name under paths.output_dir.",
    )
    parser.add_argument(
        "--group-summary-output",
        type=str,
        default="",
        help="Optional final VBench group summary CSV file name under paths.output_dir.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing merged outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_config = config["paths"]
    output_dir = Path(paths_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_stem = Path(args.config).stem

    merged_name = (
        validate_output_file_name(args.merged_output, arg_name="--merged-output")
        if args.merged_output
        else f"vbench_{config_stem}.csv"
    )
    group_summary_name = (
        validate_output_file_name(args.group_summary_output, arg_name="--group-summary-output")
        if args.group_summary_output
        else f"group_summary_vbench_{config_stem}.csv"
    )
    merged_output_path = output_dir / merged_name
    group_summary_output_path = output_dir / group_summary_name

    existing = [path for path in [merged_output_path, group_summary_output_path] if path.exists()]
    if existing and not args.force:
        parser.error(
            "Merged VBench outputs already exist; rerun with --force to overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    merged_canonical = load_group_run_cache(config=config, output_dir=output_dir)
    alias_map = build_group_alias_map(config)
    merged_display = remap_group_column(merged_canonical.copy(), alias_map)

    metric_cols = resolve_numeric_metric_columns(merged_display)
    group_summary_df = compute_group_summary(merged_display, metric_cols)
    group_summary_df = cleanup_summary_column_names(group_summary_df)
    group_summary_df = group_summary_df.sort_values("group").reset_index(drop=True)

    merged_display.to_csv(merged_output_path, index=False)
    group_summary_df.to_csv(group_summary_output_path, index=False)
    logger.info("Merged VBench CSV saved to: %s", merged_output_path)
    logger.info("VBench group summary saved to: %s", group_summary_output_path)

    copy_outputs_to_frontend(
        output_dir=output_dir,
        paths_config={},
        vbench_output=merged_output_path,
        copy_configured_outputs=False,
        extra_csv_paths=[group_summary_output_path],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
