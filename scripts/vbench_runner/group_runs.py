"""
Deterministic VBench per-group cache helpers.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

from .group_subset import get_configured_group_names


GROUP_RUNS_DIR_NAME = "vbench_group_runs"
_REQUIRED_COLUMNS = {"video_id", "group"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def sanitize_group_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip().lower())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "group"


def get_group_runs_dir(output_dir: Path) -> Path:
    return output_dir / GROUP_RUNS_DIR_NAME


def build_group_run_file_map(config: dict, output_dir: Path) -> dict[str, Path]:
    group_names = get_configured_group_names(config)
    width = max(2, len(str(max(1, len(group_names)))))
    group_dir = get_group_runs_dir(output_dir)
    return {
        group_name: group_dir / f"{index:0{width}d}__{sanitize_group_name(group_name)}.csv"
        for index, group_name in enumerate(group_names, start=1)
    }


def ensure_group_run_cache_writable(
    *,
    config: dict,
    output_dir: Path,
    target_groups: list[str],
    force: bool,
) -> list[Path]:
    group_file_map = build_group_run_file_map(config, output_dir)
    existing_targets = [
        group_file_map[group_name]
        for group_name in target_groups
        if group_name in group_file_map and group_file_map[group_name].exists()
    ]
    if existing_targets and not force:
        raise FileExistsError(
            "Per-group VBench cache already exists for targeted groups; rerun with --force to overwrite: "
            + ", ".join(str(path) for path in existing_targets)
        )
    return [group_file_map[group_name] for group_name in target_groups if group_name in group_file_map]


def _validate_group_frame(df: pd.DataFrame, *, expected_group: str, source_path: Path) -> None:
    if df.empty:
        raise ValueError(f"Group cache file is empty: {source_path}")
    missing = sorted(_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Group cache file missing required columns {missing}: {source_path}")
    values = sorted({str(value).strip() for value in df["group"].dropna().tolist() if str(value).strip()})
    if len(values) != 1:
        raise ValueError(
            f"Group cache file must contain exactly one non-empty group value: {source_path}"
        )
    if values[0] != expected_group:
        raise ValueError(
            f"Group cache file group mismatch for {source_path}: expected {expected_group!r}, "
            f"found {values[0]!r}"
        )


def write_group_run_cache(
    df: pd.DataFrame,
    *,
    config: dict,
    output_dir: Path,
    force: bool,
) -> list[Path]:
    group_file_map = build_group_run_file_map(config, output_dir)
    if not group_file_map or df.empty or "group" not in df.columns:
        return []

    target_groups = [
        group_name for group_name in group_file_map if group_name in set(df["group"].astype(str))
    ]
    if not target_groups:
        return []

    ensure_group_run_cache_writable(
        config=config,
        output_dir=output_dir,
        target_groups=target_groups,
        force=force,
    )

    group_runs_dir = get_group_runs_dir(output_dir)
    group_runs_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for group_name in target_groups:
        group_df = df[df["group"].astype(str) == group_name].copy()
        if group_df.empty:
            continue
        target_path = group_file_map[group_name]
        group_df.to_csv(target_path, index=False)
        written.append(target_path)
    return written


def load_group_run_cache(config: dict, output_dir: Path) -> pd.DataFrame:
    group_file_map = build_group_run_file_map(config, output_dir)
    if not group_file_map:
        raise ValueError("Final group-run aggregation requires explicit YAML groups[].name entries.")

    group_runs_dir = get_group_runs_dir(output_dir)
    if not group_runs_dir.exists():
        raise FileNotFoundError(f"Group-run cache directory not found: {group_runs_dir}")

    existing_csvs = {path.resolve(): path for path in group_runs_dir.glob("*.csv")}
    expected_paths = {path.resolve(): path for path in group_file_map.values()}
    unknown_files = sorted(str(existing_csvs[path]) for path in existing_csvs if path not in expected_paths)
    if unknown_files:
        raise ValueError(
            "Unexpected CSV files found in group-run cache directory: " + ", ".join(unknown_files)
        )

    frames: list[pd.DataFrame] = []
    missing_groups: list[str] = []
    for group_name, csv_path in group_file_map.items():
        if not csv_path.exists():
            missing_groups.append(group_name)
            continue
        df = pd.read_csv(csv_path)
        _validate_group_frame(df, expected_group=group_name, source_path=csv_path)
        frames.append(df)

    if missing_groups:
        raise ValueError(
            "Group-run cache coverage incomplete. Missing groups: " + ", ".join(missing_groups)
        )

    if not frames:
        raise ValueError(f"No valid group-run cache CSVs found under {group_runs_dir}")
    return pd.concat(frames, ignore_index=True)


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
    try:
        from scripts.summarize import cleanup_summary_column_names, compute_group_summary
        from scripts.vbench_runner.group_labels import build_group_alias_map, remap_group_column
        from scripts.vbench_runner.group_subset import validate_output_file_name
        from scripts.vbench_runner.video_records import copy_outputs_to_frontend
    except ImportError:
        from summarize import cleanup_summary_column_names, compute_group_summary
        from vbench_runner.group_labels import build_group_alias_map, remap_group_column
        from vbench_runner.group_subset import validate_output_file_name
        from vbench_runner.video_records import copy_outputs_to_frontend

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
