"""
Deterministic VBench per-group cache helpers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .group_subset import get_configured_group_names


GROUP_RUNS_DIR_NAME = "vbench_group_runs"
_REQUIRED_COLUMNS = {"video_id", "group"}


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
