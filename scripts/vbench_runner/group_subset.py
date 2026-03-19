"""
Strict group-subset helpers shared by run_vbench and summarize.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def get_configured_group_names(config: dict) -> list[str]:
    groups = config.get("groups", [])
    if not isinstance(groups, list):
        return []
    names: list[str] = []
    for item in groups:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name and name not in names:
            names.append(name)
    return names


def parse_skip_groups_arg(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values = [part.strip() for part in str(raw).split(",")]
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def resolve_effective_group_subset(config: dict, raw_skip_groups: str | None) -> tuple[list[str], dict]:
    configured = get_configured_group_names(config)
    skip_groups = parse_skip_groups_arg(raw_skip_groups)
    if raw_skip_groups is not None and str(raw_skip_groups).strip() and not skip_groups:
        raise ValueError("--skip-groups did not contain any valid group names.")
    if not skip_groups:
        return configured, deepcopy(config)
    if not configured:
        raise ValueError("--skip-groups requires explicit YAML groups[].name entries.")

    unknown = [name for name in skip_groups if name not in set(configured)]
    if unknown:
        raise ValueError(
            f"Unknown groups in --skip-groups: {unknown}. Allowed groups: {configured}"
        )

    effective_names = [name for name in configured if name not in set(skip_groups)]
    if not effective_names:
        raise ValueError(
            f"--skip-groups removed all configured groups. Configured groups: {configured}"
        )

    effective_config = deepcopy(config)
    effective_config["groups"] = [
        item
        for item in config.get("groups", [])
        if isinstance(item, dict) and str(item.get("name", "")).strip() in set(effective_names)
    ]
    return effective_names, effective_config


def filter_records_to_groups(video_records: list[dict], allowed_groups: list[str]) -> list[dict]:
    allowed = set(allowed_groups)
    filtered = [
        dict(record)
        for record in video_records
        if str(record.get("group", "")).strip() in allowed
    ]
    if not filtered:
        raise ValueError(f"No video records remain after group filtering. Allowed groups: {allowed_groups}")
    return filtered


def filter_df_to_groups(df, allowed_groups: list[str], *, group_col: str = "group"):
    if group_col not in df.columns:
        return df
    allowed = set(allowed_groups)
    return df[df[group_col].astype(str).isin(allowed)].copy()


def validate_output_file_name(name: str, *, arg_name: str) -> str:
    value = str(name or "").strip()
    if not value:
        raise ValueError(f"{arg_name} must be a non-empty CSV filename.")
    if Path(value).name != value:
        raise ValueError(f"{arg_name} must be a file name only, not a path: {value}")
    if not value.lower().endswith(".csv"):
        raise ValueError(f"{arg_name} must end with .csv: {value}")
    return value
