"""
Group label helpers: stable group key vs output display alias.
"""

from __future__ import annotations

import pandas as pd


def build_group_alias_map(config: dict) -> dict[str, str]:
    """
    Build mapping: group `name` -> optional output `alias`.

    Only non-empty aliases are included. Missing aliases fall back to original name.
    """
    alias_map: dict[str, str] = {}
    groups = config.get("groups", [])
    if not isinstance(groups, list):
        return alias_map

    for item in groups:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        alias_raw = item.get("alias")
        if alias_raw is None:
            continue
        alias = str(alias_raw).strip()
        if alias:
            alias_map[name] = alias
    return alias_map


def remap_group_value(value, alias_map: dict[str, str]):
    """Map one group value by alias map while preserving NaN."""
    if pd.isna(value):
        return value
    key = str(value)
    return alias_map.get(key, value)


def remap_group_column(
    df: pd.DataFrame,
    alias_map: dict[str, str],
    group_col: str = "group",
) -> pd.DataFrame:
    """
    Return DataFrame with remapped group labels for output display.
    """
    if df.empty or group_col not in df.columns or not alias_map:
        return df
    out = df.copy()
    out[group_col] = out[group_col].apply(lambda value: remap_group_value(value, alias_map))
    return out
