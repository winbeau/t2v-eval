"""
Output scaling helpers for VBench result tables.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

RESERVED_COLUMNS = {"video_id", "group", "vbench_temporal_score"}


def _normalize_name_list(raw: object) -> list[str]:
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    normalized: list[str] = []
    for item in values:
        name = str(item).strip()
        if name:
            normalized.append(name)
    return list(dict.fromkeys(normalized))


def _looks_like_zero_one_scale(
    series: pd.Series,
    tolerance: float = 0.01,
    min_ratio_within: float = 0.98,
) -> bool:
    """
    Heuristic check for [0,1] scale.

    Uses a robust in-range ratio to tolerate tiny numeric noise.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False

    within_ratio = ((numeric >= -tolerance) & (numeric <= 1.0 + tolerance)).mean()
    if within_ratio < min_ratio_within:
        return False

    q01 = float(numeric.quantile(0.01))
    q99 = float(numeric.quantile(0.99))
    return q01 >= -tolerance and q99 <= 1.0 + tolerance


def _resolve_candidate_columns(
    df: pd.DataFrame,
    candidate_columns: Iterable[str] | None,
) -> list[str]:
    if candidate_columns is not None:
        return [name for name in candidate_columns if name in df.columns]
    return [name for name in df.columns if name not in RESERVED_COLUMNS]


def resolve_output_percent_columns(
    df: pd.DataFrame,
    vbench_config: dict,
    candidate_columns: Iterable[str] | None = None,
) -> list[str]:
    """
    Resolve which result columns should be multiplied by 100.

    Config:
      - output_percent_scale: bool (default true)
      - output_percent_mode: auto_01_only | explicit_list (default auto_01_only)
      - output_percent_columns: explicit list for explicit_list mode
    """
    if not bool(vbench_config.get("output_percent_scale", True)):
        return []

    mode = str(vbench_config.get("output_percent_mode", "auto_01_only")).strip().lower()
    candidates = _resolve_candidate_columns(df, candidate_columns)

    if mode == "explicit_list":
        explicit = _normalize_name_list(vbench_config.get("output_percent_columns"))
        return [name for name in explicit if name in candidates]

    # Default and fallback mode.
    selected: list[str] = []
    for name in candidates:
        numeric = pd.to_numeric(df[name], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        if _looks_like_zero_one_scale(numeric):
            selected.append(name)
    return selected


def recompute_vbench_temporal_score(
    df: pd.DataFrame,
    candidate_columns: Iterable[str] | None = None,
) -> list[str]:
    """
    Recompute vbench_temporal_score from numeric metric columns.

    Returns metric columns used in the aggregation.
    """
    candidates = _resolve_candidate_columns(df, candidate_columns)
    usable: list[str] = []
    numeric_by_col: dict[str, pd.Series] = {}
    for name in candidates:
        numeric = pd.to_numeric(df[name], errors="coerce")
        if numeric.notna().any():
            usable.append(name)
            numeric_by_col[name] = numeric

    if not usable:
        return []

    score_df = pd.DataFrame({name: numeric_by_col[name] for name in usable})
    df["vbench_temporal_score"] = score_df.mean(axis=1)
    return usable


def apply_output_percent_scaling(
    df: pd.DataFrame,
    vbench_config: dict,
    candidate_columns: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Apply configured output scaling and keep temporal aggregate consistent.

    Returns:
      (scaled_columns, temporal_score_columns_used)
    """
    scaled_columns = resolve_output_percent_columns(
        df=df,
        vbench_config=vbench_config,
        candidate_columns=candidate_columns,
    )
    for name in scaled_columns:
        df[name] = pd.to_numeric(df[name], errors="coerce") * 100.0

    temporal_cols: list[str] = []
    if scaled_columns and ("vbench_temporal_score" in df.columns):
        temporal_cols = recompute_vbench_temporal_score(
            df=df,
            candidate_columns=candidate_columns,
        )

    return scaled_columns, temporal_cols
