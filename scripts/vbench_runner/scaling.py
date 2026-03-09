"""
Output scaling helpers for VBench result tables.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

RESERVED_COLUMNS = {
    "video_id",
    "group",
    "vbench_temporal_score",
    "vbench_quality_score",
    "vbench_semantic_score",
    "vbench_total_score",
    "vbench_semantic_lite_score",
    "vbench_total_lite_score",
}

# ---------------------------------------------------------------------------
# VBench official scoring constants (from third_party/VBench/scripts/constant.py)
# ---------------------------------------------------------------------------

# Mapping from our underscore column names to VBench's space-separated names.
_COL_TO_VBENCH_DIM: dict[str, str] = {
    "subject_consistency": "subject consistency",
    "background_consistency": "background consistency",
    "temporal_flickering": "temporal flickering",
    "motion_smoothness": "motion smoothness",
    "dynamic_degree": "dynamic degree",
    "aesthetic_quality": "aesthetic quality",
    "imaging_quality": "imaging quality",
    "object_class": "object class",
    "multiple_objects": "multiple objects",
    "human_action": "human action",
    "color": "color",
    "spatial_relationship": "spatial relationship",
    "scene": "scene",
    "appearance_style": "appearance style",
    "temporal_style": "temporal style",
    "overall_consistency": "overall consistency",
}

_NORMALIZE_DIC: dict[str, dict[str, float]] = {
    "subject consistency": {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering": {"Min": 0.6293, "Max": 1.0},
    "motion smoothness": {"Min": 0.706, "Max": 0.9975},
    "dynamic degree": {"Min": 0.0, "Max": 1.0},
    "aesthetic quality": {"Min": 0.0, "Max": 1.0},
    "imaging quality": {"Min": 0.0, "Max": 1.0},
    "object class": {"Min": 0.0, "Max": 1.0},
    "multiple objects": {"Min": 0.0, "Max": 1.0},
    "human action": {"Min": 0.0, "Max": 1.0},
    "color": {"Min": 0.0, "Max": 1.0},
    "spatial relationship": {"Min": 0.0, "Max": 1.0},
    "scene": {"Min": 0.0, "Max": 0.8222},
    "appearance style": {"Min": 0.0009, "Max": 0.2855},
    "temporal style": {"Min": 0.0, "Max": 0.364},
    "overall consistency": {"Min": 0.0, "Max": 0.364},
}

_DIM_WEIGHT: dict[str, float] = {
    "subject consistency": 1,
    "background consistency": 1,
    "temporal flickering": 1,
    "motion smoothness": 1,
    "aesthetic quality": 1,
    "imaging quality": 1,
    "dynamic degree": 0.5,
    "object class": 1,
    "multiple objects": 1,
    "human action": 1,
    "color": 1,
    "spatial relationship": 1,
    "scene": 1,
    "appearance style": 1,
    "temporal style": 1,
    "overall consistency": 1,
}

_QUALITY_LIST: set[str] = {
    "subject consistency",
    "background consistency",
    "temporal flickering",
    "motion smoothness",
    "aesthetic quality",
    "imaging quality",
    "dynamic degree",
}

_SEMANTIC_LIST: set[str] = {
    "object class",
    "multiple objects",
    "human action",
    "color",
    "spatial relationship",
    "scene",
    "appearance style",
    "temporal style",
    "overall consistency",
}

_QUALITY_WEIGHT = 4
_SEMANTIC_WEIGHT = 1

# All 16 column names required for official scoring.
ALL_16_COLUMNS: frozenset[str] = frozenset(_COL_TO_VBENCH_DIM.keys())
SEMANTIC_LITE_COLUMNS: frozenset[str] = frozenset(ALL_16_COLUMNS - {"color"})


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


def compute_official_vbench_scores(df: pd.DataFrame) -> list[str]:
    """
    Compute official VBench quality/semantic/total scores.

    Follows the official VBench scoring from ``third_party/VBench/scripts/cal_final_score.py``:
      1. Min-max normalize each of the 16 dimensions (fixed ranges from NORMALIZE_DIC).
      2. Apply per-dimension weights (all 1.0 except dynamic_degree = 0.5).
      3. Quality Score = sum(weighted_normalized) / sum(quality_weights) for 7 quality dims.
      4. Semantic Score = sum(weighted_normalized) / sum(semantic_weights) for 9 semantic dims.
      5. Total Score = (Quality × 4 + Semantic × 1) / 5.

    Scores are written on a [0, 100] scale.

    The function requires all 16 dimension columns to be present.  If any are
    missing, it returns an empty list and does nothing.

    Returns the list of new columns added (empty if skipped).
    """
    return _compute_vbench_score_bundle(
        df=df,
        required_columns=ALL_16_COLUMNS,
        semantic_columns={
            col for col, dim_name in _COL_TO_VBENCH_DIM.items() if dim_name in _SEMANTIC_LIST
        },
        semantic_col_name="vbench_semantic_score",
        total_col_name="vbench_total_score",
    )


def compute_semantic_lite_vbench_scores(df: pd.DataFrame) -> list[str]:
    """
    Compute VBench lite scores that exclude the `color` semantic dimension.

    The normalization logic, per-dimension weights, and total-score weighting
    remain identical to the official scoring path.
    """
    return _compute_vbench_score_bundle(
        df=df,
        required_columns=SEMANTIC_LITE_COLUMNS,
        semantic_columns={
            col
            for col, dim_name in _COL_TO_VBENCH_DIM.items()
            if dim_name in _SEMANTIC_LIST and dim_name != "color"
        },
        semantic_col_name="vbench_semantic_lite_score",
        total_col_name="vbench_total_lite_score",
    )


def _compute_vbench_score_bundle(
    *,
    df: pd.DataFrame,
    required_columns: frozenset[str] | set[str],
    semantic_columns: set[str],
    semantic_col_name: str,
    total_col_name: str,
) -> list[str]:
    present_cols = {col for col in _COL_TO_VBENCH_DIM if col in df.columns}
    if not set(required_columns).issubset(present_cols):
        return []

    quality_weighted_sum = pd.Series(0.0, index=df.index)
    quality_weight_sum = 0.0
    semantic_weighted_sum = pd.Series(0.0, index=df.index)
    semantic_weight_sum = 0.0

    for col, dim_name in _COL_TO_VBENCH_DIM.items():
        if col not in required_columns:
            continue
        raw = pd.to_numeric(df[col], errors="coerce")
        norm_range = _NORMALIZE_DIC[dim_name]
        lo, hi = norm_range["Min"], norm_range["Max"]
        if hi == lo:
            normalized = pd.Series(0.0, index=df.index)
        else:
            normalized = ((raw - lo) / (hi - lo)).clip(0.0, 1.0)
        weight = _DIM_WEIGHT[dim_name]
        weighted = normalized * weight

        if col in semantic_columns:
            semantic_weighted_sum += weighted
            semantic_weight_sum += weight
        else:
            quality_weighted_sum += weighted
            quality_weight_sum += weight

    quality_score = (quality_weighted_sum / quality_weight_sum) * 100.0
    semantic_score = (semantic_weighted_sum / semantic_weight_sum) * 100.0
    total_score = (quality_score * _QUALITY_WEIGHT + semantic_score * _SEMANTIC_WEIGHT) / (
        _QUALITY_WEIGHT + _SEMANTIC_WEIGHT
    )

    new_cols = ["vbench_quality_score", semantic_col_name, total_col_name]
    df["vbench_quality_score"] = quality_score
    df[semantic_col_name] = semantic_score
    df[total_col_name] = total_score
    return new_cols


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
