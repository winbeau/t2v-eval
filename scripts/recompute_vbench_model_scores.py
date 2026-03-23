#!/usr/bin/env python3
"""
Recompute VBench model-level aggregate scores from per-video CSVs and compare
against the current group-mean values already used in tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from .vbench_runner.scaling import (
        ALL_16_COLUMNS,
        SEMANTIC_LITE_COLUMNS,
        compute_official_vbench_scores,
        compute_semantic_lite_vbench_scores,
    )
except ImportError:
    from vbench_runner.scaling import (
        ALL_16_COLUMNS,
        SEMANTIC_LITE_COLUMNS,
        compute_official_vbench_scores,
        compute_semantic_lite_vbench_scores,
    )

OLD_REQUIRED_COLUMNS = (
    "group",
    "video_id",
    "vbench_quality_score",
    "vbench_semantic_lite_score",
    "vbench_total_lite_score",
)

LITE_NEW_RENAME = {
    "vbench_quality_score": "vbench_quality_score_new",
    "vbench_semantic_lite_score": "vbench_semantic_lite_score_new",
    "vbench_total_lite_score": "vbench_total_lite_score_new",
}

OFFICIAL_NEW_RENAME = {
    "vbench_quality_score": "vbench_quality_score_new",
    "vbench_semantic_score": "vbench_semantic_score_new",
    "vbench_total_score": "vbench_total_score_new",
}

COMPARE_PAIRS = (
    ("vbench_quality_score_old", "vbench_quality_score_new"),
    ("vbench_semantic_lite_score_old", "vbench_semantic_lite_score_new"),
    ("vbench_total_lite_score_old", "vbench_total_lite_score_new"),
)


def load_vbench_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path}")
    missing_old = [col for col in OLD_REQUIRED_COLUMNS if col not in df.columns]
    if missing_old:
        raise ValueError(
            f"Missing required old-score columns in {path}: {missing_old}"
        )
    return df



def detect_score_mode(df: pd.DataFrame) -> tuple[str, list[str]]:
    present = set(df.columns)
    if ALL_16_COLUMNS.issubset(present):
        return "official16", sorted(ALL_16_COLUMNS)
    if SEMANTIC_LITE_COLUMNS.issubset(present):
        return "lite12", sorted(SEMANTIC_LITE_COLUMNS)

    missing_16 = sorted(set(ALL_16_COLUMNS) - present)
    missing_12 = sorted(set(SEMANTIC_LITE_COLUMNS) - present)
    raise ValueError(
        "Missing required raw VBench dimension columns. "
        f"For 12D lite missing={missing_12}; for 16D official missing={missing_16}"
    )



def _looks_like_zero_one_scale(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    q01 = float(numeric.quantile(0.01))
    q99 = float(numeric.quantile(0.99))
    within_ratio = ((numeric >= -0.01) & (numeric <= 1.01)).mean()
    return within_ratio >= 0.98 and q01 >= -0.01 and q99 <= 1.01



def _looks_like_percent_scale(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    q01 = float(numeric.quantile(0.01))
    q99 = float(numeric.quantile(0.99))
    within_ratio = ((numeric >= -1.0) & (numeric <= 100.5)).mean()
    return within_ratio >= 0.98 and q01 >= -1.0 and q99 <= 100.5 and q99 > 1.5



def _normalize_raw_dimension_scales(df: pd.DataFrame, raw_columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for col in raw_columns:
        series = pd.to_numeric(normalized[col], errors="coerce")
        if _looks_like_zero_one_scale(series):
            normalized[col] = series
            continue
        if _looks_like_percent_scale(series):
            normalized[col] = series / 100.0
            continue
        raise ValueError(
            f"Unsupported scale for raw dimension column {col!r}: expected 0-1 or 0-100-like values."
        )
    return normalized



def compute_old_group_scores(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("group", as_index=False)
        .agg(
            n_videos=("video_id", "nunique"),
            vbench_quality_score_old=("vbench_quality_score", "mean"),
            vbench_semantic_lite_score_old=("vbench_semantic_lite_score", "mean"),
            vbench_total_lite_score_old=("vbench_total_lite_score", "mean"),
        )
        .sort_values("group")
        .reset_index(drop=True)
    )
    return grouped



def compute_group_dimension_means(df: pd.DataFrame, raw_columns: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby("group", as_index=False)[raw_columns]
        .mean(numeric_only=True)
        .sort_values("group")
        .reset_index(drop=True)
    )
    return grouped



def compute_new_group_scores(group_means: pd.DataFrame, mode: str) -> pd.DataFrame:
    score_df = _normalize_raw_dimension_scales(group_means, [c for c in group_means.columns if c != "group"])
    score_df.insert(0, "video_id", score_df["group"])

    if mode == "official16":
        compute_official_vbench_scores(score_df)
    compute_semantic_lite_vbench_scores(score_df)

    rename_map: dict[str, str] = {}
    if mode == "official16":
        rename_map.update({
            key: value
            for key, value in OFFICIAL_NEW_RENAME.items()
            if key in score_df.columns
        })
    rename_map.update({
        key: value
        for key, value in LITE_NEW_RENAME.items()
        if key in score_df.columns
    })
    score_df = score_df.rename(columns=rename_map)

    raw_mean_rename = {
        col: f"{col}_mean_raw"
        for col in group_means.columns
        if col != "group"
    }
    score_df = score_df.rename(columns=raw_mean_rename)

    keep_cols = ["group"]
    keep_cols.extend(name for name in raw_mean_rename.values())
    for col in [
        "vbench_quality_score_new",
        "vbench_semantic_lite_score_new",
        "vbench_total_lite_score_new",
        "vbench_semantic_score_new",
        "vbench_total_score_new",
    ]:
        if col in score_df.columns:
            keep_cols.append(col)

    return score_df[keep_cols]



def build_comparison_table(path: Path) -> pd.DataFrame:
    df = load_vbench_csv(path)
    mode, raw_columns = detect_score_mode(df)
    old_scores = compute_old_group_scores(df)
    raw_means = compute_group_dimension_means(df, raw_columns)
    new_scores = compute_new_group_scores(raw_means, mode)

    result = old_scores.merge(new_scores, on="group", how="inner")
    result.insert(0, "input_name", path.stem)
    result.insert(1, "score_mode", mode)

    for old_col, new_col in COMPARE_PAIRS:
        if old_col in result.columns and new_col in result.columns:
            prefix = old_col.removesuffix("_old")
            result[f"{prefix}_delta"] = result[new_col] - result[old_col]
            result[f"{prefix}_abs_delta"] = (result[new_col] - result[old_col]).abs()

    return result.sort_values(["input_name", "group"]).reset_index(drop=True)



def build_markdown_report(comparison_df: pd.DataFrame) -> str:
    lines = ["# VBench Model-Level Recompute Comparison", ""]
    if comparison_df.empty:
        lines.append("(no data)")
        return "\n".join(lines).rstrip() + "\n"

    for input_name in sorted(comparison_df["input_name"].unique()):
        part = comparison_df[comparison_df["input_name"] == input_name].copy()
        score_mode = part["score_mode"].iloc[0]
        lines.append(f"## {input_name}")
        lines.append(f"- score_mode: `{score_mode}`")
        lines.append(f"- groups: {len(part)}")
        for metric in [
            "vbench_quality_score",
            "vbench_semantic_lite_score",
            "vbench_total_lite_score",
        ]:
            abs_col = f"{metric}_abs_delta"
            delta_col = f"{metric}_delta"
            old_col = f"{metric}_old"
            new_col = f"{metric}_new"
            if abs_col not in part.columns:
                continue
            row = part.sort_values(abs_col, ascending=False).iloc[0]
            lines.append(
                f"- max |Δ| `{metric}`: group=`{row['group']}` old={row[old_col]:.4f} "
                f"new={row[new_col]:.4f} delta={row[delta_col]:+.4f}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute model-level VBench aggregate scores and compare against current group-mean values."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Path to an existing vbench_*.csv file. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for group_score_recompute_comparison.csv and report.md",
    )
    return parser.parse_args(argv)



def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = [build_comparison_table(Path(value)) for value in args.input]
    comparison_df = pd.concat(tables, ignore_index=True)

    csv_path = output_dir / "group_score_recompute_comparison.csv"
    md_path = output_dir / "group_score_recompute_report.md"
    comparison_df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown_report(comparison_df), encoding="utf-8")

    print(f"Saved comparison CSV to {csv_path}")
    print(f"Saved comparison report to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
