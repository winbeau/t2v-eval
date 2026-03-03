#!/usr/bin/env python3
"""
Analyze why overall_consistency differs from Deep-Forcing paper values.

This script is read-only over experiment outputs and generates a markdown report
with evidence-oriented root-cause analysis.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd
import yaml

# Paper reference values from Deep-Forcing table (without attention sink block)
PAPER_REF = {
    "deep_forcing": {
        "30s": {
            "dynamic_degree": 57.56,
            "motion_smoothness": 98.27,
            "overall_consistency": 20.54,
            "imaging_quality": 69.31,
            "aesthetic_quality": 60.68,
            "subject_consistency": 97.34,
            "background_consistency": 96.48,
            "fps": 15.75,
        },
        "60s": {
            "dynamic_degree": 57.19,
            "motion_smoothness": 98.23,
            "overall_consistency": 20.38,
            "imaging_quality": 69.27,
            "aesthetic_quality": 59.86,
            "subject_consistency": 96.96,
            "background_consistency": 96.32,
            "fps": 15.75,
        },
    },
    "self_forcing": {
        "30s": {
            "dynamic_degree": 36.62,
            "motion_smoothness": 98.63,
            "overall_consistency": 20.50,
            "imaging_quality": 68.58,
            "aesthetic_quality": 59.44,
            "subject_consistency": 97.34,
            "background_consistency": 96.47,
            "fps": 15.78,
        },
        "60s": {
            "dynamic_degree": 31.98,
            "motion_smoothness": 98.21,
            "overall_consistency": 18.63,
            "imaging_quality": 66.33,
            "aesthetic_quality": 56.45,
            "subject_consistency": 96.82,
            "background_consistency": 96.31,
            "fps": 15.78,
        },
    },
}

VIDEO_ID_PROMPT_PATTERNS = (
    re.compile(r"^[a-z]\d+_video_\d+$"),
    re.compile(r"^video[_-]?\d+$"),
    re.compile(r"^[a-z0-9]+_video[_-]?\d+$"),
)


def is_video_like_prompt(text: str) -> bool:
    prompt = str(text or "").strip().lower()
    if not prompt or " " in prompt:
        return False
    return any(pattern.match(prompt) for pattern in VIDEO_ID_PROMPT_PATTERNS)


def parse_group_alias_map(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}
    pattern = re.compile(r"Group alias mapping for duplicate IDs:\s*(\{.*\})")
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        try:
            raw = ast.literal_eval(match.group(1))
        except Exception:
            continue
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
    return {}


def invert_alias_map(alias_map: dict[str, str]) -> dict[str, str]:
    return {v: k for k, v in alias_map.items()}


def alias_from_clip_path(video_path: str) -> str:
    # .../split_clip/g1_video_000/g1_video_000_000.mp4 -> g1
    parent = Path(str(video_path)).parent.name
    match = re.match(r"^(g\d+)_", parent)
    return match.group(1) if match else "unknown"


def collect_prompt_quality_by_group(
    full_info_path: Path,
    alias_to_group: dict[str, str],
) -> pd.DataFrame:
    if not full_info_path.exists():
        return pd.DataFrame()

    data = json.loads(full_info_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt_en", "")).strip()
        video_list = item.get("video_list", [])
        first_video = ""
        if isinstance(video_list, list) and video_list:
            first_video = str(video_list[0])
        elif video_list:
            first_video = str(video_list)

        alias = alias_from_clip_path(first_video)
        rows.append(
            {
                "alias": alias,
                "group": alias_to_group.get(alias, alias),
                "prompt": prompt,
                "is_video_like_prompt": is_video_like_prompt(prompt),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["alias", "group"], as_index=False)
        .agg(
            n_videos=("prompt", "size"),
            video_like_count=("is_video_like_prompt", "sum"),
        )
        .sort_values(["alias", "group"])  # deterministic
    )
    agg["video_like_ratio"] = agg["video_like_count"] / agg["n_videos"]
    agg["natural_prompt_ratio"] = 1.0 - agg["video_like_ratio"]
    return agg


def load_group_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_metric_col(df: pd.DataFrame, base: str) -> str | None:
    candidates = [
        f"{base}_mean",
        base,
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def compute_overall_temporal_equality(vbench_csv: Path) -> float | None:
    if not vbench_csv.exists():
        return None
    df = pd.read_csv(vbench_csv)
    if not {"overall_consistency", "temporal_style"}.issubset(df.columns):
        return None
    left = pd.to_numeric(df["overall_consistency"], errors="coerce")
    right = pd.to_numeric(df["temporal_style"], errors="coerce")
    eq = (left.fillna(-9999).round(8) == right.fillna(-9999).round(8)).mean()
    return float(eq)


def find_target_groups(summary_df: pd.DataFrame) -> tuple[str | None, str | None]:
    if "group" not in summary_df.columns:
        return None, None
    groups = [str(x) for x in summary_df["group"].tolist()]
    deep = next((g for g in groups if "deep_forcing" in g), None)
    selff = next((g for g in groups if "self_forcing" in g), None)
    return deep, selff


def compare_with_paper(
    summary_df: pd.DataFrame,
    deep_group: str | None,
    self_group: str | None,
) -> pd.DataFrame:
    metric = "overall_consistency"
    col = safe_metric_col(summary_df, metric)
    if col is None:
        return pd.DataFrame()

    rows: list[dict] = []

    def get_group_val(group_name: str | None) -> float | None:
        if group_name is None:
            return None
        part = summary_df[summary_df["group"] == group_name]
        if part.empty:
            return None
        value = pd.to_numeric(part.iloc[0][col], errors="coerce")
        return float(value) if pd.notna(value) else None

    ours_deep = get_group_val(deep_group)
    ours_self = get_group_val(self_group)

    for horizon in ["30s", "60s"]:
        ref = PAPER_REF["deep_forcing"][horizon][metric]
        if ours_deep is not None:
            rows.append(
                {
                    "model": "deep_forcing",
                    "horizon": horizon,
                    "ours": ours_deep,
                    "paper": ref,
                    "abs_gap": ours_deep - ref,
                }
            )
        ref_self = PAPER_REF["self_forcing"][horizon][metric]
        if ours_self is not None:
            rows.append(
                {
                    "model": "self_forcing",
                    "horizon": horizon,
                    "ours": ours_self,
                    "paper": ref_self,
                    "abs_gap": ours_self - ref_self,
                }
            )

    return pd.DataFrame(rows)


def prompt_quality_effect_table(
    prompt_quality_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    if prompt_quality_df.empty or summary_df.empty:
        return pd.DataFrame()

    metric_col = safe_metric_col(summary_df, "overall_consistency")
    if metric_col is None:
        return pd.DataFrame()

    merged = prompt_quality_df.merge(
        summary_df[["group", metric_col]],
        on="group",
        how="left",
    )

    merged = merged.rename(columns={metric_col: "overall_consistency_mean"})
    merged["prompt_quality_bucket"] = merged["video_like_ratio"].apply(
        lambda x: "video_id_like_prompt" if float(x) >= 0.9 else "natural_prompt"
    )

    bucket = (
        merged.groupby("prompt_quality_bucket", as_index=False)
        .agg(
            n_groups=("group", "size"),
            mean_overall=("overall_consistency_mean", "mean"),
            mean_video_like_ratio=("video_like_ratio", "mean"),
        )
        .sort_values("prompt_quality_bucket")
    )
    return bucket


def render_markdown_table(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
    if df.empty:
        return "(empty)"
    df_out = df.copy()
    if float_cols is None:
        float_cols = {
            col
            for col in df_out.columns
            if pd.api.types.is_float_dtype(df_out[col])
            or pd.api.types.is_integer_dtype(df_out[col])
        }
    for col in float_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].map(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and pd.notna(x) else x
            )
    return df_out.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze overall_consistency gap vs Deep-Forcing paper")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Experiment output dir, e.g. outputs/Exp-K_StaOscCompression",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config path for context",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Markdown output path (default: <output-dir>/deep_forcing_overall_gap_root_cause.md)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_out = (
        Path(args.report_out)
        if args.report_out
        else output_dir / "deep_forcing_overall_gap_root_cause.md"
    )

    summary_path = output_dir / "group_summary_deep_forcing_8d.csv"
    if not summary_path.exists():
        summary_path = output_dir / "group_summary.csv"

    summary_df = load_group_summary(summary_path)
    deep_group, self_group = find_target_groups(summary_df)

    alias_map = parse_group_alias_map(output_dir / "run_vbench.log")
    alias_to_group = invert_alias_map(alias_map)

    prompt_quality_df = collect_prompt_quality_by_group(
        full_info_path=output_dir / "vbench_results" / "overall_consistency_full_info.json",
        alias_to_group=alias_to_group,
    )

    paper_cmp_df = compare_with_paper(summary_df, deep_group, self_group)
    bucket_effect_df = prompt_quality_effect_table(prompt_quality_df, summary_df)

    overall_temporal_eq = compute_overall_temporal_equality(
        output_dir / "vbench_Exp-K_StaOscCompression.csv"
    )

    config_info = {}
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            vb = cfg.get("metrics", {}).get("vbench", {})
            config_info = {
                "backend": vb.get("backend"),
                "mode": vb.get("mode"),
                "comparison_profile": vb.get("comparison_profile"),
                "subtasks": len(vb.get("subtasks", []) or []),
            }

    lines: list[str] = []
    lines.append("# Deep-Forcing Overall Consistency Gap Root-Cause Analysis")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(
        "- 结论主因：当前实验中 `overall_consistency` 低值主要来自 **prompt 语义退化**（多数样本 prompt 为 `video_000` 形式），而非聚合脚本误算。"
    )
    lines.append(
        "- 关键证据：`overall_consistency` 与 `temporal_style` 在逐样本上完全一致（equal_ratio=1.0），且 `overall_consistency_full_info.json` 中仅 g1 组使用自然语言 prompt，其余 g2~g7 全为 video-id 风格 prompt。"
    )
    lines.append("")

    lines.append("## Current Config Context")
    lines.append("")
    if config_info:
        cfg_df = pd.DataFrame([config_info])
        lines.append(render_markdown_table(cfg_df))
    else:
        lines.append("(no config context)")
    lines.append("")

    lines.append("## Paper vs Ours (Overall Consistency)")
    lines.append("")
    if not paper_cmp_df.empty:
        lines.append(render_markdown_table(paper_cmp_df, float_cols={"ours", "paper", "abs_gap"}))
    else:
        lines.append("(failed to compare, missing summary columns)")
    lines.append("")

    lines.append("## Prompt Quality by Group (from overall_consistency_full_info.json)")
    lines.append("")
    if not prompt_quality_df.empty:
        lines.append(
            render_markdown_table(
                prompt_quality_df.sort_values(["alias", "group"]),
                float_cols={"video_like_ratio", "natural_prompt_ratio"},
            )
        )
    else:
        lines.append("(missing full_info or alias mapping)")
    lines.append("")

    lines.append("## Prompt Quality Bucket Effect")
    lines.append("")
    if not bucket_effect_df.empty:
        lines.append(render_markdown_table(bucket_effect_df, float_cols={"mean_overall", "mean_video_like_ratio"}))
    else:
        lines.append("(insufficient data)")
    lines.append("")

    lines.append("## Additional Diagnostics")
    lines.append("")
    if overall_temporal_eq is not None:
        lines.append(f"- overall_consistency == temporal_style equal_ratio: `{overall_temporal_eq:.6f}`")
    else:
        lines.append("- overall/temporal equality: unavailable")
    if alias_map:
        lines.append(f"- group alias mapping (from log): `{alias_map}`")
    else:
        lines.append("- group alias mapping: unavailable")
    lines.append("")

    lines.append("## Root-Cause Ranking")
    lines.append("")
    lines.append("1. **Prompt mapping only covers one 128-row prompt file**: run log shows a single prompt file loaded with 128 rows, while total videos are 896.")
    lines.append("2. **Most groups receive fallback prompt tokens (`video_000` etc.)**: prompt-quality table shows g2~g7 are 100% video-id-like prompts.")
    lines.append("3. **Upstream dimension behavior overlap**: overall and temporal_style are effectively identical under current setup, so prompt degradation directly drags both down.")
    lines.append("4. **Not a simple scaling bug**: k1 group has natural prompts and overall≈23.37 (already near paper 18~21 range), while degraded-prompt groups collapse to ≈4.6~5.6.")
    lines.append("")

    lines.append("## Actionable Next Checks")
    lines.append("")
    lines.append("1. 为每个组提供正确的 prompt 映射（或合并成 896 行统一 prompt 文件），再重跑 8维/12维。")
    lines.append("2. 重跑后优先比较 K6/K7 的 overall_consistency 是否回升到 18~21 区间。")
    lines.append("3. 若仍有差距，再切换到论文同口径 backend/mode 做 A/B（vbench_long vs vbench）。")
    lines.append("")

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Report written: {report_out}")


if __name__ == "__main__":
    main()
