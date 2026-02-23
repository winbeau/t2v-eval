#!/usr/bin/env python3
"""
Diagnose VBench/VBench-Long output consistency.

This is a read-only utility to compare:
  1) merged VBench CSV outputs
  2) per-subtask raw JSON outputs in vbench_results/

It helps identify whether discrepancies come from upstream evaluation outputs
or from downstream aggregation/formatting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _extract_items(raw_data: dict, subtask: str) -> list[dict]:
    blob = raw_data.get(subtask)
    if isinstance(blob, list) and len(blob) >= 2 and isinstance(blob[1], list):
        return [x for x in blob[1] if isinstance(x, dict)]
    if isinstance(blob, list):
        return [x for x in blob if isinstance(x, dict)]
    return []


def _extract_score(item: dict) -> float | None:
    value = item.get("video_results", item.get("score"))
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def diagnose_csv(main_csv: Path) -> None:
    print(f"[CSV] {main_csv} exists={main_csv.exists()}")
    if not main_csv.exists():
        return

    df = pd.read_csv(main_csv)
    print(f"[CSV] rows={len(df)} cols={len(df.columns)}")

    focus = [
        "dynamic_degree",
        "motion_smoothness",
        "overall_consistency",
        "temporal_style",
        "imaging_quality",
        "aesthetic_quality",
        "subject_consistency",
        "background_consistency",
    ]
    available = [c for c in focus if c in df.columns]
    print(f"[CSV] focus columns={available}")

    for col in available:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            print(f"  - {col}: nonnull=0")
            continue
        print(
            f"  - {col}: nonnull={len(series)} mean={series.mean():.6f} std={series.std():.6f}"
        )

    if {"overall_consistency", "temporal_style"}.issubset(df.columns):
        a = pd.to_numeric(df["overall_consistency"], errors="coerce").fillna(-9999.0)
        b = pd.to_numeric(df["temporal_style"], errors="coerce").fillna(-9999.0)
        equal_ratio = float((a.round(8) == b.round(8)).mean())
        print(f"[CSV] overall_consistency == temporal_style equal_ratio={equal_ratio:.4f}")


def diagnose_json(results_dir: Path, subtasks: list[str]) -> None:
    print(f"[JSON] results_dir={results_dir} exists={results_dir.exists()}")
    if not results_dir.exists():
        return

    for subtask in subtasks:
        path = results_dir / f"{subtask}_eval_results.json"
        print(f"\n[JSON:{subtask}] {path} exists={path.exists()}")
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        items = _extract_items(data, subtask)
        scores = [s for s in (_extract_score(item) for item in items) if s is not None]
        if scores:
            mean = sum(scores) / len(scores)
            print(f"  items={len(items)} scored={len(scores)} mean={mean:.6f}")
        else:
            print(f"  items={len(items)} scored=0")
        if items:
            sample = items[0]
            print(
                "  sample:",
                {
                    "video_path": sample.get("video_path", sample.get("video_name")),
                    "video_results": sample.get("video_results"),
                    "score": sample.get("score"),
                },
            )


def compare_overall_vs_temporal(results_dir: Path) -> None:
    overall_path = results_dir / "overall_consistency_eval_results.json"
    temporal_path = results_dir / "temporal_style_eval_results.json"
    if not (overall_path.exists() and temporal_path.exists()):
        return

    jo = json.loads(overall_path.read_text(encoding="utf-8"))
    jt = json.loads(temporal_path.read_text(encoding="utf-8"))
    overall_items = _extract_items(jo, "overall_consistency")
    temporal_items = _extract_items(jt, "temporal_style")

    def to_df(items: list[dict], score_col: str) -> pd.DataFrame:
        rows = []
        for item in items:
            vp = item.get("video_path", item.get("video_name"))
            score = _extract_score(item)
            if vp and score is not None:
                rows.append((str(vp), score))
        return pd.DataFrame(rows, columns=["video_path", score_col])

    dfo = to_df(overall_items, "overall")
    dft = to_df(temporal_items, "temporal")
    merged = dfo.merge(dft, on="video_path", how="inner")
    if len(merged) == 0:
        print("[PAIR] overall vs temporal_style: no matched video_path rows")
        return

    equal_ratio = float((merged["overall"].round(8) == merged["temporal"].round(8)).mean())
    abs_diff = (merged["overall"] - merged["temporal"]).abs()
    print(
        "[PAIR] overall vs temporal_style:",
        {
            "matched_rows": int(len(merged)),
            "equal_ratio": round(equal_ratio, 6),
            "mean_abs_diff": round(float(abs_diff.mean()), 6),
            "max_abs_diff": round(float(abs_diff.max()), 6),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose VBench alignment and aggregation consistency")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Experiment output dir, e.g. outputs/Exp-K_StaOscCompression",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional explicit CSV path; defaults to first vbench_*.csv in output dir",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.csv:
        main_csv = Path(args.csv)
    else:
        candidates = sorted(output_dir.glob("vbench_*.csv"))
        main_csv = candidates[0] if candidates else output_dir / "vbench_per_video.csv"

    diagnose_csv(main_csv)
    results_dir = output_dir / "vbench_results"
    subtasks = [
        "dynamic_degree",
        "motion_smoothness",
        "overall_consistency",
        "temporal_style",
        "imaging_quality",
        "aesthetic_quality",
        "subject_consistency",
        "background_consistency",
    ]
    diagnose_json(results_dir, subtasks)
    compare_overall_vs_temporal(results_dir)


if __name__ == "__main__":
    main()
