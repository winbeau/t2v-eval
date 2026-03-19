#!/usr/bin/env python3
"""
Analyze suspiciously high `scene` / `human_action` scores from VBench outputs.

This is a read-only diagnostic utility for custom long-video experiments that:
  1) reads merged `vbench_per_video.csv`
  2) inspects `scene_*` / `human_action_*` full-info and eval-result artifacts
  3) explains prompt-derived labels and matching modes
  4) compares 30s / 60s runs on the same prompt set
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vbench_runner.auxiliary import explain_scene_from_prompt
from vbench_runner.compat import load_human_action_categories, match_human_action_prompt
from vbench_runner.results import resolve_video_id


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


def _emit(lines: list[str], message: str) -> None:
    print(message)
    lines.append(message)


def strip_duration_suffix(group: str) -> str:
    return re.sub(r"-(?:\d+s)$", "", str(group or "").strip())


def infer_duration_tag(name: str) -> str:
    match = re.search(r"(\d+s)", str(name or ""))
    return match.group(1) if match else "unknown"


def load_vbench_per_video(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "vbench_per_video.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing vbench_per_video.csv under {output_dir}")
    df = pd.read_csv(path)
    required = {"video_id", "group", "scene", "human_action"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df["video_id"] = df["video_id"].astype(str)
    df["group"] = df["group"].astype(str)
    return df


def load_full_info_map(
    output_dir: Path,
    subtask: str,
    valid_video_ids: set[str],
    action_categories: list[str],
) -> pd.DataFrame:
    path = output_dir / "vbench_results" / f"{subtask}_full_info.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    entries = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        prompt = str(entry.get("prompt_en", entry.get("prompt", ""))).strip()
        if subtask == "scene":
            explain = explain_scene_from_prompt(prompt)
            label = explain["label"]
            mode = explain["mode"]
            extra = {"label_overlap": None, "label_ratio": None}
        else:
            match = match_human_action_prompt(prompt, categories=action_categories)
            label = match["label"]
            mode = match["mode"]
            extra = {
                "label_overlap": int(match["overlap"]),
                "label_ratio": float(match["ratio"]),
            }
        for video_path in entry.get("video_list", []):
            video_id = resolve_video_id(str(video_path), valid_video_ids)
            if video_id in seen:
                continue
            rows.append(
                {
                    "video_id": video_id,
                    f"{subtask}_prompt": prompt,
                    f"{subtask}_label": label,
                    f"{subtask}_mode": mode,
                    f"{subtask}_label_overlap": extra["label_overlap"],
                    f"{subtask}_label_ratio": extra["label_ratio"],
                }
            )
            seen.add(video_id)
    return pd.DataFrame(rows)


def load_eval_result_scores(output_dir: Path, subtask: str, valid_video_ids: set[str]) -> pd.DataFrame:
    path = output_dir / "vbench_results" / f"{subtask}_eval_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[tuple[str, float]] = []
    for item in _extract_items(data, subtask):
        video_path = item.get("video_path", item.get("video_name"))
        score = _extract_score(item)
        if not video_path or score is None:
            continue
        video_id = resolve_video_id(str(video_path), valid_video_ids)
        rows.append((video_id, score))
    if not rows:
        return pd.DataFrame(columns=["video_id", f"{subtask}_raw_score", f"{subtask}_raw_count"])
    df = pd.DataFrame(rows, columns=["video_id", f"{subtask}_raw_score"])
    agg = (
        df.groupby("video_id", as_index=False)[f"{subtask}_raw_score"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{subtask}_raw_score",
                "count": f"{subtask}_raw_count",
            }
        )
    )
    return agg


def parse_human_action_match_stats(output_dir: Path) -> dict[str, int] | None:
    pattern = re.compile(
        r"human_action K-400 matching: exact=(\d+) keyword=(\d+) unmatched=(\d+)"
    )
    candidates = [output_dir / "run_vbench.log", *sorted(output_dir.glob("run_vbench.rank*.log"))]
    for path in candidates:
        if not path.exists():
            continue
        matches = pattern.findall(path.read_text(encoding="utf-8", errors="ignore"))
        if matches:
            exact, keyword, unmatched = matches[-1]
            return {
                "exact": int(exact),
                "keyword": int(keyword),
                "unmatched": int(unmatched),
            }
    return None


def build_experiment_diagnostics(output_dir: Path, action_categories: list[str]) -> tuple[pd.DataFrame, dict]:
    merged = load_vbench_per_video(output_dir).copy()
    valid_video_ids = set(merged["video_id"].astype(str))
    scene_info = load_full_info_map(output_dir, "scene", valid_video_ids, action_categories)
    action_info = load_full_info_map(output_dir, "human_action", valid_video_ids, action_categories)
    scene_raw = load_eval_result_scores(output_dir, "scene", valid_video_ids)
    action_raw = load_eval_result_scores(output_dir, "human_action", valid_video_ids)

    merged = merged.merge(scene_info, on="video_id", how="left")
    merged = merged.merge(action_info, on="video_id", how="left")
    merged = merged.merge(scene_raw, on="video_id", how="left")
    merged = merged.merge(action_raw, on="video_id", how="left")

    merged["run"] = output_dir.name
    merged["duration"] = infer_duration_tag(output_dir.name)
    merged["base_group"] = merged["group"].map(strip_duration_suffix)
    merged["scene"] = pd.to_numeric(merged["scene"], errors="coerce")
    merged["human_action"] = pd.to_numeric(merged["human_action"], errors="coerce")

    meta = {
        "output_dir": str(output_dir),
        "human_action_match_stats": parse_human_action_match_stats(output_dir),
    }
    return merged, meta


def build_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["run", "duration", "group", "base_group"], as_index=False)
        .agg(
            n_videos=("video_id", "nunique"),
            scene_mean=("scene", "mean"),
            scene_std=("scene", "std"),
            human_action_mean=("human_action", "mean"),
            human_action_std=("human_action", "std"),
            scene_keyword_ratio=("scene_mode", lambda s: (s == "keyword").mean()),
            scene_preposition_ratio=("scene_mode", lambda s: (s == "preposition").mean()),
            scene_fallback_ratio=("scene_mode", lambda s: (s == "fallback").mean()),
            human_action_exact_ratio=("human_action_mode", lambda s: (s == "exact").mean()),
            human_action_keyword_ratio=("human_action_mode", lambda s: (s == "keyword").mean()),
            human_action_unmatched_ratio=("human_action_mode", lambda s: (s == "unmatched").mean()),
        )
    )
    return summary.sort_values(["run", "group"]).reset_index(drop=True)


def build_label_frequency(df: pd.DataFrame, subtask: str) -> pd.DataFrame:
    label_col = f"{subtask}_label"
    mode_col = f"{subtask}_mode"
    score_col = subtask
    freq = (
        df.groupby(["run", "group", mode_col, label_col], dropna=False, as_index=False)
        .agg(
            n_videos=("video_id", "count"),
            mean_score=(score_col, "mean"),
        )
        .sort_values(["run", "group", "n_videos", "mean_score"], ascending=[True, True, False, False])
    )
    return freq.reset_index(drop=True)


def build_mode_summary(df: pd.DataFrame, subtask: str) -> pd.DataFrame:
    mode_col = f"{subtask}_mode"
    label_col = f"{subtask}_label"
    score_col = subtask
    cols = ["run", "group", mode_col]
    summary = (
        df.groupby(cols, dropna=False, as_index=False)
        .agg(
            n_videos=("video_id", "count"),
            mean_score=(score_col, "mean"),
            unique_labels=(label_col, lambda s: s.fillna("<missing>").nunique()),
        )
        .sort_values(["run", "group", "n_videos"], ascending=[True, True, False])
    )
    return summary.reset_index(drop=True)


def build_cross_run_alignment(df: pd.DataFrame) -> pd.DataFrame:
    if df["run"].nunique() < 2:
        return pd.DataFrame()
    slim = df[["run", "duration", "group", "base_group", "video_id", "scene", "human_action"]].copy()
    pairs: list[pd.DataFrame] = []
    runs = sorted(slim["run"].unique())
    for idx, left_run in enumerate(runs):
        for right_run in runs[idx + 1 :]:
            left = slim[slim["run"] == left_run].rename(
                columns={
                    "scene": "scene_left",
                    "human_action": "human_action_left",
                    "duration": "duration_left",
                    "group": "group_left",
                }
            )
            right = slim[slim["run"] == right_run].rename(
                columns={
                    "scene": "scene_right",
                    "human_action": "human_action_right",
                    "duration": "duration_right",
                    "group": "group_right",
                }
            )
            merged = left.merge(
                right,
                on=["base_group", "video_id"],
                how="inner",
                suffixes=("_left", "_right"),
            )
            if len(merged) == 0:
                continue
            merged["left_run"] = left_run
            merged["right_run"] = right_run
            merged["scene_abs_diff"] = (merged["scene_left"] - merged["scene_right"]).abs()
            merged["human_action_abs_diff"] = (
                merged["human_action_left"] - merged["human_action_right"]
            ).abs()
            pairs.append(merged)
    if not pairs:
        return pd.DataFrame()
    return pd.concat(pairs, ignore_index=True)


def build_suspicious_samples(df: pd.DataFrame, subtask: str, top_k: int = 20) -> pd.DataFrame:
    score_col = subtask
    mode_col = f"{subtask}_mode"
    label_col = f"{subtask}_label"
    prompt_col = f"{subtask}_prompt"
    ranking = {"unmatched": 0, "fallback": 0, "keyword": 1, "preposition": 1, "exact": 2}
    suspicious = df.copy()
    suspicious["mode_rank"] = suspicious[mode_col].map(ranking).fillna(99)
    if subtask == "scene":
        suspicious = suspicious[
            suspicious[mode_col].isin(["fallback", "preposition"]) | suspicious[label_col].isin(["outdoor"])
        ]
    else:
        suspicious = suspicious[suspicious[mode_col].isin(["unmatched", "keyword"])]
    if suspicious.empty:
        suspicious = df.copy()
        suspicious["mode_rank"] = suspicious[mode_col].map(ranking).fillna(99)
    suspicious = suspicious.sort_values([score_col, "mode_rank"], ascending=[False, True])
    keep = [
        "run",
        "group",
        "video_id",
        prompt_col,
        label_col,
        mode_col,
        score_col,
        f"{subtask}_raw_score",
        f"{subtask}_raw_count",
    ]
    return suspicious[keep].head(top_k).reset_index(drop=True)


def build_report(
    df: pd.DataFrame,
    group_summary: pd.DataFrame,
    scene_modes: pd.DataFrame,
    human_modes: pd.DataFrame,
    scene_labels: pd.DataFrame,
    human_labels: pd.DataFrame,
    cross_run: pd.DataFrame,
    run_meta: dict[str, dict],
) -> str:
    lines: list[str] = []
    _emit(lines, "# Scene / Human Action Audit")
    _emit(lines, "")
    _emit(lines, "## Summary")
    _emit(
        lines,
        f"- Runs: {', '.join(sorted(df['run'].unique()))}",
    )
    _emit(lines, f"- Total rows inspected: {len(df)}")
    _emit(lines, "")

    _emit(lines, "## Group Summary")
    for row in group_summary.itertuples(index=False):
        _emit(
            lines,
            (
                f"- {row.run} / {row.group}: scene={row.scene_mean:.3f}±{row.scene_std:.3f}, "
                f"human_action={row.human_action_mean:.3f}±{row.human_action_std:.3f}, "
                f"scene_fallback_ratio={row.scene_fallback_ratio:.3f}, "
                f"human_action_unmatched_ratio={row.human_action_unmatched_ratio:.3f}"
            ),
        )
    _emit(lines, "")

    _emit(lines, "## Human Action Match Modes")
    for run_name in sorted(df["run"].unique()):
        stats = run_meta.get(run_name, {}).get("human_action_match_stats")
        if stats:
            _emit(
                lines,
                f"- {run_name} log stats: exact={stats['exact']} keyword={stats['keyword']} unmatched={stats['unmatched']}",
            )
        subset = human_modes[human_modes["run"] == run_name]
        for row in subset.itertuples(index=False):
            _emit(
                lines,
                f"  - {row.group} / {getattr(row, 'human_action_mode')}: n={row.n_videos} mean={row.mean_score:.3f}",
            )
    _emit(lines, "")

    _emit(lines, "## Scene Labeling Modes")
    for run_name in sorted(df["run"].unique()):
        subset = scene_modes[scene_modes["run"] == run_name]
        for row in subset.itertuples(index=False):
            _emit(
                lines,
                f"  - {row.group} / {getattr(row, 'scene_mode')}: n={row.n_videos} mean={row.mean_score:.3f}",
            )
    _emit(lines, "")

    _emit(lines, "## Top Scene Labels")
    for row in scene_labels.head(12).itertuples(index=False):
        _emit(
            lines,
            f"- {row.run} / {row.group} / {getattr(row, 'scene_mode')} / {getattr(row, 'scene_label')}: n={row.n_videos} mean={row.mean_score:.3f}",
        )
    _emit(lines, "")

    _emit(lines, "## Top Human Action Labels")
    for row in human_labels.head(12).itertuples(index=False):
        _emit(
            lines,
            f"- {row.run} / {row.group} / {getattr(row, 'human_action_mode')} / {getattr(row, 'human_action_label')}: n={row.n_videos} mean={row.mean_score:.3f}",
        )
    _emit(lines, "")

    if not cross_run.empty:
        summary = (
            cross_run.groupby(["left_run", "right_run", "base_group"], as_index=False)
            .agg(
                matched_videos=("video_id", "count"),
                scene_mean_abs_diff=("scene_abs_diff", "mean"),
                human_action_mean_abs_diff=("human_action_abs_diff", "mean"),
            )
            .sort_values(["left_run", "right_run", "base_group"])
        )
        _emit(lines, "## Cross-Run Alignment")
        for row in summary.itertuples(index=False):
            _emit(
                lines,
                f"- {row.base_group}: {row.left_run} vs {row.right_run}, matched={row.matched_videos}, "
                f"scene_mean_abs_diff={row.scene_mean_abs_diff:.3f}, "
                f"human_action_mean_abs_diff={row.human_action_mean_abs_diff:.3f}",
            )
        _emit(lines, "")

    _emit(lines, "## Initial Read")
    overall_scene = group_summary["scene_fallback_ratio"].mean()
    overall_unmatched = group_summary["human_action_unmatched_ratio"].mean()
    _emit(
        lines,
        f"- Average scene fallback ratio across groups: {overall_scene:.3f}",
    )
    _emit(
        lines,
        f"- Average human_action unmatched ratio across groups: {overall_unmatched:.3f}",
    )
    _emit(
        lines,
        "- If high-scoring rows cluster in scene fallback/preposition modes or human_action unmatched mode, "
        "the scores are likely being driven more by prompt heuristics than by robust semantic grounding.",
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit scene/human_action VBench dimensions.")
    parser.add_argument(
        "--output-dir",
        action="append",
        required=True,
        help="Experiment output directory containing vbench_per_video.csv and vbench_results/",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="outputs/scene_human_action_audit",
        help="Directory to store analysis CSVs and report",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help="Optional explicit path for Markdown report (defaults to <artifacts-dir>/report.md)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_out = Path(args.report_out) if args.report_out else artifacts_dir / "report.md"

    action_categories = load_human_action_categories()
    all_rows: list[pd.DataFrame] = []
    run_meta: dict[str, dict] = {}
    for raw_output_dir in args.output_dir:
        output_dir = Path(raw_output_dir)
        df, meta = build_experiment_diagnostics(output_dir, action_categories)
        all_rows.append(df)
        run_meta[output_dir.name] = meta

    combined = pd.concat(all_rows, ignore_index=True)
    group_summary = build_group_summary(combined)
    scene_modes = build_mode_summary(combined, "scene")
    human_modes = build_mode_summary(combined, "human_action")
    scene_labels = build_label_frequency(combined, "scene")
    human_labels = build_label_frequency(combined, "human_action")
    cross_run = build_cross_run_alignment(combined)
    scene_suspicious = build_suspicious_samples(combined, "scene")
    human_suspicious = build_suspicious_samples(combined, "human_action")

    combined.to_csv(artifacts_dir / "per_video_diagnostics.csv", index=False)
    group_summary.to_csv(artifacts_dir / "group_dimension_summary.csv", index=False)
    scene_modes.to_csv(artifacts_dir / "scene_mode_summary.csv", index=False)
    human_modes.to_csv(artifacts_dir / "human_action_mode_summary.csv", index=False)
    scene_labels.to_csv(artifacts_dir / "scene_label_frequency.csv", index=False)
    human_labels.to_csv(artifacts_dir / "human_action_label_frequency.csv", index=False)
    if not cross_run.empty:
        cross_run.to_csv(artifacts_dir / "cross_run_alignment.csv", index=False)
    scene_suspicious.to_csv(artifacts_dir / "scene_suspicious_top20.csv", index=False)
    human_suspicious.to_csv(artifacts_dir / "human_action_suspicious_top20.csv", index=False)

    report = build_report(
        combined,
        group_summary,
        scene_modes,
        human_modes,
        scene_labels,
        human_labels,
        cross_run,
        run_meta,
    )
    report_out.write_text(report, encoding="utf-8")
    print(f"Saved report to {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
