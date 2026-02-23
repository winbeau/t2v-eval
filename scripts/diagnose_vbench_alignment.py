#!/usr/bin/env python3
"""
Diagnose VBench/VBench-Long output consistency.

This is a read-only utility to compare:
  1) merged VBench CSV outputs
  2) per-subtask raw JSON outputs in vbench_results/
  3) per-subtask full-info inputs used by VBench-Long
  4) upstream third_party/VBench implementation similarity

It helps identify whether discrepancies come from upstream evaluation outputs
or from downstream aggregation/formatting.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import yaml

VIDEO_ID_PROMPT_PATTERNS = (
    re.compile(r"^[a-z]\d+_video_\d+$"),
    re.compile(r"^video[_-]?\d+$"),
    re.compile(r"^[a-z0-9]+_video[_-]?\d+$"),
)


def emit(lines: list[str], message: str) -> None:
    print(message)
    lines.append(message)


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


def summarize_numeric_similarity(
    left: pd.Series,
    right: pd.Series,
    round_digits: int = 8,
) -> dict[str, float]:
    left_num = pd.to_numeric(left, errors="coerce").fillna(-9999.0)
    right_num = pd.to_numeric(right, errors="coerce").fillna(-9999.0)
    equal_mask = left_num.round(round_digits) == right_num.round(round_digits)
    abs_diff = (left_num - right_num).abs()
    return {
        "matched_rows": float(len(left_num)),
        "equal_ratio": float(equal_mask.mean()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
    }


def diagnose_csv(
    main_csv: Path,
    lines: list[str],
    focus_pair: tuple[str, str],
) -> pd.DataFrame | None:
    emit(lines, f"[CSV] {main_csv} exists={main_csv.exists()}")
    if not main_csv.exists():
        return None

    df = pd.read_csv(main_csv)
    emit(lines, f"[CSV] rows={len(df)} cols={len(df.columns)}")

    focus = [
        "dynamic_degree",
        "motion_smoothness",
        focus_pair[0],
        focus_pair[1],
        "imaging_quality",
        "aesthetic_quality",
        "subject_consistency",
        "background_consistency",
    ]
    focus = list(dict.fromkeys(focus))
    available = [c for c in focus if c in df.columns]
    emit(lines, f"[CSV] focus columns={available}")

    for col in available:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            emit(lines, f"  - {col}: nonnull=0")
            continue
        emit(
            lines,
            f"  - {col}: nonnull={len(series)} mean={series.mean():.6f} std={series.std():.6f}",
        )

    left_col, right_col = focus_pair
    if {left_col, right_col}.issubset(df.columns):
        sim = summarize_numeric_similarity(df[left_col], df[right_col])
        emit(
            lines,
            f"[CSV] {left_col} == {right_col} equal_ratio={sim['equal_ratio']:.4f}",
        )
    return df


def load_subtask_items(results_dir: Path, subtask: str) -> list[dict]:
    path = results_dir / f"{subtask}_eval_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return _extract_items(data, subtask)


def diagnose_json(
    results_dir: Path,
    subtasks: list[str],
    lines: list[str],
) -> None:
    emit(lines, f"[JSON] results_dir={results_dir} exists={results_dir.exists()}")
    if not results_dir.exists():
        return

    for subtask in subtasks:
        path = results_dir / f"{subtask}_eval_results.json"
        emit(lines, f"\n[JSON:{subtask}] {path} exists={path.exists()}")
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        items = _extract_items(data, subtask)
        scores = [s for s in (_extract_score(item) for item in items) if s is not None]
        if scores:
            mean = sum(scores) / len(scores)
            emit(lines, f"  items={len(items)} scored={len(scores)} mean={mean:.6f}")
        else:
            emit(lines, f"  items={len(items)} scored=0")
        if items:
            sample = items[0]
            emit(lines, "  sample:")
            emit(
                lines,
                f"    {{'video_path': {sample.get('video_path', sample.get('video_name'))!r}, "
                f"'video_results': {sample.get('video_results')!r}, "
                f"'score': {sample.get('score')!r}}}",
            )


def compare_subtask_pair(
    results_dir: Path,
    left_subtask: str,
    right_subtask: str,
    lines: list[str],
) -> dict[str, float] | None:
    left_items = load_subtask_items(results_dir, left_subtask)
    right_items = load_subtask_items(results_dir, right_subtask)
    if not left_items or not right_items:
        emit(
            lines,
            f"[PAIR] {left_subtask} vs {right_subtask}: missing one or both JSON result files",
        )
        return None

    def to_df(items: list[dict], score_col: str) -> pd.DataFrame:
        rows = []
        for item in items:
            vp = item.get("video_path", item.get("video_name"))
            score = _extract_score(item)
            if vp and score is not None:
                rows.append((str(vp), score))
        return pd.DataFrame(rows, columns=["video_path", score_col])

    left_df = to_df(left_items, "left")
    right_df = to_df(right_items, "right")
    merged = left_df.merge(right_df, on="video_path", how="inner")
    if len(merged) == 0:
        emit(lines, f"[PAIR] {left_subtask} vs {right_subtask}: no matched video_path rows")
        return None

    sim = summarize_numeric_similarity(merged["left"], merged["right"])
    emit(lines, f"[PAIR] {left_subtask} vs {right_subtask}:")
    emit(
        lines,
        (
            "  {"
            f"'matched_rows': {int(sim['matched_rows'])}, "
            f"'equal_ratio': {sim['equal_ratio']:.6f}, "
            f"'mean_abs_diff': {sim['mean_abs_diff']:.6f}, "
            f"'max_abs_diff': {sim['max_abs_diff']:.6f}"
            "}"
        ),
    )
    return sim


def looks_like_video_id_prompt(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if not text or " " in text:
        return False
    return any(pattern.match(text) for pattern in VIDEO_ID_PROMPT_PATTERNS)


def _entry_key(entry: dict, idx: int) -> str:
    video_list = entry.get("video_list")
    if isinstance(video_list, list) and video_list:
        first = Path(str(video_list[0]))
        if first.parent.name:
            return first.parent.name
    if video_list:
        return Path(str(video_list)).stem
    prompt = str(entry.get("prompt_en", "")).strip()
    return f"{idx}:{prompt}"


def _canonical_video_list(entry: dict) -> tuple[str, ...]:
    raw = entry.get("video_list")
    if isinstance(raw, list):
        values = [str(v) for v in raw]
    elif raw:
        values = [str(raw)]
    else:
        values = []
    return tuple(sorted(values))


def analyze_full_info_pair(left_entries: list[dict], right_entries: list[dict]) -> dict[str, float]:
    left_map: dict[str, dict] = {}
    right_map: dict[str, dict] = {}

    for idx, entry in enumerate(left_entries):
        left_map[_entry_key(entry, idx)] = entry
    for idx, entry in enumerate(right_entries):
        right_map[_entry_key(entry, idx)] = entry

    left_keys = set(left_map.keys())
    right_keys = set(right_map.keys())
    matched_keys = sorted(left_keys & right_keys)

    prompt_equal = 0
    video_list_equal = 0
    for key in matched_keys:
        if str(left_map[key].get("prompt_en", "")).strip() == str(
            right_map[key].get("prompt_en", "")
        ).strip():
            prompt_equal += 1
        if _canonical_video_list(left_map[key]) == _canonical_video_list(right_map[key]):
            video_list_equal += 1

    left_prompt_like = sum(
        1 for item in left_map.values() if looks_like_video_id_prompt(str(item.get("prompt_en", "")))
    )
    right_prompt_like = sum(
        1 for item in right_map.values() if looks_like_video_id_prompt(str(item.get("prompt_en", "")))
    )

    matched = len(matched_keys)
    return {
        "left_entries": float(len(left_entries)),
        "right_entries": float(len(right_entries)),
        "matched_entries": float(matched),
        "left_only_keys": float(len(left_keys - right_keys)),
        "right_only_keys": float(len(right_keys - left_keys)),
        "prompt_equal_ratio": float(prompt_equal / matched) if matched else 0.0,
        "video_list_equal_ratio": float(video_list_equal / matched) if matched else 0.0,
        "left_prompt_like_video_id_ratio": float(left_prompt_like / len(left_map)) if left_map else 0.0,
        "right_prompt_like_video_id_ratio": float(right_prompt_like / len(right_map))
        if right_map
        else 0.0,
    }


def diagnose_full_info_pair(
    output_dir: Path,
    left_subtask: str,
    right_subtask: str,
    lines: list[str],
) -> dict[str, float] | None:
    results_dir = output_dir / "vbench_results"
    left_path = results_dir / f"{left_subtask}_full_info.json"
    right_path = results_dir / f"{right_subtask}_full_info.json"
    emit(
        lines,
        f"[FULL_INFO] left={left_path} exists={left_path.exists()} right={right_path} exists={right_path.exists()}",
    )
    if not left_path.exists() or not right_path.exists():
        return None

    left_data = json.loads(left_path.read_text(encoding="utf-8"))
    right_data = json.loads(right_path.read_text(encoding="utf-8"))
    left_entries = [x for x in left_data if isinstance(x, dict)]
    right_entries = [x for x in right_data if isinstance(x, dict)]
    summary = analyze_full_info_pair(left_entries, right_entries)

    emit(
        lines,
        (
            f"[FULL_INFO] matched={int(summary['matched_entries'])} "
            f"prompt_equal_ratio={summary['prompt_equal_ratio']:.6f} "
            f"video_list_equal_ratio={summary['video_list_equal_ratio']:.6f}"
        ),
    )
    emit(
        lines,
        (
            "[FULL_INFO] prompt_like_video_id_ratio "
            f"left={summary['left_prompt_like_video_id_ratio']:.6f} "
            f"right={summary['right_prompt_like_video_id_ratio']:.6f}"
        ),
    )
    return summary


def normalize_source_for_similarity(source: str) -> str:
    normalized_lines: list[str] = []
    for raw_line in source.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        normalized_lines.append(" ".join(line.split()))
    return "\n".join(normalized_lines)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def analyze_upstream_source_similarity(project_root: Path) -> dict[str, str | float] | None:
    left_path = project_root / "third_party" / "VBench" / "vbench" / "overall_consistency.py"
    right_path = project_root / "third_party" / "VBench" / "vbench" / "temporal_style.py"
    if not left_path.exists() or not right_path.exists():
        return None

    left_raw = left_path.read_text(encoding="utf-8")
    right_raw = right_path.read_text(encoding="utf-8")
    left_norm = normalize_source_for_similarity(left_raw)
    right_norm = normalize_source_for_similarity(right_raw)

    raw_ratio = difflib.SequenceMatcher(a=left_raw, b=right_raw).ratio()
    norm_ratio = difflib.SequenceMatcher(a=left_norm, b=right_norm).ratio()

    left_dim_agnostic = re.sub(r"overall_consistency|temporal_style", "<DIM>", left_norm)
    right_dim_agnostic = re.sub(r"overall_consistency|temporal_style", "<DIM>", right_norm)
    dim_agnostic_ratio = difflib.SequenceMatcher(
        a=left_dim_agnostic,
        b=right_dim_agnostic,
    ).ratio()

    return {
        "raw_similarity_ratio": float(raw_ratio),
        "normalized_similarity_ratio": float(norm_ratio),
        "dimension_agnostic_similarity_ratio": float(dim_agnostic_ratio),
        "left_norm_sha": _hash_text(left_norm)[:12],
        "right_norm_sha": _hash_text(right_norm)[:12],
    }


def diagnose_upstream_source_similarity(
    project_root: Path,
    lines: list[str],
) -> dict[str, str | float] | None:
    result = analyze_upstream_source_similarity(project_root)
    if result is None:
        emit(lines, "[SOURCE] third_party VBench source files not found; skip source similarity check")
        return None

    emit(
        lines,
        "[SOURCE] similarity ratios: "
        f"raw={result['raw_similarity_ratio']:.6f} "
        f"normalized={result['normalized_similarity_ratio']:.6f} "
        f"dimension_agnostic={result['dimension_agnostic_similarity_ratio']:.6f}",
    )
    emit(
        lines,
        "[SOURCE] normalized_sha "
        f"overall={result['left_norm_sha']} temporal={result['right_norm_sha']}",
    )
    return result


def diagnose_config(
    config_path: Path | None,
    lines: list[str],
    left_subtask: str,
) -> None:
    if config_path is None:
        return
    emit(lines, f"[CONFIG] {config_path} exists={config_path.exists()}")
    if not config_path.exists():
        return
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    vbench = cfg.get("metrics", {}).get("vbench", {})
    backend = vbench.get("backend")
    mode = vbench.get("mode")
    profile = vbench.get("comparison_profile")
    scale_to_percent = vbench.get("scale_to_percent") or []
    scale_to_percent = [str(x) for x in scale_to_percent]
    emit(
        lines,
        f"[CONFIG] backend={backend!r} mode={mode!r} comparison_profile={profile!r}",
    )
    emit(
        lines,
        (
            f"[CONFIG] scale_to_percent includes `{left_subtask}`="
            f"{left_subtask in set(scale_to_percent)} "
            f"(len={len(scale_to_percent)})"
        ),
    )


def infer_root_cause(
    pair_stats: dict[str, float] | None,
    full_info_stats: dict[str, float] | None,
    source_stats: dict[str, str | float] | None,
) -> str:
    if pair_stats is None:
        return "insufficient_results"

    eq = pair_stats.get("equal_ratio", 0.0)
    if eq < 0.999:
        return "pair_not_identical_check_data_or_runtime_first"

    if full_info_stats and full_info_stats.get("video_list_equal_ratio", 0.0) < 0.999:
        return "full_info_mismatch_between_dimensions"

    if source_stats and float(source_stats.get("dimension_agnostic_similarity_ratio", 0.0)) >= 0.995:
        return "upstream_dimension_logic_effectively_identical_under_current_setup"

    return "likely_upstream_or_input_alignment_issue"


def write_report(path: Path, lines: list[str], root_cause: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# VBench Alignment Diagnostic Report\n\n")
        f.write("## Conclusion\n\n")
        f.write(f"- root_cause: `{root_cause}`\n\n")
        f.write("## Logs\n\n")
        f.write("```text\n")
        for line in lines:
            f.write(f"{line}\n")
        f.write("```\n")


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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file for scale/profile context",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="overall_consistency,temporal_style",
        help="Two subtasks to compare, comma separated",
    )
    parser.add_argument(
        "--skip-full-info",
        action="store_true",
        help="Skip *_full_info.json input consistency check",
    )
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Skip upstream third_party/VBench source similarity check",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent),
        help="Project root used to locate third_party/VBench",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Optional markdown report output path",
    )
    args = parser.parse_args()

    report_lines: list[str] = []
    output_dir = Path(args.output_dir)
    if args.csv:
        main_csv = Path(args.csv)
    else:
        candidates = sorted(output_dir.glob("vbench_*.csv"))
        main_csv = candidates[0] if candidates else output_dir / "vbench_per_video.csv"

    raw_pair = [p.strip() for p in str(args.pair).split(",") if p.strip()]
    if len(raw_pair) != 2:
        raise ValueError("--pair must contain exactly two subtasks separated by comma")
    left_subtask, right_subtask = raw_pair

    diagnose_csv(main_csv, report_lines, focus_pair=(left_subtask, right_subtask))
    results_dir = output_dir / "vbench_results"
    subtasks = [
        "dynamic_degree",
        "motion_smoothness",
        left_subtask,
        right_subtask,
        "imaging_quality",
        "aesthetic_quality",
        "subject_consistency",
        "background_consistency",
    ]
    subtasks = list(dict.fromkeys(subtasks))
    diagnose_json(results_dir, subtasks, report_lines)
    pair_stats = compare_subtask_pair(
        results_dir=results_dir,
        left_subtask=left_subtask,
        right_subtask=right_subtask,
        lines=report_lines,
    )

    full_info_stats = None
    if not args.skip_full_info:
        full_info_stats = diagnose_full_info_pair(
            output_dir=output_dir,
            left_subtask=left_subtask,
            right_subtask=right_subtask,
            lines=report_lines,
        )

    source_stats = None
    if not args.skip_source_check:
        source_stats = diagnose_upstream_source_similarity(
            project_root=Path(args.project_root),
            lines=report_lines,
        )

    diagnose_config(
        config_path=Path(args.config) if args.config else None,
        lines=report_lines,
        left_subtask=left_subtask,
    )

    root_cause = infer_root_cause(
        pair_stats=pair_stats,
        full_info_stats=full_info_stats,
        source_stats=source_stats,
    )
    emit(report_lines, f"[CONCLUSION] root_cause={root_cause}")

    if args.report_out:
        report_path = Path(args.report_out)
        write_report(report_path, report_lines, root_cause)
        emit(report_lines, f"[REPORT] wrote {report_path}")


if __name__ == "__main__":
    main()
