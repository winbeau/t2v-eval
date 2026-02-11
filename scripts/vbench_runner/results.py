"""
Score extraction and video ID resolution from VBench output.
"""

import re
from pathlib import Path

import pandas as pd

try:
    from .env import logger
except ImportError:
    from vbench_runner.env import logger


def resolve_video_id(video_path: str, valid_video_ids: set[str]) -> str:
    """
    Resolve a VBench/VBench-Long output path to original video_id.

    Long-mode outputs are often clip-level paths, so we attempt multiple candidates.
    """
    path = Path(video_path)
    stem = path.stem
    candidates = [stem, path.parent.name]

    if stem.rsplit("_", 1)[-1].isdigit():
        candidates.append(stem.rsplit("_", 1)[0])
    if "-Scene" in stem:
        candidates.append(stem.split("-Scene")[0])
    if "-Scene" in path.parent.name:
        candidates.append(path.parent.name.split("-Scene")[0])

    for candidate in candidates:
        if candidate in valid_video_ids:
            return candidate

    return stem


def apply_long_consistency_prefix_fallback(
    parsed_items: list[dict],
    unresolved_items: list[tuple[str, float]],
    subtask: str,
    valid_video_ids: set[str],
) -> list[dict]:
    """
    Fallback for VBench-Long subject/background id truncation.

    Some upstream outputs collapse video ids to coarse prefixes
    (e.g. frame/head/osc/stable). When this happens, expand prefix
    scores to all matching video_ids in current dataset.
    """
    if subtask not in {"subject_consistency", "background_consistency"}:
        return parsed_items
    if not unresolved_items:
        return parsed_items

    prefix_to_video_ids: dict[str, list[str]] = {}
    for video_id in sorted(valid_video_ids):
        prefix = video_id.split("_", 1)[0].lower()
        prefix_to_video_ids.setdefault(prefix, []).append(video_id)

    prefix_scores: dict[str, list[float]] = {}
    for video_path, score in unresolved_items:
        stem = Path(str(video_path)).stem.lower()
        prefix = stem.split("_", 1)[0]
        if prefix in prefix_to_video_ids:
            prefix_scores.setdefault(prefix, []).append(float(score))

    if not prefix_scores:
        return parsed_items

    existing_video_ids = {item["video_id"] for item in parsed_items}
    added = 0
    for prefix, scores in prefix_scores.items():
        mean_score = float(sum(scores) / len(scores))
        for video_id in prefix_to_video_ids[prefix]:
            if video_id in existing_video_ids:
                continue
            parsed_items.append(
                {
                    "video_id": video_id,
                    "subtask": subtask,
                    "score": mean_score,
                }
            )
            existing_video_ids.add(video_id)
            added += 1

    if added > 0:
        logger.info(
            "[%s] prefix fallback expanded %d unresolved entries to %d video assignments",
            subtask,
            len(unresolved_items),
            added,
        )
    return parsed_items


def extract_subtask_scores(
    dimension_data,
    subtask: str,
    valid_video_ids: set[str],
    long_mode: bool = False,
) -> list[dict]:
    """Extract per-video scores from one subtask result blob."""
    if not isinstance(dimension_data, list):
        return []

    per_video_items = []
    if len(dimension_data) >= 2 and isinstance(dimension_data[1], list):
        per_video_items = dimension_data[1]
    elif isinstance(dimension_data, list):
        per_video_items = [item for item in dimension_data if isinstance(item, dict)]

    parsed_items = []
    unresolved_count = 0
    unresolved_items: list[tuple[str, float]] = []
    for item in per_video_items:
        if not isinstance(item, dict):
            continue
        video_path = item.get("video_path", item.get("video_name", ""))
        score = item.get("video_results", item.get("score"))
        if not video_path or score is None:
            continue
        resolved_video_id = resolve_video_id(str(video_path), valid_video_ids)
        if resolved_video_id not in valid_video_ids:
            unresolved_count += 1
            unresolved_items.append((str(video_path), float(score)))
            continue
        parsed_items.append(
            {
                "video_id": resolved_video_id,
                "subtask": subtask,
                "score": float(score),
            }
        )

    if long_mode:
        parsed_items = apply_long_consistency_prefix_fallback(
            parsed_items=parsed_items,
            unresolved_items=unresolved_items,
            subtask=subtask,
            valid_video_ids=valid_video_ids,
        )

    if unresolved_count > 0:
        logger.info(
            f"[{subtask}] skipped {unresolved_count} unresolved video ids from raw VBench output"
        )

    if not long_mode:
        return parsed_items

    # VBench-Long can emit clip-level results; aggregate to per-original-video.
    if not parsed_items:
        return parsed_items

    df_long = pd.DataFrame(parsed_items)
    agg = (
        df_long.groupby(["video_id", "subtask"], as_index=False)["score"]
        .mean()
        .round(6)
    )
    return agg.to_dict("records")
