"""Integration-style tests for 16-dimension long-mode result parsing and coverage."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from scripts.vbench_runner.core import _apply_color_coverage_policy
from scripts.vbench_runner.distributed import merge_rank_partial_results
from scripts.vbench_runner.env import get_vbench_subtasks
from scripts.vbench_runner.results import extract_subtask_scores
from scripts.vbench_runner.scaling import (
    compute_official_vbench_scores,
    compute_semantic_lite_vbench_scores,
)


LONG_16 = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "temporal_style",
    "appearance_style",
    "scene",
    "object_class",
    "multiple_objects",
    "spatial_relationship",
    "human_action",
    "color",
    "overall_consistency",
    "dynamic_degree",
    "imaging_quality",
    "aesthetic_quality",
]


def _scene_clip(video_id: str, scene_idx: int, clip_name: str) -> str:
    return f"/tmp/split_clip/{video_id}-Scene-{scene_idx:03d}/{clip_name}.mp4"


@pytest.mark.parametrize("subtask", LONG_16)
def test_extract_subtask_scores_aggregates_long_mode_for_each_dimension(subtask: str):
    data = [
        0.0,
        [
            {"video_path": _scene_clip("vid_001", 1, "clip_a"), "video_results": 0.2},
            {"video_path": _scene_clip("vid_001", 2, "clip_b"), "video_results": 0.6},
        ],
    ]

    rows = extract_subtask_scores(
        dimension_data=data,
        subtask=subtask,
        valid_video_ids={"vid_001"},
        long_mode=True,
        strict_integrity=True,
    )

    assert rows == [{"video_id": "vid_001", "subtask": subtask, "score": 0.4}]


@pytest.mark.parametrize("subtask", LONG_16)
def test_extract_subtask_scores_strict_unresolved_raises_for_each_dimension(subtask: str):
    data = [
        0.0,
        [
            {"video_path": "/tmp/split_clip/unknown/clip_000.mp4", "video_results": 0.5},
        ],
    ]

    with pytest.raises(RuntimeError, match="strict integrity violation"):
        extract_subtask_scores(
            dimension_data=data,
            subtask=subtask,
            valid_video_ids={"vid_001"},
            long_mode=True,
            strict_integrity=True,
        )


def test_get_vbench_subtasks_returns_exact_long_16_config():
    config = {
        "metrics": {
            "vbench": {
                "enabled": True,
                "backend": "vbench_long",
                "use_long": True,
                "dimension_profile": "long_16",
                "subtasks": list(LONG_16),
            }
        }
    }

    assert get_vbench_subtasks(config) == LONG_16


def test_merge_rank_partial_results_preserves_all_16_columns():
    partials = [
        pd.DataFrame(
            {
                "video_id": ["vid_001", "vid_002"],
                "subject_consistency": [0.91, 0.92],
                "background_consistency": [0.81, 0.82],
                "temporal_flickering": [0.71, 0.72],
                "motion_smoothness": [0.61, 0.62],
            }
        ),
        pd.DataFrame(
            {
                "video_id": ["vid_001", "vid_002"],
                "temporal_style": [0.55, 0.56],
                "appearance_style": [0.45, 0.46],
                "scene": [0.35, 0.36],
                "object_class": [0.25, 0.26],
            }
        ),
        pd.DataFrame(
            {
                "video_id": ["vid_001", "vid_002"],
                "multiple_objects": [0.15, 0.16],
                "spatial_relationship": [0.65, 0.66],
                "human_action": [0.75, 0.76],
                "color": [0.85, 0.86],
            }
        ),
        pd.DataFrame(
            {
                "video_id": ["vid_001", "vid_002"],
                "overall_consistency": [0.95, 0.96],
                "dynamic_degree": [0.52, 0.53],
                "imaging_quality": [68.0, 69.0],
                "aesthetic_quality": [0.58, 0.59],
            }
        ),
    ]

    merged = merge_rank_partial_results(partials)

    assert set(LONG_16).issubset(set(merged.columns))
    assert "vbench_temporal_score" in merged.columns
    assert list(merged["video_id"]) == ["vid_001", "vid_002"]
    assert merged[LONG_16].notna().all().all()


def test_strict_color_coverage_flags_missing_color_in_full_16():
    coverage_rows = [(name, 2, 2) for name in LONG_16 if name != "color"]
    coverage_rows.append(("color", 1, 2))

    issues = _apply_color_coverage_policy(
        coverage_rows=coverage_rows,
        all_subtasks=LONG_16,
        expected_count=2,
        vbench_config={"color_min_coverage_ratio": 0.5},
        record_diagnostics={"missing_groups": [], "fallback_count": 0},
        strict_integrity=True,
    )

    assert ("color", 1, 2) in issues


def test_full_16_coverage_has_no_issues_when_everything_is_present():
    coverage_rows = [(name, 3, 3) for name in LONG_16]

    issues = _apply_color_coverage_policy(
        coverage_rows=coverage_rows,
        all_subtasks=LONG_16,
        expected_count=3,
        vbench_config={},
        record_diagnostics={"missing_groups": [], "fallback_count": 0},
        strict_integrity=True,
    )

    assert issues == []


def test_full_16_coverage_flags_non_color_dimension_under_strict_mode():
    coverage_rows = [(name, 4, 4) for name in LONG_16]
    mutated = deepcopy(coverage_rows)
    scene_idx = LONG_16.index("scene")
    mutated[scene_idx] = ("scene", 3, 4)

    issues = _apply_color_coverage_policy(
        coverage_rows=mutated,
        all_subtasks=LONG_16,
        expected_count=4,
        vbench_config={},
        record_diagnostics={"missing_groups": [], "fallback_count": 0},
        strict_integrity=True,
    )

    assert ("scene", 3, 4) in issues


def test_lite_scores_survive_pipeline_shape_when_grit_dims_are_absent():
    """Lite scores should compute with 12D data (all 4 GrIT dims absent)."""
    df = pd.DataFrame(
        [
            {
                "video_id": "vid_001",
                "subject_consistency": 0.91,
                "background_consistency": 0.81,
                "temporal_flickering": 0.71,
                "motion_smoothness": 0.61,
                "temporal_style": 0.55,
                "appearance_style": 0.45,
                "scene": 0.35,
                "human_action": 0.75,
                "overall_consistency": 0.95,
                "dynamic_degree": 0.52,
                "imaging_quality": 0.68,
                "aesthetic_quality": 0.58,
            }
        ]
    )

    assert compute_official_vbench_scores(df) == []
    assert compute_semantic_lite_vbench_scores(df) == [
        "vbench_quality_score",
        "vbench_semantic_lite_score",
        "vbench_total_lite_score",
    ]
    assert "vbench_semantic_lite_score" in df.columns
    assert "vbench_total_lite_score" in df.columns
