"""
Tests for scripts/vbench_runner/results.py — VBench score extraction.

Covers:
  - resolve_video_id()
  - apply_long_consistency_prefix_fallback()
  - extract_subtask_scores()
"""

import pandas as pd
import pytest

from scripts.vbench_runner.results import (
    apply_long_consistency_prefix_fallback,
    extract_subtask_scores,
    resolve_video_id,
)


# ---------------------------------------------------------------------------
# resolve_video_id
# ---------------------------------------------------------------------------
class TestResolveVideoId:
    def test_exact_stem_match(self):
        valid = {"vid_001", "vid_002"}
        result = resolve_video_id("/path/to/vid_001.mp4", valid)
        assert result == "vid_001"

    def test_parent_name_match(self):
        valid = {"vid_001", "vid_002"}
        result = resolve_video_id("/path/vid_002/clip_001.mp4", valid)
        assert result == "vid_002"

    def test_numeric_suffix_stripped(self):
        valid = {"vid_001"}
        result = resolve_video_id("/path/vid_001_3.mp4", valid)
        assert result == "vid_001"

    def test_scene_split(self):
        valid = {"vid_001"}
        result = resolve_video_id("/path/vid_001-Scene-001/clip.mp4", valid)
        assert result == "vid_001"

    def test_scene_in_parent(self):
        valid = {"vid_001"}
        result = resolve_video_id("/path/vid_001-Scene-001/some_clip.mp4", valid)
        assert result == "vid_001"

    def test_no_match_returns_stem(self):
        valid = {"vid_001"}
        result = resolve_video_id("/path/unknown_video.mp4", valid)
        assert result == "unknown_video"

    def test_empty_valid_ids(self):
        result = resolve_video_id("/path/vid_001.mp4", set())
        assert result == "vid_001"


# ---------------------------------------------------------------------------
# apply_long_consistency_prefix_fallback
# ---------------------------------------------------------------------------
class TestApplyLongConsistencyPrefixFallback:
    def test_subject_consistency_expansion(self):
        valid_ids = {"frame_001", "frame_002", "head_001"}
        parsed = []
        unresolved = [("/path/frame.mp4", 0.9)]

        result = apply_long_consistency_prefix_fallback(
            parsed, unresolved, "subject_consistency", valid_ids
        )
        # Should expand "frame" prefix to frame_001 and frame_002
        assigned_ids = {item["video_id"] for item in result}
        assert "frame_001" in assigned_ids
        assert "frame_002" in assigned_ids

    def test_non_consistency_dimension_unchanged(self):
        """Fallback should only apply to subject/background_consistency."""
        parsed = []
        unresolved = [("/path/frame.mp4", 0.9)]
        valid_ids = {"frame_001"}

        result = apply_long_consistency_prefix_fallback(
            parsed, unresolved, "motion_smoothness", valid_ids
        )
        assert len(result) == 0

    def test_no_unresolved(self):
        parsed = [{"video_id": "v1", "subtask": "subject_consistency", "score": 0.9}]
        result = apply_long_consistency_prefix_fallback(
            parsed, [], "subject_consistency", {"v1"}
        )
        assert len(result) == 1

    def test_no_prefix_match(self):
        valid_ids = {"alpha_001"}
        parsed = []
        unresolved = [("/path/beta.mp4", 0.9)]

        result = apply_long_consistency_prefix_fallback(
            parsed, unresolved, "subject_consistency", valid_ids
        )
        assert len(result) == 0

    def test_skip_already_assigned(self):
        """Should not overwrite already-parsed entries."""
        valid_ids = {"frame_001", "frame_002"}
        parsed = [{"video_id": "frame_001", "subtask": "subject_consistency", "score": 0.8}]
        unresolved = [("/path/frame.mp4", 0.95)]

        result = apply_long_consistency_prefix_fallback(
            parsed, unresolved, "subject_consistency", valid_ids
        )
        # frame_001 should keep its original score
        frame_001 = [r for r in result if r["video_id"] == "frame_001"]
        assert len(frame_001) == 1
        assert frame_001[0]["score"] == 0.8
        # frame_002 should get the expanded score
        frame_002 = [r for r in result if r["video_id"] == "frame_002"]
        assert len(frame_002) == 1
        assert frame_002[0]["score"] == 0.95

    def test_background_consistency_also_applies(self):
        valid_ids = {"head_001"}
        parsed = []
        unresolved = [("/path/head.mp4", 0.88)]

        result = apply_long_consistency_prefix_fallback(
            parsed, unresolved, "background_consistency", valid_ids
        )
        assert len(result) == 1
        assert result[0]["video_id"] == "head_001"


# ---------------------------------------------------------------------------
# extract_subtask_scores
# ---------------------------------------------------------------------------
class TestExtractSubtaskScores:
    def test_basic_extraction(self):
        data = [
            0.85,  # aggregate score
            [
                {"video_path": "/path/vid_001.mp4", "video_results": 0.9},
                {"video_path": "/path/vid_002.mp4", "video_results": 0.8},
            ],
        ]
        valid = {"vid_001", "vid_002"}
        result = extract_subtask_scores(data, "motion_smoothness", valid)
        assert len(result) == 2
        scores = {r["video_id"]: r["score"] for r in result}
        assert scores["vid_001"] == 0.9
        assert scores["vid_002"] == 0.8

    def test_score_key_fallback(self):
        """Uses 'score' key when 'video_results' is missing."""
        data = [
            0.85,
            [
                {"video_name": "/path/vid_001.mp4", "score": 0.9},
            ],
        ]
        valid = {"vid_001"}
        result = extract_subtask_scores(data, "test", valid)
        assert len(result) == 1
        assert result[0]["score"] == 0.9

    def test_unresolved_skipped(self):
        data = [
            0.85,
            [
                {"video_path": "/path/unknown.mp4", "video_results": 0.7},
            ],
        ]
        valid = {"vid_001"}
        result = extract_subtask_scores(data, "test", valid)
        assert len(result) == 0

    def test_non_list_input(self):
        result = extract_subtask_scores("not_a_list", "test", set())
        assert result == []

    def test_empty_data(self):
        result = extract_subtask_scores([], "test", set())
        assert result == []

    def test_long_mode_aggregation(self):
        """Long mode should aggregate clip-level scores per video."""
        data = [
            0.85,
            [
                {"video_path": "/path/vid_001-Scene-001/clip1.mp4", "video_results": 0.8},
                {"video_path": "/path/vid_001-Scene-002/clip2.mp4", "video_results": 0.9},
            ],
        ]
        valid = {"vid_001"}
        result = extract_subtask_scores(data, "test", valid, long_mode=True)
        assert len(result) == 1
        assert result[0]["video_id"] == "vid_001"
        assert result[0]["score"] == pytest.approx(0.85, abs=1e-4)

    def test_missing_score_skipped(self):
        data = [
            0.85,
            [
                {"video_path": "/path/vid_001.mp4"},  # no score
            ],
        ]
        valid = {"vid_001"}
        result = extract_subtask_scores(data, "test", valid)
        assert len(result) == 0

    def test_missing_video_path_skipped(self):
        data = [
            0.85,
            [
                {"video_results": 0.9},  # no video_path
            ],
        ]
        valid = {"vid_001"}
        result = extract_subtask_scores(data, "test", valid)
        assert len(result) == 0

    def test_dict_items_in_list(self):
        """Handles case where dimension_data is a flat list of dicts."""
        data = [
            {"video_path": "/path/vid_001.mp4", "video_results": 0.9},
            {"video_path": "/path/vid_002.mp4", "video_results": 0.8},
        ]
        valid = {"vid_001", "vid_002"}
        result = extract_subtask_scores(data, "test", valid)
        assert len(result) == 2
