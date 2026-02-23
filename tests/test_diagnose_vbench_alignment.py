"""Tests for scripts/diagnose_vbench_alignment.py."""

import pandas as pd
import pytest

from scripts.diagnose_vbench_alignment import (
    analyze_full_info_pair,
    infer_root_cause,
    looks_like_video_id_prompt,
    normalize_source_for_similarity,
    summarize_numeric_similarity,
)


def test_summarize_numeric_similarity_identical() -> None:
    left = pd.Series([0.1, 0.2, None])
    right = pd.Series([0.1, 0.2, None])

    result = summarize_numeric_similarity(left, right)

    assert result["matched_rows"] == 3.0
    assert result["equal_ratio"] == pytest.approx(1.0)
    assert result["mean_abs_diff"] == pytest.approx(0.0)
    assert result["max_abs_diff"] == pytest.approx(0.0)


def test_summarize_numeric_similarity_detects_gap() -> None:
    left = pd.Series([0.1, 0.2, 0.3])
    right = pd.Series([0.1, 0.25, 0.3])

    result = summarize_numeric_similarity(left, right)

    assert result["equal_ratio"] < 1.0
    assert result["mean_abs_diff"] > 0.0


def test_looks_like_video_id_prompt() -> None:
    assert looks_like_video_id_prompt("g1_video_000")
    assert looks_like_video_id_prompt("video_123")
    assert looks_like_video_id_prompt("k6_video-42")
    assert not looks_like_video_id_prompt("a cat running on the beach")


def test_analyze_full_info_pair_identity() -> None:
    left_entries = [
        {
            "prompt_en": "a cat running",
            "video_list": ["/tmp/split_clip/g1_video_000/g1_video_000_000.mp4"],
        },
        {
            "prompt_en": "a dog jumping",
            "video_list": ["/tmp/split_clip/g1_video_001/g1_video_001_000.mp4"],
        },
    ]
    right_entries = [
        {
            "prompt_en": "a cat running",
            "video_list": ["/tmp/split_clip/g1_video_000/g1_video_000_000.mp4"],
        },
        {
            "prompt_en": "a dog jumping",
            "video_list": ["/tmp/split_clip/g1_video_001/g1_video_001_000.mp4"],
        },
    ]

    result = analyze_full_info_pair(left_entries, right_entries)

    assert result["matched_entries"] == 2.0
    assert result["prompt_equal_ratio"] == pytest.approx(1.0)
    assert result["video_list_equal_ratio"] == pytest.approx(1.0)
    assert result["left_prompt_like_video_id_ratio"] == pytest.approx(0.0)


def test_analyze_full_info_pair_detects_prompt_id_bias() -> None:
    left_entries = [
        {
            "prompt_en": "g1_video_000",
            "video_list": ["/tmp/split_clip/g1_video_000/g1_video_000_000.mp4"],
        }
    ]
    right_entries = [
        {
            "prompt_en": "a cat running",
            "video_list": ["/tmp/split_clip/g1_video_000/g1_video_000_000.mp4"],
        }
    ]

    result = analyze_full_info_pair(left_entries, right_entries)

    assert result["matched_entries"] == 1.0
    assert result["prompt_equal_ratio"] == pytest.approx(0.0)
    assert result["video_list_equal_ratio"] == pytest.approx(1.0)
    assert result["left_prompt_like_video_id_ratio"] == pytest.approx(1.0)


def test_normalize_source_for_similarity_strips_comments_and_whitespace() -> None:
    source = """
# comment line

def foo(a, b):  # inline comment
    return a + b
"""
    normalized = normalize_source_for_similarity(source)

    assert "comment" not in normalized
    assert "def foo(a, b):" in normalized
    assert "return a + b" in normalized


def test_infer_root_cause_prefers_upstream_identical_logic() -> None:
    pair_stats = {"equal_ratio": 1.0}
    full_info_stats = {"video_list_equal_ratio": 1.0}
    source_stats = {"dimension_agnostic_similarity_ratio": 1.0}

    result = infer_root_cause(pair_stats, full_info_stats, source_stats)

    assert result == "upstream_dimension_logic_effectively_identical_under_current_setup"
