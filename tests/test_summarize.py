"""
Tests for scripts/summarize.py — metric aggregation and summary.

Covers:
  - load_metric_csv()
  - merge_metrics()
  - compute_group_summary()
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.summarize import compute_group_summary, load_metric_csv, merge_metrics


# ---------------------------------------------------------------------------
# load_metric_csv
# ---------------------------------------------------------------------------
class TestLoadMetricCsv:
    def test_loads_existing_csv(self, tmp_path, sample_flicker_df):
        csv_path = tmp_path / "flicker.csv"
        sample_flicker_df.to_csv(csv_path, index=False)
        result = load_metric_csv(csv_path, "Flicker")
        assert result is not None
        assert len(result) == 4
        assert "flicker_mean" in result.columns

    def test_returns_none_for_missing_file(self, tmp_path):
        result = load_metric_csv(tmp_path / "nonexistent.csv", "Missing")
        assert result is None

    def test_returns_none_for_empty_csv(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("video_id,group,flicker_mean\n")
        result = load_metric_csv(csv_path, "Empty")
        assert result is None

    def test_returns_none_for_truly_empty_file(self, tmp_path):
        csv_path = tmp_path / "blank.csv"
        csv_path.write_text("")
        result = load_metric_csv(csv_path, "Blank")
        assert result is None


# ---------------------------------------------------------------------------
# merge_metrics
# ---------------------------------------------------------------------------
class TestMergeMetrics:
    def test_basic_merge(self, sample_processed_metadata_df, sample_flicker_df, sample_niqe_df):
        base = sample_processed_metadata_df[["video_id", "group", "prompt"]]
        metric_dfs = {
            "flicker": sample_flicker_df,
            "niqe": sample_niqe_df,
        }
        result = merge_metrics(base, metric_dfs)

        assert "flicker_mean" in result.columns
        assert "niqe_mean" in result.columns
        assert len(result) == 4

    def test_merge_with_none_df(self, sample_processed_metadata_df, sample_flicker_df):
        base = sample_processed_metadata_df[["video_id", "group"]]
        metric_dfs = {
            "flicker": sample_flicker_df,
            "niqe": None,
        }
        result = merge_metrics(base, metric_dfs)
        assert "flicker_mean" in result.columns
        assert "niqe_mean" not in result.columns

    def test_merge_with_empty_df(self, sample_processed_metadata_df, sample_flicker_df):
        base = sample_processed_metadata_df[["video_id", "group"]]
        empty_df = pd.DataFrame(columns=["video_id", "group", "niqe_mean"])
        metric_dfs = {
            "flicker": sample_flicker_df,
            "niqe": empty_df,
        }
        result = merge_metrics(base, metric_dfs)
        assert "flicker_mean" in result.columns

    def test_merge_preserves_base_columns(self, sample_processed_metadata_df, sample_flicker_df):
        base = sample_processed_metadata_df[["video_id", "group", "prompt"]]
        metric_dfs = {"flicker": sample_flicker_df}
        result = merge_metrics(base, metric_dfs)
        assert "prompt" in result.columns

    def test_merge_left_join_missing_videos(self, sample_processed_metadata_df):
        """Metric data with subset of videos should produce NaN for missing."""
        base = sample_processed_metadata_df[["video_id", "group"]]
        partial = pd.DataFrame(
            {
                "video_id": ["vid_001", "vid_002"],
                "group": ["group_a", "group_a"],
                "some_metric": [1.0, 2.0],
            }
        )
        result = merge_metrics(base, {"partial": partial})
        assert result["some_metric"].isna().sum() == 2  # vid_003, vid_004 missing

    def test_merge_deduplicates_video_ids(self, sample_processed_metadata_df):
        """Duplicate video_ids in metric df should be deduplicated."""
        base = sample_processed_metadata_df[["video_id", "group"]]
        duplicated = pd.DataFrame(
            {
                "video_id": ["vid_001", "vid_001", "vid_002"],
                "group": ["group_a", "group_a", "group_a"],
                "metric_val": [1.0, 1.5, 2.0],
            }
        )
        result = merge_metrics(base, {"dup": duplicated})
        # drop_duplicates keeps first occurrence
        assert len(result) == 4


# ---------------------------------------------------------------------------
# compute_group_summary
# ---------------------------------------------------------------------------
class TestComputeGroupSummary:
    def test_basic_summary(self, sample_flicker_df):
        df = sample_flicker_df.copy()
        summary = compute_group_summary(df, ["flicker_mean"])

        assert "group" in summary.columns
        assert "n_videos" in summary.columns
        assert "flicker_mean_mean" in summary.columns
        assert "flicker_mean_std" in summary.columns
        assert len(summary) == 2  # group_a and group_b

    def test_n_videos_correct(self, sample_flicker_df):
        summary = compute_group_summary(sample_flicker_df, ["flicker_mean"])
        group_a = summary[summary["group"] == "group_a"].iloc[0]
        group_b = summary[summary["group"] == "group_b"].iloc[0]
        assert group_a["n_videos"] == 2
        assert group_b["n_videos"] == 2

    def test_mean_values_correct(self, sample_flicker_df):
        summary = compute_group_summary(sample_flicker_df, ["flicker_mean"])
        group_a = summary[summary["group"] == "group_a"].iloc[0]
        expected_mean = (0.01 + 0.02) / 2
        assert group_a["flicker_mean_mean"] == pytest.approx(expected_mean, abs=1e-4)

    def test_missing_column_skipped(self, sample_flicker_df):
        summary = compute_group_summary(sample_flicker_df, ["nonexistent_col"])
        assert "nonexistent_col_mean" not in summary.columns

    def test_multiple_metrics(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1", "v2", "v3"],
                "group": ["g1", "g1", "g2"],
                "metric_a": [1.0, 2.0, 3.0],
                "metric_b": [10.0, 20.0, 30.0],
            }
        )
        summary = compute_group_summary(df, ["metric_a", "metric_b"])
        assert "metric_a_mean" in summary.columns
        assert "metric_b_mean" in summary.columns
        assert "metric_a_std" in summary.columns
        assert "metric_b_std" in summary.columns

    def test_nan_handling(self):
        """NaN values should be excluded from mean/std computation."""
        df = pd.DataFrame(
            {
                "video_id": ["v1", "v2", "v3"],
                "group": ["g1", "g1", "g1"],
                "score": [1.0, np.nan, 3.0],
            }
        )
        summary = compute_group_summary(df, ["score"])
        assert summary.iloc[0]["score_mean"] == pytest.approx(2.0, abs=1e-4)

    def test_all_nan_produces_none(self):
        """When all values are NaN, mean and std should be None."""
        df = pd.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "group": ["g1", "g1"],
                "score": [np.nan, np.nan],
            }
        )
        summary = compute_group_summary(df, ["score"])
        assert summary.iloc[0]["score_mean"] is None
        assert summary.iloc[0]["score_std"] is None

    def test_single_video_group(self):
        """Group with one video: std should be NaN (pandas default for ddof=1)."""
        df = pd.DataFrame(
            {
                "video_id": ["v1"],
                "group": ["solo"],
                "score": [5.0],
            }
        )
        summary = compute_group_summary(df, ["score"])
        assert summary.iloc[0]["score_mean"] == pytest.approx(5.0, abs=1e-4)
        # pandas std with ddof=1 on single element → NaN
        assert pd.isna(summary.iloc[0]["score_std"]) or summary.iloc[0]["score_std"] is None
