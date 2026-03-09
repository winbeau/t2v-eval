"""
Tests for scripts/vbench_runner/scaling.py — output percent normalization.
"""

import pandas as pd
import pytest

from scripts.vbench_runner.scaling import (
    ALL_16_COLUMNS,
    apply_output_percent_scaling,
    compute_official_vbench_scores,
    compute_semantic_lite_vbench_scores,
    resolve_output_percent_columns,
)


class TestResolveOutputPercentColumns:
    def test_auto_mode_selects_only_zero_one_columns(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "dynamic_degree": [0.5, 0.6],
                "motion_smoothness": [0.8, 0.9],
                "imaging_quality": [68.0, 70.0],
            }
        )
        cols = resolve_output_percent_columns(
            df=df,
            vbench_config={"output_percent_scale": True, "output_percent_mode": "auto_01_only"},
            candidate_columns=["dynamic_degree", "motion_smoothness", "imaging_quality"],
        )
        assert cols == ["dynamic_degree", "motion_smoothness"]

    def test_explicit_mode_uses_configured_columns(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1"],
                "imaging_quality": [69.31],
            }
        )
        cols = resolve_output_percent_columns(
            df=df,
            vbench_config={
                "output_percent_mode": "explicit_list",
                "output_percent_columns": ["imaging_quality", "missing_col"],
            },
        )
        assert cols == ["imaging_quality"]


class TestApplyOutputPercentScaling:
    def test_scaling_and_temporal_recompute(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "dynamic_degree": [0.5756, 0.5719],
                "motion_smoothness": [0.9827, 0.9823],
                "imaging_quality": [69.31, 69.27],
                "vbench_temporal_score": [5.0, 5.0],
            }
        )

        scaled_cols, temporal_cols = apply_output_percent_scaling(
            df=df,
            vbench_config={"output_percent_scale": True, "output_percent_mode": "auto_01_only"},
            candidate_columns=["dynamic_degree", "motion_smoothness", "imaging_quality"],
        )

        assert scaled_cols == ["dynamic_degree", "motion_smoothness"]
        assert temporal_cols == ["dynamic_degree", "motion_smoothness", "imaging_quality"]
        assert df.loc[0, "dynamic_degree"] == pytest.approx(57.56, abs=1e-6)
        assert df.loc[0, "motion_smoothness"] == pytest.approx(98.27, abs=1e-6)
        assert df.loc[0, "imaging_quality"] == pytest.approx(69.31, abs=1e-6)
        assert df.loc[0, "vbench_temporal_score"] == pytest.approx((57.56 + 98.27 + 69.31) / 3, abs=1e-6)

    def test_disabled_scaling_keeps_values_unchanged(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1"],
                "dynamic_degree": [0.5],
                "vbench_temporal_score": [0.5],
            }
        )
        scaled_cols, temporal_cols = apply_output_percent_scaling(
            df=df,
            vbench_config={"output_percent_scale": False},
            candidate_columns=["dynamic_degree"],
        )
        assert scaled_cols == []
        assert temporal_cols == []
        assert df.loc[0, "dynamic_degree"] == pytest.approx(0.5, abs=1e-6)
        assert df.loc[0, "vbench_temporal_score"] == pytest.approx(0.5, abs=1e-6)

    def test_already_percent_value_not_rescaled(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "subject_consistency": [97.34, 96.96],
                "vbench_temporal_score": [97.34, 96.96],
            }
        )
        scaled_cols, _ = apply_output_percent_scaling(
            df=df,
            vbench_config={"output_percent_scale": True, "output_percent_mode": "auto_01_only"},
            candidate_columns=["subject_consistency"],
        )
        assert scaled_cols == []
        assert df.loc[0, "subject_consistency"] == pytest.approx(97.34, abs=1e-6)


class TestComputeOfficialVBenchScores:
    """Tests for compute_official_vbench_scores (official VBench aggregation)."""

    @staticmethod
    def _make_full_16d_df(**overrides):
        """Build a single-row DataFrame with all 16 VBench dimensions at given values."""
        defaults = {col: 0.5 for col in ALL_16_COLUMNS}
        defaults["video_id"] = "v1"
        defaults.update(overrides)
        return pd.DataFrame([defaults])

    def test_skips_when_missing_dimensions(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1"],
                "subject_consistency": [0.9],
                "background_consistency": [0.8],
            }
        )
        result = compute_official_vbench_scores(df)
        assert result == []
        assert "vbench_total_score" not in df.columns

    def test_adds_three_score_columns(self):
        df = self._make_full_16d_df()
        result = compute_official_vbench_scores(df)
        assert set(result) == {
            "vbench_quality_score",
            "vbench_semantic_score",
            "vbench_total_score",
        }
        assert "vbench_quality_score" in df.columns
        assert "vbench_semantic_score" in df.columns
        assert "vbench_total_score" in df.columns

    def test_total_is_weighted_average_of_quality_and_semantic(self):
        df = self._make_full_16d_df()
        compute_official_vbench_scores(df)
        q = df.loc[0, "vbench_quality_score"]
        s = df.loc[0, "vbench_semantic_score"]
        expected_total = (q * 4 + s * 1) / 5
        assert df.loc[0, "vbench_total_score"] == pytest.approx(expected_total, abs=1e-6)

    def test_all_max_scores_give_100(self):
        """When every dimension is at its NORMALIZE_DIC Max, scores should be 100."""
        from scripts.vbench_runner.scaling import _COL_TO_VBENCH_DIM, _NORMALIZE_DIC

        max_vals = {}
        for col, dim_name in _COL_TO_VBENCH_DIM.items():
            max_vals[col] = _NORMALIZE_DIC[dim_name]["Max"]
        df = pd.DataFrame([{"video_id": "v1", **max_vals}])
        compute_official_vbench_scores(df)
        assert df.loc[0, "vbench_quality_score"] == pytest.approx(100.0, abs=1e-6)
        assert df.loc[0, "vbench_semantic_score"] == pytest.approx(100.0, abs=1e-6)
        assert df.loc[0, "vbench_total_score"] == pytest.approx(100.0, abs=1e-6)

    def test_all_min_scores_give_0(self):
        """When every dimension is at its NORMALIZE_DIC Min, scores should be 0."""
        from scripts.vbench_runner.scaling import _COL_TO_VBENCH_DIM, _NORMALIZE_DIC

        min_vals = {}
        for col, dim_name in _COL_TO_VBENCH_DIM.items():
            min_vals[col] = _NORMALIZE_DIC[dim_name]["Min"]
        df = pd.DataFrame([{"video_id": "v1", **min_vals}])
        compute_official_vbench_scores(df)
        assert df.loc[0, "vbench_quality_score"] == pytest.approx(0.0, abs=1e-6)
        assert df.loc[0, "vbench_semantic_score"] == pytest.approx(0.0, abs=1e-6)
        assert df.loc[0, "vbench_total_score"] == pytest.approx(0.0, abs=1e-6)

    def test_dynamic_degree_half_weight(self):
        """dynamic_degree has weight 0.5 — verify it's handled correctly."""
        from scripts.vbench_runner.scaling import _COL_TO_VBENCH_DIM, _NORMALIZE_DIC

        # Set all quality dims to their Max (normalized=1) except dynamic_degree at Min (normalized=0).
        vals = {}
        for col, dim_name in _COL_TO_VBENCH_DIM.items():
            vals[col] = _NORMALIZE_DIC[dim_name]["Max"]
        vals["dynamic_degree"] = 0.0  # Min for dynamic_degree is 0.0, so normalized=0
        df = pd.DataFrame([{"video_id": "v1", **vals}])
        compute_official_vbench_scores(df)
        # Quality: 6 dims at weight 1.0 * 1.0 + 1 dim (dynamic_degree) at weight 0.5 * 0.0
        # = 6.0 / 6.5 * 100
        expected_quality = (6.0 / 6.5) * 100.0
        assert df.loc[0, "vbench_quality_score"] == pytest.approx(expected_quality, abs=1e-6)
        # Semantic: all at max, so still 100
        assert df.loc[0, "vbench_semantic_score"] == pytest.approx(100.0, abs=1e-6)


class TestComputeSemanticLiteVBenchScores:
    @staticmethod
    def _make_full_df_without_color(**overrides):
        defaults = {col: 0.5 for col in ALL_16_COLUMNS if col != "color"}
        defaults["video_id"] = "v1"
        defaults.update(overrides)
        return pd.DataFrame([defaults])

    def test_lite_scores_compute_when_color_is_missing(self):
        df = self._make_full_df_without_color()

        result = compute_semantic_lite_vbench_scores(df)

        assert set(result) == {
            "vbench_quality_score",
            "vbench_semantic_lite_score",
            "vbench_total_lite_score",
        }
        assert "vbench_semantic_lite_score" in df.columns
        assert "vbench_total_lite_score" in df.columns
        assert "vbench_semantic_score" not in df.columns
        assert "vbench_total_score" not in df.columns

    def test_lite_scores_skip_when_non_color_semantic_dimension_is_missing(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1"],
                "subject_consistency": [0.9],
                "background_consistency": [0.8],
                "temporal_flickering": [0.7],
                "motion_smoothness": [0.6],
                "dynamic_degree": [0.5],
                "aesthetic_quality": [0.4],
                "imaging_quality": [0.3],
                "object_class": [0.2],
                "multiple_objects": [0.1],
                "human_action": [0.9],
                "spatial_relationship": [0.8],
                "scene": [0.7],
                "appearance_style": [0.6],
                "temporal_style": [0.5],
                # overall_consistency missing
            }
        )

        result = compute_semantic_lite_vbench_scores(df)

        assert result == []
        assert "vbench_semantic_lite_score" not in df.columns
        assert "vbench_total_lite_score" not in df.columns

    def test_lite_scores_skip_when_quality_dimension_is_missing(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1"],
                "subject_consistency": [0.9],
                "background_consistency": [0.8],
                "temporal_flickering": [0.7],
                "motion_smoothness": [0.6],
                # dynamic_degree missing
                "aesthetic_quality": [0.4],
                "imaging_quality": [0.3],
                "object_class": [0.2],
                "multiple_objects": [0.1],
                "human_action": [0.9],
                "spatial_relationship": [0.8],
                "scene": [0.7],
                "appearance_style": [0.6],
                "temporal_style": [0.5],
                "overall_consistency": [0.4],
            }
        )

        result = compute_semantic_lite_vbench_scores(df)

        assert result == []

    def test_lite_total_uses_same_quality_semantic_weighting(self):
        df = self._make_full_df_without_color()

        compute_semantic_lite_vbench_scores(df)
        q = df.loc[0, "vbench_quality_score"]
        s_lite = df.loc[0, "vbench_semantic_lite_score"]
        expected_total = (q * 4 + s_lite * 1) / 5

        assert df.loc[0, "vbench_total_lite_score"] == pytest.approx(expected_total, abs=1e-6)

    def test_full_16_can_emit_official_and_lite_scores_together(self):
        df = pd.DataFrame([{**{col: 0.5 for col in ALL_16_COLUMNS}, "video_id": "v1"}])

        compute_official_vbench_scores(df)
        compute_semantic_lite_vbench_scores(df)

        assert "vbench_semantic_score" in df.columns
        assert "vbench_total_score" in df.columns
        assert "vbench_semantic_lite_score" in df.columns
        assert "vbench_total_lite_score" in df.columns
