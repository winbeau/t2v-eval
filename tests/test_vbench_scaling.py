"""
Tests for scripts/vbench_runner/scaling.py — output percent normalization.
"""

import pandas as pd
import pytest

from scripts.vbench_runner.scaling import (
    apply_output_percent_scaling,
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
