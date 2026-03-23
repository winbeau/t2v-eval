from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.recompute_vbench_model_scores import (
    _normalize_raw_dimension_scales,
    build_comparison_table,
    build_markdown_report,
    detect_score_mode,
    main,
)
from scripts.vbench_runner.scaling import (
    ALL_16_COLUMNS,
    SEMANTIC_LITE_COLUMNS,
    compute_official_vbench_scores,
    compute_semantic_lite_vbench_scores,
)


def _write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


class TestDetectScoreMode:
    def test_detects_lite12(self):
        df = pd.DataFrame([{**{col: 0.5 for col in SEMANTIC_LITE_COLUMNS}, "video_id": "v1", "group": "g1", "vbench_quality_score": 1.0, "vbench_semantic_lite_score": 1.0, "vbench_total_lite_score": 1.0}])
        mode, raw_cols = detect_score_mode(df)
        assert mode == "lite12"
        assert set(raw_cols) == set(SEMANTIC_LITE_COLUMNS)

    def test_detects_official16(self):
        df = pd.DataFrame([{**{col: 0.5 for col in ALL_16_COLUMNS}, "video_id": "v1", "group": "g1", "vbench_quality_score": 1.0, "vbench_semantic_lite_score": 1.0, "vbench_total_lite_score": 1.0}])
        mode, raw_cols = detect_score_mode(df)
        assert mode == "official16"
        assert set(raw_cols) == set(ALL_16_COLUMNS)


class TestScaleNormalization:
    def test_unscales_percent_raw_dimensions(self):
        df = pd.DataFrame([{"group": "g1", **{col: 50.0 for col in SEMANTIC_LITE_COLUMNS}}])
        out = _normalize_raw_dimension_scales(df, sorted(SEMANTIC_LITE_COLUMNS))
        assert out.loc[0, "dynamic_degree"] == pytest.approx(0.5)
        assert out.loc[0, "subject_consistency"] == pytest.approx(0.5)


class TestBuildComparisonTable:
    def test_lite12_outputs_old_new_and_deltas(self, tmp_path: Path):
        row_a = {col: 0.5 for col in SEMANTIC_LITE_COLUMNS}
        row_b = {col: 0.6 for col in SEMANTIC_LITE_COLUMNS}
        row_c = {col: 0.7 for col in SEMANTIC_LITE_COLUMNS}
        df = pd.DataFrame(
            [
                {"video_id": "a1", "group": "g1", **row_a, "vbench_quality_score": 81.0, "vbench_semantic_lite_score": 52.0, "vbench_total_lite_score": 75.2},
                {"video_id": "a2", "group": "g1", **row_b, "vbench_quality_score": 83.0, "vbench_semantic_lite_score": 54.0, "vbench_total_lite_score": 77.2},
                {"video_id": "b1", "group": "g2", **row_c, "vbench_quality_score": 85.0, "vbench_semantic_lite_score": 56.0, "vbench_total_lite_score": 79.2},
            ]
        )
        csv_path = _write_csv(tmp_path / "vbench_sample.csv", df)

        out = build_comparison_table(csv_path)

        assert list(out["group"]) == ["g1", "g2"]
        assert out.loc[out["group"] == "g1", "vbench_quality_score_old"].iloc[0] == pytest.approx(82.0)
        assert "vbench_quality_score_new" in out.columns
        assert "vbench_semantic_lite_score_new" in out.columns
        assert "vbench_total_lite_score_new" in out.columns
        assert "vbench_quality_score_delta" in out.columns

        grouped = (
            df.groupby("group", as_index=False)[sorted(SEMANTIC_LITE_COLUMNS)]
            .mean(numeric_only=True)
            .rename(columns={"group": "video_id"})
        )
        grouped.insert(0, "group", ["g1", "g2"])
        tmp = grouped.drop(columns=["group"]).copy()
        compute_semantic_lite_vbench_scores(tmp)
        expected_g1 = tmp.loc[0, "vbench_quality_score"]
        actual_g1 = out.loc[out["group"] == "g1", "vbench_quality_score_new"].iloc[0]
        assert actual_g1 == pytest.approx(expected_g1)

    def test_percent_scaled_inputs_are_unscaled_before_recompute(self, tmp_path: Path):
        row_a = {col: 50.0 for col in SEMANTIC_LITE_COLUMNS}
        row_b = {col: 60.0 for col in SEMANTIC_LITE_COLUMNS}
        df = pd.DataFrame(
            [
                {"video_id": "a1", "group": "g1", **row_a, "vbench_quality_score": 81.0, "vbench_semantic_lite_score": 52.0, "vbench_total_lite_score": 75.2},
                {"video_id": "a2", "group": "g1", **row_b, "vbench_quality_score": 83.0, "vbench_semantic_lite_score": 54.0, "vbench_total_lite_score": 77.2},
            ]
        )
        csv_path = _write_csv(tmp_path / "vbench_percent.csv", df)

        out = build_comparison_table(csv_path)

        grouped = pd.DataFrame([{col: 0.55 for col in sorted(SEMANTIC_LITE_COLUMNS)}])
        grouped.insert(0, "video_id", "g1")
        compute_semantic_lite_vbench_scores(grouped)
        expected = grouped.loc[0, "vbench_quality_score"]
        actual = out.loc[out["group"] == "g1", "vbench_quality_score_new"].iloc[0]
        assert actual == pytest.approx(expected)
        assert actual < 100.0

    def test_official16_emits_official_new_columns(self, tmp_path: Path):
        row_a = {col: 0.5 for col in ALL_16_COLUMNS}
        row_b = {col: 0.8 for col in ALL_16_COLUMNS}
        df = pd.DataFrame(
            [
                {"video_id": "a1", "group": "g1", **row_a, "vbench_quality_score": 80.0, "vbench_semantic_lite_score": 51.0, "vbench_total_lite_score": 74.2},
                {"video_id": "b1", "group": "g2", **row_b, "vbench_quality_score": 88.0, "vbench_semantic_lite_score": 60.0, "vbench_total_lite_score": 82.4},
            ]
        )
        csv_path = _write_csv(tmp_path / "vbench_full.csv", df)

        out = build_comparison_table(csv_path)

        assert "vbench_semantic_score_new" in out.columns
        assert "vbench_total_score_new" in out.columns

        grouped = (
            df.groupby("group", as_index=False)[sorted(ALL_16_COLUMNS)]
            .mean(numeric_only=True)
            .rename(columns={"group": "video_id"})
        )
        grouped.insert(0, "group", ["g1", "g2"])
        tmp = grouped.drop(columns=["group"]).copy()
        compute_official_vbench_scores(tmp)
        expected_g2 = tmp.loc[1, "vbench_total_score"]
        actual_g2 = out.loc[out["group"] == "g2", "vbench_total_score_new"].iloc[0]
        assert actual_g2 == pytest.approx(expected_g2)

    def test_missing_old_columns_errors(self, tmp_path: Path):
        df = pd.DataFrame(
            [{"video_id": "v1", "group": "g1", **{col: 0.5 for col in SEMANTIC_LITE_COLUMNS}}]
        )
        csv_path = _write_csv(tmp_path / "missing_old.csv", df)

        with pytest.raises(ValueError, match="Missing required old-score columns"):
            build_comparison_table(csv_path)


class TestReportAndCli:
    def test_report_mentions_max_delta(self, tmp_path: Path):
        df = pd.DataFrame(
            [
                {
                    "input_name": "vbench_x",
                    "score_mode": "lite12",
                    "group": "g1",
                    "n_videos": 2,
                    "vbench_quality_score_old": 80.0,
                    "vbench_quality_score_new": 82.5,
                    "vbench_quality_score_delta": 2.5,
                    "vbench_quality_score_abs_delta": 2.5,
                    "vbench_semantic_lite_score_old": 50.0,
                    "vbench_semantic_lite_score_new": 49.0,
                    "vbench_semantic_lite_score_delta": -1.0,
                    "vbench_semantic_lite_score_abs_delta": 1.0,
                    "vbench_total_lite_score_old": 74.0,
                    "vbench_total_lite_score_new": 75.8,
                    "vbench_total_lite_score_delta": 1.8,
                    "vbench_total_lite_score_abs_delta": 1.8,
                }
            ]
        )
        report = build_markdown_report(df)
        assert "## vbench_x" in report
        assert "max |Δ| `vbench_quality_score`" in report
        assert "group=`g1`" in report

    def test_cli_writes_outputs(self, tmp_path: Path):
        df = pd.DataFrame(
            [
                {"video_id": "a1", "group": "g1", **{col: 0.5 for col in SEMANTIC_LITE_COLUMNS}, "vbench_quality_score": 81.0, "vbench_semantic_lite_score": 52.0, "vbench_total_lite_score": 75.2},
                {"video_id": "a2", "group": "g1", **{col: 0.6 for col in SEMANTIC_LITE_COLUMNS}, "vbench_quality_score": 83.0, "vbench_semantic_lite_score": 54.0, "vbench_total_lite_score": 77.2},
            ]
        )
        csv_path = _write_csv(tmp_path / "vbench_cli.csv", df)
        out_dir = tmp_path / "artifacts"

        exit_code = main(["--input", str(csv_path), "--output-dir", str(out_dir)])

        assert exit_code == 0
        assert (out_dir / "group_score_recompute_comparison.csv").exists()
        assert (out_dir / "group_score_recompute_report.md").exists()
