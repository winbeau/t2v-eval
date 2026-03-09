from scripts.vbench_runner.core import (
    _analyze_video_record_diagnostics,
    _apply_color_coverage_policy,
)


class TestAnalyzeVideoRecordDiagnostics:
    def test_reports_missing_groups_and_prompt_fallback_count(self):
        records = [
            {"video_id": "v1", "group": "g1", "prompt_source": "fallback_video_id"},
            {"video_id": "v2", "group": "g1", "prompt_source": "group_id"},
        ]
        config = {"groups": [{"name": "g1"}, {"name": "g2"}]}

        diagnostics = _analyze_video_record_diagnostics(records, config)

        assert diagnostics["configured_groups"] == ["g1", "g2"]
        assert diagnostics["discovered_groups"] == ["g1"]
        assert diagnostics["missing_groups"] == ["g2"]
        assert diagnostics["fallback_count"] == 1


class TestApplyColorCoveragePolicy:
    def test_default_threshold_requires_full_color_coverage(self):
        coverage_rows = [("color", 8, 10), ("scene", 10, 10)]
        issues = _apply_color_coverage_policy(
            coverage_rows=coverage_rows,
            all_subtasks=["color", "scene"],
            expected_count=10,
            vbench_config={},
            record_diagnostics={"missing_groups": [], "fallback_count": 0},
        )
        assert ("color", 8, 10) in issues

    def test_lower_threshold_allows_partial_color_coverage(self):
        coverage_rows = [("color", 8, 10), ("scene", 10, 10)]
        issues = _apply_color_coverage_policy(
            coverage_rows=coverage_rows,
            all_subtasks=["color", "scene"],
            expected_count=10,
            vbench_config={"color_min_coverage_ratio": 0.7},
            record_diagnostics={"missing_groups": [], "fallback_count": 0},
        )
        assert issues == []

    def test_non_color_coverage_issues_remain_strict(self):
        coverage_rows = [("color", 10, 10), ("scene", 6, 10)]
        issues = _apply_color_coverage_policy(
            coverage_rows=coverage_rows,
            all_subtasks=["color", "scene"],
            expected_count=10,
            vbench_config={"color_min_coverage_ratio": 0.7},
            record_diagnostics={"missing_groups": [], "fallback_count": 0},
        )
        assert ("scene", 6, 10) in issues
