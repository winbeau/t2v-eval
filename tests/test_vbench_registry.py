"""
Tests for scripts/vbench_runner/dimensions/registry.py — dimension registry.

Covers:
  - LONG_DIMENSIONS_16 completeness
  - LONG_DIMENSIONS_6_RECOMMENDED subset
  - normalize_subtasks()
  - default_long_subtasks()
  - supported_long_subtasks()
  - DimensionSpec data integrity
  - CLIP_REQUIRED / PYIQA_REQUIRED sets
"""

import pytest

from scripts.vbench_runner.dimensions.base import DimensionSpec
from scripts.vbench_runner.dimensions.registry import (
    CLIP_REQUIRED_DIMENSIONS,
    LONG_DIMENSION_KEYS,
    LONG_DIMENSION_SET,
    LONG_DIMENSIONS,
    LONG_DIMENSIONS_6_RECOMMENDED,
    LONG_DIMENSIONS_16,
    PYIQA_REQUIRED_DIMENSIONS,
    default_long_subtasks,
    normalize_subtasks,
    supported_long_subtasks,
)


# ---------------------------------------------------------------------------
# Dimension list integrity
# ---------------------------------------------------------------------------
class TestDimensionLists:
    def test_16_dimensions_count(self):
        assert len(LONG_DIMENSIONS_16) == 16

    def test_6_recommended_count(self):
        assert len(LONG_DIMENSIONS_6_RECOMMENDED) == 6

    def test_recommended_is_subset_of_16(self):
        keys_16 = {s.key for s in LONG_DIMENSIONS_16}
        keys_6 = {s.key for s in LONG_DIMENSIONS_6_RECOMMENDED}
        assert keys_6.issubset(keys_16)

    def test_all_specs_are_dimension_spec(self):
        for spec in LONG_DIMENSIONS_16:
            assert isinstance(spec, DimensionSpec)

    def test_unique_keys(self):
        keys = [s.key for s in LONG_DIMENSIONS_16]
        assert len(keys) == len(set(keys)), "Duplicate dimension keys found"

    def test_known_dimensions_present(self):
        expected = {
            "subject_consistency",
            "background_consistency",
            "temporal_flickering",
            "motion_smoothness",
            "dynamic_degree",
            "imaging_quality",
            "aesthetic_quality",
            "color",
            "object_class",
            "multiple_objects",
            "spatial_relationship",
            "scene",
            "human_action",
            "overall_consistency",
            "temporal_style",
            "appearance_style",
        }
        assert LONG_DIMENSION_SET == expected

    def test_long_dimensions_alias(self):
        """LONG_DIMENSIONS should be the same as LONG_DIMENSIONS_16."""
        assert LONG_DIMENSIONS is LONG_DIMENSIONS_16

    def test_dimension_keys_match_list(self):
        assert LONG_DIMENSION_KEYS == [s.key for s in LONG_DIMENSIONS]

    def test_dimension_set_matches_keys(self):
        assert LONG_DIMENSION_SET == set(LONG_DIMENSION_KEYS)


# ---------------------------------------------------------------------------
# DimensionSpec properties
# ---------------------------------------------------------------------------
class TestDimensionSpec:
    def test_frozen(self):
        spec = DimensionSpec(key="test", description="Test dimension")
        with pytest.raises(AttributeError):
            spec.key = "modified"

    def test_defaults(self):
        spec = DimensionSpec(key="test", description="Test")
        assert spec.requires_clip is False
        assert spec.requires_pyiqa is False
        assert spec.long_mode_only is True

    def test_custom_flags(self):
        spec = DimensionSpec(
            key="test",
            description="Test",
            requires_clip=True,
            requires_pyiqa=True,
            long_mode_only=False,
        )
        assert spec.requires_clip is True
        assert spec.requires_pyiqa is True
        assert spec.long_mode_only is False


# ---------------------------------------------------------------------------
# CLIP / PYIQA requirement sets
# ---------------------------------------------------------------------------
class TestRequirementSets:
    def test_clip_required_is_subset(self):
        assert CLIP_REQUIRED_DIMENSIONS.issubset(LONG_DIMENSION_SET)

    def test_pyiqa_required_is_subset(self):
        assert PYIQA_REQUIRED_DIMENSIONS.issubset(LONG_DIMENSION_SET)

    def test_imaging_quality_requires_pyiqa(self):
        """imaging_quality should require pyiqa."""
        assert "imaging_quality" in PYIQA_REQUIRED_DIMENSIONS

    def test_background_consistency_requires_clip(self):
        """background_consistency requires CLIP."""
        assert "background_consistency" in CLIP_REQUIRED_DIMENSIONS


# ---------------------------------------------------------------------------
# normalize_subtasks
# ---------------------------------------------------------------------------
class TestNormalizeSubtasks:
    def test_deduplication(self):
        result = normalize_subtasks(["a", "b", "a", "c", "b"])
        assert result == ["a", "b", "c"]

    def test_whitespace_stripping(self):
        result = normalize_subtasks(["  a  ", " b", "c "])
        assert result == ["a", "b", "c"]

    def test_empty_strings_filtered(self):
        result = normalize_subtasks(["a", "", "  ", "b"])
        assert result == ["a", "b"]

    def test_empty_list(self):
        assert normalize_subtasks([]) == []

    def test_preserves_order(self):
        result = normalize_subtasks(["z", "a", "m"])
        assert result == ["z", "a", "m"]

    def test_non_string_conversion(self):
        """Non-string items should be converted to str."""
        result = normalize_subtasks([1, 2, 3])
        assert result == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# default_long_subtasks
# ---------------------------------------------------------------------------
class TestDefaultLongSubtasks:
    @pytest.mark.parametrize(
        "profile",
        ["long_16", "16", "16d", "full", "full_16"],
    )
    def test_full_16_profiles(self, profile):
        result = default_long_subtasks(profile)
        assert len(result) == 16

    @pytest.mark.parametrize(
        "profile",
        ["long_6", "6", "default", "recommended", ""],
    )
    def test_recommended_6_profiles(self, profile):
        result = default_long_subtasks(profile)
        assert len(result) == 6

    def test_returns_strings(self):
        result = default_long_subtasks("long_16")
        assert all(isinstance(s, str) for s in result)

    def test_case_insensitive(self):
        result_upper = default_long_subtasks("LONG_16")
        result_lower = default_long_subtasks("long_16")
        assert result_upper == result_lower


# ---------------------------------------------------------------------------
# supported_long_subtasks
# ---------------------------------------------------------------------------
class TestSupportedLongSubtasks:
    def test_returns_all_16(self):
        result = supported_long_subtasks()
        assert len(result) == 16

    def test_returns_strings(self):
        result = supported_long_subtasks()
        assert all(isinstance(s, str) for s in result)

    def test_matches_dimension_keys(self):
        assert supported_long_subtasks() == LONG_DIMENSION_KEYS
