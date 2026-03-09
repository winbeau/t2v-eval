"""Contract tests for the 16 registered VBench-Long dimensions."""

import pytest

from scripts.vbench_runner.dimensions.base import DimensionSpec
from scripts.vbench_runner.dimensions.registry import (
    CLIP_REQUIRED_DIMENSIONS,
    LONG_DIMENSIONS_16,
    LONG_DIMENSION_KEYS,
    LONG_DIMENSION_SET,
    PYIQA_REQUIRED_DIMENSIONS,
    default_long_subtasks,
    normalize_subtasks,
    supported_long_subtasks,
)


EXPECTED_LONG_16_ORDER = [
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


def test_long_dimension_registry_exact_order_and_uniqueness():
    specs = LONG_DIMENSIONS_16
    assert all(isinstance(spec, DimensionSpec) for spec in specs)
    assert [spec.key for spec in specs] == EXPECTED_LONG_16_ORDER
    assert LONG_DIMENSION_KEYS == EXPECTED_LONG_16_ORDER
    assert LONG_DIMENSION_SET == set(EXPECTED_LONG_16_ORDER)
    assert len({spec.key for spec in specs}) == len(specs) == 16


@pytest.mark.parametrize("spec", LONG_DIMENSIONS_16, ids=[spec.key for spec in LONG_DIMENSIONS_16])
def test_each_dimension_spec_has_nonempty_contract(spec: DimensionSpec):
    assert spec.key
    assert spec.description
    assert spec.long_mode_only is True


def test_clip_required_dimensions_match_registered_specs():
    expected = {
        spec.key
        for spec in LONG_DIMENSIONS_16
        if spec.requires_clip
    }
    assert expected == CLIP_REQUIRED_DIMENSIONS
    assert expected == {
        "background_consistency",
        "temporal_style",
        "appearance_style",
        "human_action",
        "overall_consistency",
        "aesthetic_quality",
    }


def test_pyiqa_required_dimensions_match_registered_specs():
    expected = {
        spec.key
        for spec in LONG_DIMENSIONS_16
        if spec.requires_pyiqa
    }
    assert expected == PYIQA_REQUIRED_DIMENSIONS
    assert expected == {"imaging_quality"}


def test_default_long_subtasks_profiles_are_stable():
    assert default_long_subtasks("long_16") == EXPECTED_LONG_16_ORDER
    assert default_long_subtasks("16") == EXPECTED_LONG_16_ORDER
    assert default_long_subtasks("full_16") == EXPECTED_LONG_16_ORDER
    assert default_long_subtasks("long_6") == [
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "imaging_quality",
        "aesthetic_quality",
    ]


def test_supported_long_subtasks_matches_full_registry():
    assert supported_long_subtasks() == EXPECTED_LONG_16_ORDER


def test_normalize_subtasks_preserves_order_and_deduplicates():
    raw = [" color ", "scene", "color", "", "scene", "object_class"]
    assert normalize_subtasks(raw) == ["color", "scene", "object_class"]
