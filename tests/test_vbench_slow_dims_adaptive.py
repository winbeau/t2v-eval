"""Tests for adaptive detection execution planning in slow_dims fused runner."""

from scripts.vbench_runner.slow_dims import _resolve_det_exec_plan


def test_resolve_det_exec_plan_probe_uses_both_paths():
    det_dims_batch, det_dims_single, probe_dims = _resolve_det_exec_plan(
        det_dims_present=["object_class", "spatial_relationship"],
        det_single_image_dims_set=set(),
        det_adaptive_state={
            "spatial_relationship": {"probed": 0, "triggered": False, "max_abs_diff": 0.0}
        },
        det_adaptive_probe_clips=8,
    )

    assert det_dims_batch == ["object_class", "spatial_relationship"]
    assert det_dims_single == ["spatial_relationship"]
    assert probe_dims == ["spatial_relationship"]


def test_resolve_det_exec_plan_triggered_dim_forces_single_only():
    det_dims_batch, det_dims_single, probe_dims = _resolve_det_exec_plan(
        det_dims_present=["spatial_relationship"],
        det_single_image_dims_set=set(),
        det_adaptive_state={
            "spatial_relationship": {"probed": 3, "triggered": True, "max_abs_diff": 0.06}
        },
        det_adaptive_probe_clips=8,
    )

    assert det_dims_batch == []
    assert det_dims_single == ["spatial_relationship"]
    assert probe_dims == []


def test_resolve_det_exec_plan_hard_single_overrides_adaptive():
    det_dims_batch, det_dims_single, probe_dims = _resolve_det_exec_plan(
        det_dims_present=["multiple_objects", "spatial_relationship"],
        det_single_image_dims_set={"spatial_relationship"},
        det_adaptive_state={
            "spatial_relationship": {"probed": 0, "triggered": False, "max_abs_diff": 0.0},
            "multiple_objects": {"probed": 1, "triggered": False, "max_abs_diff": 0.0},
        },
        det_adaptive_probe_clips=8,
    )

    assert det_dims_batch == ["multiple_objects"]
    assert det_dims_single == ["multiple_objects", "spatial_relationship"]
    assert probe_dims == ["multiple_objects"]
