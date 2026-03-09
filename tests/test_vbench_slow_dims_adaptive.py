"""Tests for adaptive detection execution planning in slow_dims fused runner."""

import sys
from types import ModuleType

import numpy as np
import pytest

from scripts.vbench_runner import slow_dims
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


def test_run_fused_slow_dimensions_color_strict_raises(monkeypatch: pytest.MonkeyPatch):
    vbench_pkg = ModuleType("vbench")
    vbench_pkg.__path__ = []  # type: ignore[attr-defined]
    third_party_pkg = ModuleType("vbench.third_party")
    third_party_pkg.__path__ = []  # type: ignore[attr-defined]

    color_mod = ModuleType("vbench.color")
    color_mod.check_generate = lambda color_key, object_key, predictions: (0, 0)

    multi_mod = ModuleType("vbench.multiple_objects")
    object_mod = ModuleType("vbench.object_class")
    spatial_mod = ModuleType("vbench.spatial_relationship")

    grit_model_mod = ModuleType("vbench.third_party.grit_model")

    class _FakeDenseCaptioning:
        def __init__(self, device):
            self.device = device

        def initialize_model(self, model_weight=None):
            return None

        def run_caption_tensor_batch(self, image_arrays):
            return [[] for _ in range(len(image_arrays))]

    grit_model_mod.DenseCaptioning = _FakeDenseCaptioning

    utils_mod = ModuleType("vbench.utils")
    utils_mod.init_submodules = lambda active_dims, local=True, read_frame=False: {
        "color": {"model_weight": "fake_weight"}
    }
    utils_mod.load_video = lambda path, num_frames=16, return_tensor=False: np.zeros(
        (16, 4, 4, 3), dtype=np.uint8
    )

    monkeypatch.setitem(sys.modules, "vbench", vbench_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party", third_party_pkg)
    monkeypatch.setitem(sys.modules, "vbench.color", color_mod)
    monkeypatch.setitem(sys.modules, "vbench.multiple_objects", multi_mod)
    monkeypatch.setitem(sys.modules, "vbench.object_class", object_mod)
    monkeypatch.setitem(sys.modules, "vbench.spatial_relationship", spatial_mod)
    monkeypatch.setitem(sys.modules, "vbench.third_party.grit_model", grit_model_mod)
    monkeypatch.setitem(sys.modules, "vbench.utils", utils_mod)

    monkeypatch.setattr(
        slow_dims,
        "_collect_clip_meta",
        lambda full_info_path, dim: {
            "/tmp/dataset/g1/clip_scene/clip.mp4": {
                "prompt": "a red car driving",
                "auxiliary_info": {
                    "color": "red",
                    "object": "car",
                    "object_candidates": ["car"],
                    "object_key": "car driving",
                },
            }
        },
    )

    with pytest.raises(
        RuntimeError,
        match="color strict integrity violation.*object='car'.*object_candidates=\\['car'\\]",
    ):
        slow_dims.run_fused_slow_dimensions(
            full_info_path=None,
            subtasks=["color"],
            valid_video_ids={"clip"},
            rank=0,
            world_size=1,
            device="cpu",
            strict_integrity=True,
            color_fill_zero_on_no_object=False,
            decode_workers=1,
            decode_prefetch=1,
            stage_profile=False,
        )
