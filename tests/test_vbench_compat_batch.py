"""Tests for GrIT batch compatibility patch in scripts/vbench_runner/compat.py."""

import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from scripts.vbench_runner import compat


class _FakeAug:
    """Mimic detectron2 augmenter API used by patched code."""

    def get_transform(self, image):
        return self

    def apply_image(self, image):
        return image


class _FakePredictor:
    """Deterministic predictor for comparing sequential vs batched paths."""

    def __init__(self):
        self.aug = _FakeAug()

    @staticmethod
    def _score_from_hwc(image: np.ndarray) -> float:
        return float(np.asarray(image, dtype=np.float32).mean())

    def __call__(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        return {
            "score": self._score_from_hwc(image),
            "height": int(h),
            "width": int(w),
        }

    def model(self, batch_inputs: list[dict]) -> list[dict]:
        outputs = []
        for item in batch_inputs:
            tensor = item["image"]  # CHW float32 tensor
            score = float(tensor.float().mean().item())
            outputs.append(
                {
                    "score": score,
                    "height": int(item["height"]),
                    "width": int(item["width"]),
                }
            )
        return outputs


class _FakeVisualizationDemo:
    """Drop-in replacement for vbench.third_party...VisualizationDemo."""

    def __init__(self):
        self.predictor = _FakePredictor()

    def run_on_batch(self, image_arrays):
        # Original implementation placeholder; patched function should override this.
        return ["ORIGINAL_SENTINEL", len(image_arrays)]


class _AssertOnBatchPredictor(_FakePredictor):
    """Simulate GRiT ROIHeads that reject multi-image batch proposals."""

    def model(self, batch_inputs: list[dict]) -> list[dict]:
        if len(batch_inputs) > 1:
            raise AssertionError("len(boxes) == 1")
        return super().model(batch_inputs)


class _AssertVisualizationDemo:
    """VisualizationDemo variant whose batch path triggers model-side assertion."""

    def __init__(self):
        self.predictor = _AssertOnBatchPredictor()

    def run_on_batch(self, image_arrays):
        return ["ORIGINAL_SENTINEL", len(image_arrays)]


@pytest.fixture
def fake_grit_predictor_module(monkeypatch: pytest.MonkeyPatch):
    """Inject fake vbench grit predictor module for import in compat patch."""

    vbench_pkg = ModuleType("vbench")
    vbench_pkg.__path__ = []  # type: ignore[attr-defined]

    third_party_pkg = ModuleType("vbench.third_party")
    third_party_pkg.__path__ = []  # type: ignore[attr-defined]

    grit_src_pkg = ModuleType("vbench.third_party.grit_src")
    grit_src_pkg.__path__ = []  # type: ignore[attr-defined]

    grit_pkg = ModuleType("vbench.third_party.grit_src.grit")
    grit_pkg.__path__ = []  # type: ignore[attr-defined]

    predictor_mod = ModuleType("vbench.third_party.grit_src.grit.predictor")
    predictor_mod.VisualizationDemo = _FakeVisualizationDemo

    monkeypatch.setitem(sys.modules, "vbench", vbench_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party", third_party_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party.grit_src", grit_src_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party.grit_src.grit", grit_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party.grit_src.grit.predictor", predictor_mod)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Clean any previous patch flags from the fake class.
    for attr in [
        "_patched_batch_compat",
        "_orig_run_on_batch",
        "_grit_batch_parallel_enable",
        "_grit_batch_size",
    ]:
        if hasattr(_FakeVisualizationDemo, attr):
            delattr(_FakeVisualizationDemo, attr)

    return _FakeVisualizationDemo


@pytest.fixture
def fake_grit_predictor_module_with_batch_assert(monkeypatch: pytest.MonkeyPatch):
    """Inject fake module whose predictor fails on multi-image model(batch_inputs)."""

    vbench_pkg = ModuleType("vbench")
    vbench_pkg.__path__ = []  # type: ignore[attr-defined]

    third_party_pkg = ModuleType("vbench.third_party")
    third_party_pkg.__path__ = []  # type: ignore[attr-defined]

    grit_src_pkg = ModuleType("vbench.third_party.grit_src")
    grit_src_pkg.__path__ = []  # type: ignore[attr-defined]

    grit_pkg = ModuleType("vbench.third_party.grit_src.grit")
    grit_pkg.__path__ = []  # type: ignore[attr-defined]

    predictor_mod = ModuleType("vbench.third_party.grit_src.grit.predictor")
    predictor_mod.VisualizationDemo = _AssertVisualizationDemo

    monkeypatch.setitem(sys.modules, "vbench", vbench_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party", third_party_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party.grit_src", grit_src_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party.grit_src.grit", grit_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party.grit_src.grit.predictor", predictor_mod)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    for attr in [
        "_patched_batch_compat",
        "_orig_run_on_batch",
        "_grit_batch_parallel_enable",
        "_grit_batch_size",
    ]:
        if hasattr(_AssertVisualizationDemo, attr):
            delattr(_AssertVisualizationDemo, attr)

    return _AssertVisualizationDemo


@pytest.fixture
def sample_images() -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.random((5, 8, 8, 3), dtype=np.float32)


def _assert_predictions_close(actual: list[dict], expected_scores: list[float]) -> None:
    assert len(actual) == len(expected_scores)
    for idx, pred in enumerate(actual):
        assert pred.keys() == {"score", "height", "width"}
        np.testing.assert_allclose(pred["score"], expected_scores[idx], rtol=1e-5, atol=1e-6)


def test_run_on_batch_consistent_between_safe_and_parallel(
    fake_grit_predictor_module,
    sample_images,
):
    expected_scores = [float(img.mean()) for img in sample_images]

    compat.patch_grit_batch_inference_compat(enable_batch_parallel=False, batch_size=1)
    safe_demo = fake_grit_predictor_module()
    safe_outputs = safe_demo.run_on_batch(sample_images)

    compat.patch_grit_batch_inference_compat(enable_batch_parallel=True, batch_size=2)
    parallel_demo = fake_grit_predictor_module()
    parallel_outputs = parallel_demo.run_on_batch(sample_images)

    _assert_predictions_close(safe_outputs, expected_scores)
    _assert_predictions_close(parallel_outputs, expected_scores)

    for safe_pred, par_pred in zip(safe_outputs, parallel_outputs, strict=False):
        np.testing.assert_allclose(safe_pred["score"], par_pred["score"], rtol=1e-5, atol=1e-6)
        assert safe_pred["height"] == par_pred["height"]
        assert safe_pred["width"] == par_pred["width"]


def test_parallel_chunking_non_divisible_batch_keeps_full_order(
    fake_grit_predictor_module,
    sample_images,
):
    expected_scores = [float(img.mean()) for img in sample_images]

    compat.patch_grit_batch_inference_compat(enable_batch_parallel=True, batch_size=3)
    demo = fake_grit_predictor_module()
    outputs = demo.run_on_batch(sample_images)

    _assert_predictions_close(outputs, expected_scores)


def test_parallel_assertion_bubbles_up_in_strict_mode(
    fake_grit_predictor_module_with_batch_assert,
    sample_images,
):
    compat.patch_grit_batch_inference_compat(enable_batch_parallel=True, batch_size=4)
    demo = fake_grit_predictor_module_with_batch_assert()
    with pytest.raises(AssertionError, match="len\\(boxes\\) == 1"):
        demo.run_on_batch(sample_images)


def test_safe_mode_asserts_when_batch_size_violates_invariant(
    fake_grit_predictor_module,
    sample_images,
):
    compat.patch_grit_batch_inference_compat(enable_batch_parallel=False, batch_size=1)
    # Simulate corrupted runtime setting: safe mode but batch size != 1
    fake_grit_predictor_module._grit_batch_size = 2
    fake_grit_predictor_module._grit_batch_parallel_enable = False

    demo = fake_grit_predictor_module()
    with pytest.raises(AssertionError, match="grit_batch_size == 1"):
        demo.run_on_batch(sample_images)


def test_patch_is_idempotent_and_preserves_original_reference(fake_grit_predictor_module):
    original_impl = fake_grit_predictor_module.run_on_batch

    compat.patch_grit_batch_inference_compat(enable_batch_parallel=False, batch_size=1)
    first_patched_impl = fake_grit_predictor_module.run_on_batch

    compat.patch_grit_batch_inference_compat(enable_batch_parallel=True, batch_size=4)
    second_patched_impl = fake_grit_predictor_module.run_on_batch

    assert first_patched_impl is second_patched_impl
    assert fake_grit_predictor_module._orig_run_on_batch is original_impl


def test_apply_vbench_compat_patches_forwards_batch_switch(monkeypatch: pytest.MonkeyPatch):
    forwarded: dict[str, int | bool] = {}

    monkeypatch.setattr(compat, "patch_transformers_compat", lambda: None)
    monkeypatch.setattr(compat, "patch_pretrained_model_tied_weights", lambda: None)
    monkeypatch.setattr(compat, "patch_config_pruned_heads", lambda: None)
    monkeypatch.setattr(compat, "patch_tokenizer_special_tokens_ids", lambda: None)
    monkeypatch.setattr(compat, "patch_clip_tokenize_truncate", lambda: None)
    monkeypatch.setattr(compat, "patch_grit_device_compat", lambda: None)
    monkeypatch.setattr(compat, "patch_generation_mixin", lambda: None)
    monkeypatch.setattr(compat, "patch_grit_fast_inference", lambda: None)
    monkeypatch.setattr(compat, "patch_color_object_matching", lambda: None)
    monkeypatch.setattr(compat, "patch_human_action_prompt_matching", lambda: None)

    def _capture(*, enable_batch_parallel: bool, batch_size: int):
        forwarded["enable_batch_parallel"] = enable_batch_parallel
        forwarded["batch_size"] = batch_size

    monkeypatch.setattr(compat, "patch_grit_batch_inference_compat", _capture)

    compat.apply_vbench_compat_patches(grit_batch_parallel_enable=True, grit_batch_size=8)

    assert forwarded == {"enable_batch_parallel": True, "batch_size": 8}


@pytest.fixture
def fake_grit_image_dense_module(monkeypatch: pytest.MonkeyPatch):
    """Inject fake image_dense_captions module for device-binding patch tests."""

    vbench_pkg = ModuleType("vbench")
    vbench_pkg.__path__ = []  # type: ignore[attr-defined]

    third_party_pkg = ModuleType("vbench.third_party")
    third_party_pkg.__path__ = []  # type: ignore[attr-defined]

    grit_src_pkg = ModuleType("vbench.third_party.grit_src")
    grit_src_pkg.__path__ = []  # type: ignore[attr-defined]

    captured: dict[str, object] = {}

    def _fake_get_parser(device, model_weight="dummy_weight.pth"):
        captured["device"] = device
        return {
            "cpu": getattr(device, "type", "") == "cpu",
            "config_file": "dummy_cfg.yaml",
            "confidence_threshold": 0.5,
            "test_task": "DenseCap",
            "opts": ["MODEL.WEIGHTS", model_weight],
        }

    idc_mod = ModuleType("vbench.third_party.grit_src.image_dense_captions")
    idc_mod.get_parser = _fake_get_parser
    grit_src_pkg.image_dense_captions = idc_mod

    monkeypatch.setitem(sys.modules, "vbench", vbench_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party", third_party_pkg)
    monkeypatch.setitem(sys.modules, "vbench.third_party.grit_src", grit_src_pkg)
    monkeypatch.setitem(
        sys.modules,
        "vbench.third_party.grit_src.image_dense_captions",
        idc_mod,
    )

    return idc_mod, captured


def test_grit_device_patch_expands_cuda_to_explicit_index(
    fake_grit_image_dense_module,
    monkeypatch: pytest.MonkeyPatch,
):
    idc_mod, captured = fake_grit_image_dense_module
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)

    compat.patch_grit_device_compat()
    parser_args = idc_mod.get_parser("cuda")

    assert str(captured["device"]) == "cuda:2"
    assert parser_args["opts"][-2:] == ["MODEL.DEVICE", "cuda:2"]


def test_grit_device_patch_keeps_explicit_cuda_index(
    fake_grit_image_dense_module,
    monkeypatch: pytest.MonkeyPatch,
):
    idc_mod, captured = fake_grit_image_dense_module
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    compat.patch_grit_device_compat()
    parser_args = idc_mod.get_parser("cuda:3")

    assert str(captured["device"]) == "cuda:3"
    assert parser_args["opts"][-2:] == ["MODEL.DEVICE", "cuda:3"]
