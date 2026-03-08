"""GPU integration parity test for GrIT batch=1 vs batch=4 paths.

This test is intentionally opt-in because it requires:
- CUDA GPU
- detectron2 + VBench deps
- GRiT weights in local cache
- a real video clip
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.vbench_runner import compat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "hf"
    / "AdaHead"
    / "Exp-K_StaOscCompression"
    / "k3_sta_posneg_osc_phase6_tail4"
    / "video_001.mp4"
)
DEFAULT_WEIGHT = Path.home() / ".cache" / "vbench" / "grit_model" / "grit_b_densecap_objectdet.pth"

pytestmark = pytest.mark.gpu_integration


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _ensure_enabled_or_skip() -> None:
    if not _env_true("RUN_VBENCH_GPU_INTEGRATION"):
        pytest.skip(
            "Set RUN_VBENCH_GPU_INTEGRATION=1 to run heavy GPU parity integration tests."
        )


def _maybe_add_vbench_submodule_path() -> None:
    vbench_repo = PROJECT_ROOT / "third_party" / "VBench"
    if str(vbench_repo) not in sys.path:
        sys.path.insert(0, str(vbench_repo))


def _resolve_video_path() -> Path:
    override = os.getenv("VBENCH_PARITY_VIDEO", "").strip()
    path = Path(override) if override else DEFAULT_VIDEO
    if not path.exists():
        pytest.skip(f"Parity video not found: {path}")
    return path


def _resolve_weight_path() -> Path:
    override = os.getenv("VBENCH_GRIT_MODEL_WEIGHT", "").strip()
    path = Path(override) if override else DEFAULT_WEIGHT
    if not path.exists():
        pytest.skip(f"GRiT model weight not found: {path}")
    return path


def _serialize_prediction(pred: dict) -> dict:
    inst = pred["instances"].to("cpu")

    boxes = (
        inst.pred_boxes.tensor.detach().cpu().numpy().astype(np.float32)
        if inst.has("pred_boxes")
        else np.zeros((0, 4), dtype=np.float32)
    )
    scores = (
        inst.scores.detach().cpu().numpy().astype(np.float32)
        if inst.has("scores")
        else np.zeros((0,), dtype=np.float32)
    )
    classes = (
        inst.pred_classes.detach().cpu().numpy().astype(np.int64)
        if inst.has("pred_classes")
        else np.zeros((0,), dtype=np.int64)
    )
    descriptions = list(inst.pred_object_descriptions.data) if inst.has("pred_object_descriptions") else []
    det_obj = list(inst.det_obj.data) if inst.has("det_obj") else []

    return {
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
        "descriptions": descriptions,
        "det_obj": det_obj,
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


@pytest.fixture(scope="module")
def parity_outputs():
    _ensure_enabled_or_skip()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this integration test")

    _maybe_add_vbench_submodule_path()

    try:
        import detectron2  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent skip
        pytest.skip(f"detectron2 import failed: {exc}")

    from vbench.third_party.grit_model import DenseCaptioning
    from vbench.third_party.grit_src.image_dense_captions import dense_pred_to_caption_tuple
    from vbench.utils import load_video

    video_path = _resolve_video_path()
    weight_path = _resolve_weight_path()

    try:
        frames = load_video(str(video_path), num_frames=8, return_tensor=False)
    except Exception as exc:  # pragma: no cover - env-dependent skip
        pytest.skip(f"Failed to decode video for parity test: {exc}")

    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        pytest.skip(f"Unexpected frame shape from load_video: {frames.shape}")

    device = torch.device("cuda:0")

    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        model = DenseCaptioning(device)
        model.initialize_model_det(model_weight=str(weight_path))

        compat.patch_grit_batch_inference_compat(
            enable_batch_parallel=False,
            batch_size=1,
            autocast_enabled=False,
            deterministic=True,
        )
        preds_bs1 = model.demo.run_on_batch(frames)

        compat.patch_grit_batch_inference_compat(
            enable_batch_parallel=True,
            batch_size=4,
            autocast_enabled=False,
            deterministic=True,
        )
        preds_bs4 = model.demo.run_on_batch(frames)
    finally:
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic

    sig_bs1 = [_serialize_prediction(p) for p in preds_bs1]
    sig_bs4 = [_serialize_prediction(p) for p in preds_bs4]

    caps_bs1 = [dense_pred_to_caption_tuple(p) for p in preds_bs1]
    caps_bs4 = [dense_pred_to_caption_tuple(p) for p in preds_bs4]

    return {
        "video": str(video_path),
        "frames": int(frames.shape[0]),
        "sig_bs1": sig_bs1,
        "sig_bs4": sig_bs4,
        "caps_bs1": caps_bs1,
        "caps_bs4": caps_bs4,
    }


def test_grit_batch_parallel_predictions_have_high_semantic_overlap(parity_outputs):
    sig_bs1 = parity_outputs["sig_bs1"]
    sig_bs4 = parity_outputs["sig_bs4"]

    assert len(sig_bs1) == len(sig_bs4)
    jaccards = []
    for idx, (a, b) in enumerate(zip(sig_bs1, sig_bs4, strict=False)):
        assert abs(a["boxes"].shape[0] - b["boxes"].shape[0]) <= 2, (
            f"frame={idx} detection count drift too large: "
            f"{a['boxes'].shape[0]} vs {b['boxes'].shape[0]}"
        )
        set_a = set(a["det_obj"])
        set_b = set(b["det_obj"])
        j = _jaccard(set_a, set_b)
        jaccards.append(j)

    assert min(jaccards) >= 0.75
    assert float(np.mean(jaccards)) >= 0.9


def test_grit_batch_parallel_caption_tuple_api_is_close(parity_outputs):
    caps_bs1 = parity_outputs["caps_bs1"]
    caps_bs4 = parity_outputs["caps_bs4"]

    assert len(caps_bs1) == len(caps_bs4)
    text_overlap = []
    for idx, (c1, c4) in enumerate(zip(caps_bs1, caps_bs4, strict=False)):
        assert abs(len(c1) - len(c4)) <= 2, f"frame={idx} caption count mismatch"
        desc1 = {str(t[0]) for t in c1}
        desc4 = {str(t[0]) for t in c4}
        text_overlap.append(_jaccard(desc1, desc4))

    assert min(text_overlap) >= 0.7
    assert float(np.mean(text_overlap)) >= 0.85
