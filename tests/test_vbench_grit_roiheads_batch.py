"""Regression checks for GRiT ROIHeads multi-image inference patch."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROI_HEADS_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "VBench"
    / "vbench"
    / "third_party"
    / "grit_src"
    / "grit"
    / "modeling"
    / "roi_heads"
    / "grit_roi_heads.py"
)


def _load_source() -> str:
    return ROI_HEADS_PATH.read_text(encoding="utf-8")


def test_roiheads_no_single_image_assertions():
    source = _load_source()
    assert "assert len(boxes) == 1" not in source
    assert "assert len(pred_instances) == 1" not in source


def test_roiheads_beam_merge_uses_per_image_size():
    source = _load_source()
    assert "Instances(image_sizes[i])" in source
