"""
Tests for scripts/run_flicker.py — temporal flicker metric.

Covers:
  - compute_flicker_score() with synthetic frames
  - L1 vs L2 methods
  - Grayscale mode
  - Normalize flag
  - Edge cases (single frame, identical frames)
  - load_config()
"""

from unittest.mock import patch

import numpy as np
import pytest

# We import the module functions directly; read_video_frames is mocked
# to avoid needing actual video files.
from scripts.run_flicker import compute_flicker_score, load_config


# ---------------------------------------------------------------------------
# Helpers to mock read_video_frames
# ---------------------------------------------------------------------------
def _make_mock_read(frames: np.ndarray):
    """Return a mock that replaces read_video_frames and returns *frames*."""
    def _read(video_path: str) -> np.ndarray:
        return frames
    return _read


# ---------------------------------------------------------------------------
# compute_flicker_score — basic behaviour
# ---------------------------------------------------------------------------
class TestComputeFlickerScore:
    """Tests for the core compute_flicker_score function."""

    def test_identical_frames_zero_flicker(self, constant_frames):
        """Identical frames should produce zero flicker (mean & std)."""
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(constant_frames)):
            mean, std = compute_flicker_score("dummy.mp4")
        assert mean == pytest.approx(0.0, abs=1e-7)
        assert std == pytest.approx(0.0, abs=1e-7)

    def test_gradient_frames_positive_flicker(self, gradient_frames):
        """Frames with increasing brightness should yield positive flicker."""
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(gradient_frames)):
            mean, std = compute_flicker_score("dummy.mp4")
        assert mean > 0
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_gradient_frames_l1_value(self, gradient_frames):
        """L1 flicker on uniform-gradient frames has a known analytical value.

        Each frame is constant val = i*0.1; consecutive diff = 0.1 for all pixels.
        flicker_mean = 0.1, flicker_std = 0.0
        """
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(gradient_frames)):
            mean, std = compute_flicker_score("dummy.mp4", method="l1")
        assert mean == pytest.approx(0.1, abs=1e-6)
        assert std == pytest.approx(0.0, abs=1e-6)

    def test_l2_method(self, gradient_frames):
        """L2 method should return root-mean-square difference."""
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(gradient_frames)):
            mean, std = compute_flicker_score("dummy.mp4", method="l2")
        # For uniform diff of 0.1 across all pixels, L2 == L1 == 0.1
        assert mean == pytest.approx(0.1, abs=1e-6)

    def test_unknown_method_raises(self, gradient_frames):
        """Unknown method should raise ValueError."""
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(gradient_frames)):
            with pytest.raises(ValueError, match="Unknown method"):
                compute_flicker_score("dummy.mp4", method="l3")

    def test_single_frame_returns_zero(self, single_frame):
        """Videos with <2 frames should return (0.0, 0.0)."""
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(single_frame)):
            mean, std = compute_flicker_score("dummy.mp4")
        assert mean == 0.0
        assert std == 0.0

    def test_grayscale_mode(self, gradient_frames):
        """Grayscale conversion should still compute valid flicker."""
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(gradient_frames)):
            mean, std = compute_flicker_score("dummy.mp4", grayscale=True)
        assert mean > 0
        assert isinstance(mean, float)

    def test_normalize_false(self, gradient_frames):
        """When normalize=False, frames are scaled to [0, 255]."""
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(gradient_frames)):
            mean_norm, _ = compute_flicker_score("dummy.mp4", normalize=True)
            mean_raw, _ = compute_flicker_score("dummy.mp4", normalize=False)
        # Raw values should be ~255x larger
        assert mean_raw == pytest.approx(mean_norm * 255.0, rel=0.01)

    def test_return_types(self, gradient_frames):
        """Return types must be (float, float)."""
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(gradient_frames)):
            result = compute_flicker_score("dummy.mp4")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# compute_flicker_score — two-frame edge case
# ---------------------------------------------------------------------------
class TestFlickerTwoFrames:
    """Minimum valid input: exactly 2 frames."""

    def test_two_identical_frames(self):
        frames = np.zeros((2, 4, 4, 3), dtype=np.float32)
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(frames)):
            mean, std = compute_flicker_score("dummy.mp4")
        assert mean == 0.0
        assert std == 0.0  # std of 1 element is 0

    def test_two_different_frames(self):
        frames = np.zeros((2, 4, 4, 3), dtype=np.float32)
        frames[1] = 1.0  # max difference
        with patch("scripts.run_flicker.read_video_frames", _make_mock_read(frames)):
            mean, std = compute_flicker_score("dummy.mp4")
        assert mean == pytest.approx(1.0, abs=1e-6)
        assert std == pytest.approx(0.0, abs=1e-6)  # single diff → zero std


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_loads_valid_yaml(self, sample_config):
        config, config_path = sample_config
        loaded = load_config(str(config_path))
        assert loaded["metrics"]["flicker"]["method"] == "l1"
        assert loaded["runtime"]["device"] == "cpu"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
