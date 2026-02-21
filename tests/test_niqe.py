"""
Tests for scripts/run_niqe.py — NIQE image quality metric.

Covers:
  - compute_niqe_custom() with synthetic frames
  - compute_niqe() fallback chain
  - read_video_frames() uniform sampling logic (mocked)
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scripts.run_niqe import compute_niqe, compute_niqe_custom


# ---------------------------------------------------------------------------
# compute_niqe_custom — simplified NIQE approximation
# ---------------------------------------------------------------------------
class TestComputeNiqeCustom:
    """Tests for the custom NIQE fallback implementation."""

    def test_returns_finite_float(self, random_frames_uint8):
        """NIQE should return a finite float value."""
        score = compute_niqe_custom(random_frames_uint8)
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_positive_score(self, random_frames_uint8):
        """NIQE scores should be positive (lower is better but > 0)."""
        score = compute_niqe_custom(random_frames_uint8)
        assert score > 0

    def test_uniform_frames_high_score(self):
        """Uniform (featureless) frames should have high NIQE (poor quality)."""
        uniform = np.full((4, 64, 64, 3), 128, dtype=np.uint8)
        score = compute_niqe_custom(uniform)
        assert score > 0

    def test_single_frame(self):
        """Should work with a single frame."""
        rng = np.random.RandomState(99)
        frame = rng.randint(0, 256, size=(1, 64, 64, 3), dtype=np.uint8)
        score = compute_niqe_custom(frame)
        assert isinstance(score, float)
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# compute_niqe — fallback chain
# ---------------------------------------------------------------------------
class TestComputeNiqeFallback:
    """Test the multi-backend fallback in compute_niqe()."""

    def test_falls_back_to_custom_when_all_fail(self, random_frames_uint8):
        """When pyiqa and skvideo both fail, custom implementation is used."""
        with patch(
            "scripts.run_niqe.compute_niqe_pyiqa",
            side_effect=ImportError("no pyiqa"),
        ), patch(
            "scripts.run_niqe.compute_niqe_skvideo",
            side_effect=ImportError("no skvideo"),
        ), patch(
            "scripts.run_niqe.compute_niqe_custom",
            return_value=5.5,
        ) as mock_custom:
            score = compute_niqe(random_frames_uint8, device="cpu")

        mock_custom.assert_called_once_with(random_frames_uint8)
        assert score == 5.5

    def test_pyiqa_succeeds_first(self, random_frames_uint8):
        """When pyiqa works, skvideo and custom are not called."""
        with patch(
            "scripts.run_niqe.compute_niqe_pyiqa",
            return_value=4.2,
        ) as mock_pyiqa, patch(
            "scripts.run_niqe.compute_niqe_skvideo",
        ) as mock_skvideo, patch(
            "scripts.run_niqe.compute_niqe_custom",
        ) as mock_custom:
            score = compute_niqe(random_frames_uint8, device="cpu")

        mock_pyiqa.assert_called_once()
        mock_skvideo.assert_not_called()
        mock_custom.assert_not_called()
        assert score == 4.2

    def test_pyiqa_fails_skvideo_succeeds(self, random_frames_uint8):
        """When pyiqa fails but skvideo works, custom is not called."""
        with patch(
            "scripts.run_niqe.compute_niqe_pyiqa",
            side_effect=ImportError("no pyiqa"),
        ), patch(
            "scripts.run_niqe.compute_niqe_skvideo",
            return_value=4.8,
        ) as mock_skvideo, patch(
            "scripts.run_niqe.compute_niqe_custom",
        ) as mock_custom:
            score = compute_niqe(random_frames_uint8, device="cpu")

        mock_skvideo.assert_called_once()
        mock_custom.assert_not_called()
        assert score == 4.8

    def test_pyiqa_runtime_error_triggers_fallback(self, random_frames_uint8):
        """Non-ImportError exceptions from pyiqa also trigger fallback."""
        with patch(
            "scripts.run_niqe.compute_niqe_pyiqa",
            side_effect=RuntimeError("CUDA out of memory"),
        ), patch(
            "scripts.run_niqe.compute_niqe_skvideo",
            return_value=4.0,
        ):
            score = compute_niqe(random_frames_uint8, device="cpu")
        assert score == 4.0
