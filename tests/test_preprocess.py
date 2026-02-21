"""
Tests for scripts/preprocess_videos.py — video preprocessing.

Covers:
  - compute_frame_indices() — all sampling/padding strategies
  - load_config()
"""

import numpy as np
import pytest

from scripts.preprocess_videos import compute_frame_indices, load_config


# ---------------------------------------------------------------------------
# compute_frame_indices — uniform sampling (enough frames)
# ---------------------------------------------------------------------------
class TestComputeFrameIndicesUniform:
    """Uniform sampling when total_frames >= target_frames."""

    def test_exact_match(self):
        """total_frames == target_frames → identity indices."""
        indices = compute_frame_indices(16, 16, sampling="uniform")
        assert len(indices) == 16
        assert indices[0] == 0
        assert indices[-1] == 15

    def test_downsample(self):
        """total_frames > target_frames → uniformly spaced."""
        indices = compute_frame_indices(100, 10, sampling="uniform")
        assert len(indices) == 10
        assert indices[0] == 0
        assert indices[-1] == 99
        # Should be monotonically increasing
        assert all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))

    def test_large_downsample(self):
        """Extreme downsampling: 1000 frames → 4 frames."""
        indices = compute_frame_indices(1000, 4, sampling="uniform")
        assert len(indices) == 4
        assert indices[0] == 0
        assert indices[-1] == 999

    def test_sequential_sampling(self):
        """Non-uniform (sequential) sampling."""
        indices = compute_frame_indices(100, 10, sampling="sequential")
        assert len(indices) == 10
        np.testing.assert_array_equal(indices, np.arange(10))


# ---------------------------------------------------------------------------
# compute_frame_indices — padding strategies (insufficient frames)
# ---------------------------------------------------------------------------
class TestComputeFrameIndicesPadding:
    """Padding when total_frames < target_frames."""

    def test_loop_padding(self):
        """Loop padding cycles through available frames."""
        indices = compute_frame_indices(3, 7, padding="loop")
        assert len(indices) == 7
        # Should cycle: 0,1,2,0,1,2,0
        expected = np.array([0, 1, 2, 0, 1, 2, 0])
        np.testing.assert_array_equal(indices, expected)

    def test_repeat_last_padding(self):
        """Repeat-last pads with the last frame index."""
        indices = compute_frame_indices(3, 7, padding="repeat_last")
        assert len(indices) == 7
        expected = np.array([0, 1, 2, 2, 2, 2, 2])
        np.testing.assert_array_equal(indices, expected)

    def test_truncate_padding(self):
        """Truncate returns only available frames."""
        indices = compute_frame_indices(3, 7, padding="truncate")
        assert len(indices) == 3
        np.testing.assert_array_equal(indices, np.array([0, 1, 2]))

    def test_unknown_padding_raises(self):
        """Unknown padding strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown padding strategy"):
            compute_frame_indices(3, 7, padding="unknown")

    def test_loop_exact_multiple(self):
        """Loop when target is exact multiple of total."""
        indices = compute_frame_indices(3, 6, padding="loop")
        assert len(indices) == 6
        expected = np.array([0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(indices, expected)

    def test_single_frame_loop(self):
        """Single source frame looped to target."""
        indices = compute_frame_indices(1, 5, padding="loop")
        assert len(indices) == 5
        np.testing.assert_array_equal(indices, np.zeros(5, dtype=int))

    def test_single_frame_repeat_last(self):
        """Single source frame repeated."""
        indices = compute_frame_indices(1, 5, padding="repeat_last")
        assert len(indices) == 5
        np.testing.assert_array_equal(indices, np.zeros(5, dtype=int))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestComputeFrameIndicesEdge:
    def test_target_equals_one(self):
        """Request exactly 1 frame from multi-frame video."""
        indices = compute_frame_indices(100, 1, sampling="uniform")
        assert len(indices) == 1

    def test_both_one(self):
        """1 frame source, 1 frame target."""
        indices = compute_frame_indices(1, 1, sampling="uniform")
        assert len(indices) == 1
        assert indices[0] == 0

    def test_indices_are_integers(self):
        """All returned indices should be integer type."""
        indices = compute_frame_indices(50, 8, sampling="uniform")
        assert indices.dtype in (np.int32, np.int64, np.intp)

    def test_indices_in_range(self):
        """All indices should be within [0, total_frames - 1]."""
        total = 30
        indices = compute_frame_indices(total, 10, sampling="uniform")
        assert all(0 <= idx < total for idx in indices)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_loads_protocol_section(self, sample_config):
        config, path = sample_config
        loaded = load_config(str(path))
        assert "protocol" in loaded
        assert loaded["protocol"]["fps_eval"] == 8
        assert loaded["protocol"]["num_frames"] == 16
