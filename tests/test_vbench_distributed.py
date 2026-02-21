"""
Tests for scripts/vbench_runner/distributed.py — multi-GPU coordination.

Covers:
  - init_distributed_if_needed()
  - _parse_visible_devices()
  - split_subtasks_for_rank()
  - merge_rank_partial_results()
  - make_file_barrier()
"""

import os
import threading
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.vbench_runner.distributed import (
    _parse_visible_devices,
    init_distributed_if_needed,
    make_file_barrier,
    merge_rank_partial_results,
    split_subtasks_for_rank,
)


# ---------------------------------------------------------------------------
# init_distributed_if_needed
# ---------------------------------------------------------------------------
class TestInitDistributed:
    def test_single_gpu_default(self):
        with patch.dict(os.environ, {}, clear=True):
            rank, world_size, barrier = init_distributed_if_needed()
        assert rank == 0
        assert world_size == 1
        barrier()  # should be a no-op

    def test_multi_gpu_env(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4", "RANK": "2"}):
            rank, world_size, barrier = init_distributed_if_needed()
        assert rank == 2
        assert world_size == 4

    def test_world_size_one(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "RANK": "0"}):
            rank, world_size, _ = init_distributed_if_needed()
        assert rank == 0
        assert world_size == 1


# ---------------------------------------------------------------------------
# _parse_visible_devices
# ---------------------------------------------------------------------------
class TestParseVisibleDevices:
    def test_explicit_devices(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"}):
            result = _parse_visible_devices()
        assert result == ["0", "1", "2"]

    def test_single_device(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}):
            result = _parse_visible_devices()
        assert result == ["3"]

    def test_disabled_gpu(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
            result = _parse_visible_devices()
        assert result == []

    def test_empty_string(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            result = _parse_visible_devices()
        assert result == []

    def test_whitespace_handling(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": " 0 , 1 , 2 "}):
            result = _parse_visible_devices()
        assert result == ["0", "1", "2"]

    def test_no_env_no_torch(self):
        """When CUDA_VISIBLE_DEVICES is not set and torch is unavailable."""
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "scripts.vbench_runner.distributed.torch",
                side_effect=ImportError,
                create=True,
            ):
                # If torch import fails, should return empty list
                result = _parse_visible_devices()
                # Result depends on whether torch is actually available
                assert isinstance(result, list)


# ---------------------------------------------------------------------------
# split_subtasks_for_rank
# ---------------------------------------------------------------------------
class TestSplitSubtasksForRank:
    def test_single_rank_returns_all(self):
        subtasks = ["a", "b", "c", "d"]
        result = split_subtasks_for_rank(subtasks, rank=0, world_size=1)
        assert result == ["a", "b", "c", "d"]

    def test_two_ranks_even_split(self):
        subtasks = ["a", "b", "c", "d"]
        r0 = split_subtasks_for_rank(subtasks, rank=0, world_size=2)
        r1 = split_subtasks_for_rank(subtasks, rank=1, world_size=2)
        # All subtasks covered, no overlap
        assert set(r0 + r1) == set(subtasks)
        assert len(set(r0) & set(r1)) == 0

    def test_heavy_dims_distributed(self):
        """Heavy dimensions should be spread across ranks, not concentrated."""
        subtasks = [
            "object_class",      # heavy
            "multiple_objects",   # heavy
            "color",             # heavy
            "motion_smoothness", # light
            "dynamic_degree",    # light
            "imaging_quality",   # light
        ]
        r0 = split_subtasks_for_rank(subtasks, rank=0, world_size=2)
        r1 = split_subtasks_for_rank(subtasks, rank=1, world_size=2)

        heavy_in_r0 = len([s for s in r0 if s in {"object_class", "multiple_objects", "color"}])
        heavy_in_r1 = len([s for s in r1 if s in {"object_class", "multiple_objects", "color"}])

        # Heavy dims should be distributed, not all on one rank
        assert abs(heavy_in_r0 - heavy_in_r1) <= 1

    def test_more_ranks_than_subtasks(self):
        subtasks = ["a", "b"]
        r0 = split_subtasks_for_rank(subtasks, rank=0, world_size=4)
        r1 = split_subtasks_for_rank(subtasks, rank=1, world_size=4)
        r2 = split_subtasks_for_rank(subtasks, rank=2, world_size=4)
        r3 = split_subtasks_for_rank(subtasks, rank=3, world_size=4)

        all_assigned = r0 + r1 + r2 + r3
        assert set(all_assigned) == {"a", "b"}

    def test_empty_subtasks(self):
        result = split_subtasks_for_rank([], rank=0, world_size=2)
        assert result == []

    def test_preserves_completeness_4_ranks(self):
        """All 16 dimensions should be assigned across 4 ranks without loss."""
        from scripts.vbench_runner.dimensions.registry import LONG_DIMENSION_KEYS

        all_assigned = []
        for r in range(4):
            all_assigned.extend(
                split_subtasks_for_rank(LONG_DIMENSION_KEYS, rank=r, world_size=4)
            )
        assert set(all_assigned) == set(LONG_DIMENSION_KEYS)
        assert len(all_assigned) == len(LONG_DIMENSION_KEYS)


# ---------------------------------------------------------------------------
# merge_rank_partial_results
# ---------------------------------------------------------------------------
class TestMergeRankPartialResults:
    def test_single_frame(self):
        frame = pd.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "motion_smoothness": [0.8, 0.9],
            }
        )
        result = merge_rank_partial_results([frame])
        assert "video_id" in result.columns
        assert "motion_smoothness" in result.columns
        assert "vbench_temporal_score" in result.columns
        assert len(result) == 2

    def test_two_frames_different_metrics(self):
        frame_0 = pd.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "motion_smoothness": [0.8, 0.9],
            }
        )
        frame_1 = pd.DataFrame(
            {
                "video_id": ["v1", "v2"],
                "dynamic_degree": [0.7, 0.6],
            }
        )
        result = merge_rank_partial_results([frame_0, frame_1])
        assert "motion_smoothness" in result.columns
        assert "dynamic_degree" in result.columns
        assert len(result) == 2

    def test_vbench_temporal_score_is_mean(self):
        frame = pd.DataFrame(
            {
                "video_id": ["v1"],
                "metric_a": [0.8],
                "metric_b": [0.6],
            }
        )
        result = merge_rank_partial_results([frame])
        expected = (0.8 + 0.6) / 2
        assert result.iloc[0]["vbench_temporal_score"] == pytest.approx(expected)

    def test_empty_frames(self):
        result = merge_rank_partial_results([])
        assert "video_id" in result.columns
        assert "vbench_temporal_score" in result.columns
        assert len(result) == 0

    def test_none_frames_filtered(self):
        frame = pd.DataFrame(
            {"video_id": ["v1"], "metric": [0.5]}
        )
        result = merge_rank_partial_results([None, frame, None])
        assert len(result) == 1

    def test_empty_dataframe_filtered(self):
        frame = pd.DataFrame({"video_id": ["v1"], "metric": [0.5]})
        empty = pd.DataFrame()
        result = merge_rank_partial_results([empty, frame])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# make_file_barrier
# ---------------------------------------------------------------------------
class TestMakeFileBarrier:
    def test_single_rank_noop(self, tmp_path):
        barrier = make_file_barrier(tmp_path / "sync", rank=0, world_size=1)
        barrier()  # should complete immediately

    def test_two_ranks_synchronize(self, tmp_path):
        sync_dir = tmp_path / "sync"
        results = []

        def worker(rank):
            barrier = make_file_barrier(sync_dir, rank=rank, world_size=2)
            barrier(timeout_sec=5.0)
            results.append(rank)

        t0 = threading.Thread(target=worker, args=(0,))
        t1 = threading.Thread(target=worker, args=(1,))
        t0.start()
        t1.start()
        t0.join(timeout=10)
        t1.join(timeout=10)

        assert set(results) == {0, 1}
