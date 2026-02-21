"""
Shared fixtures for t2v-eval test suite.

Provides reusable test data (synthetic video frames, sample configs,
CSV files, metadata DataFrames) used across multiple test modules.
"""

import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Synthetic video frames
# ---------------------------------------------------------------------------
@pytest.fixture
def constant_frames():
    """4 identical frames (T=4, H=8, W=8, C=3), float32 [0,1]. Flicker should be 0."""
    frame = np.full((8, 8, 3), 0.5, dtype=np.float32)
    return np.stack([frame] * 4)


@pytest.fixture
def gradient_frames():
    """4 frames with increasing brightness. Predictable flicker."""
    frames = []
    for i in range(4):
        val = i * 0.1  # 0.0, 0.1, 0.2, 0.3
        frames.append(np.full((8, 8, 3), val, dtype=np.float32))
    return np.stack(frames)


@pytest.fixture
def random_frames_uint8():
    """8 random frames in uint8 RGB format (for NIQE tests)."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 256, size=(8, 64, 64, 3), dtype=np.uint8)


@pytest.fixture
def single_frame():
    """1 frame — edge case for flicker (less than 2 frames)."""
    return np.full((1, 8, 8, 3), 0.5, dtype=np.float32)


# ---------------------------------------------------------------------------
# Sample YAML config
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_config(tmp_path):
    """Create a minimal YAML config and return (config_dict, config_path)."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    cache_dir = tmp_path / "eval_cache"
    cache_dir.mkdir()

    config = {
        "dataset": {
            "repo_id": "test/dataset",
            "split": "test",
            "use_local_videos": True,
            "local_video_dir": str(tmp_path / "videos"),
            "prompt_file": str(tmp_path / "prompts.csv"),
        },
        "groups": [
            {"name": "group_a"},
            {"name": "group_b"},
        ],
        "metrics": {
            "enabled": ["flicker", "niqe"],
            "flicker": {
                "method": "l1",
                "normalize": True,
                "compute_std": True,
                "grayscale": False,
            },
            "niqe": {"num_frames_for_niqe": 4},
            "clip_or_vqa": {
                "mode": "clip",
                "model_name": "ViT-B-32",
                "pretrained": "openai",
                "num_frames_for_score": 8,
                "aggregation": "mean",
            },
            "vbench": {
                "enabled": True,
                "backend": "vbench_long",
                "use_long": True,
                "dimension_profile": "long_16",
                "mode": "long_custom_input",
                "subtasks": [
                    "subject_consistency",
                    "background_consistency",
                    "motion_smoothness",
                ],
            },
        },
        "runtime": {
            "device": "cpu",
            "batch_size": 1,
            "num_workers": 0,
            "seed": 42,
        },
        "paths": {
            "cache_dir": str(cache_dir),
            "output_dir": str(output_dir),
            "metadata_file": "metadata.csv",
            "processed_metadata": "processed_metadata.csv",
            "per_video_metrics": "per_video_metrics.csv",
            "group_summary": "group_summary.csv",
            "experiment_output": "test_experiment.csv",
        },
        "protocol": {
            "fps_eval": 8,
            "num_frames": 16,
            "resize": 256,
            "frame_sampling": "uniform",
            "frame_padding": "loop",
        },
        "logging": {
            "level": "INFO",
        },
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config, config_path


# ---------------------------------------------------------------------------
# Sample metadata DataFrames
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_metadata_df():
    """Base metadata DataFrame with video_id, group, prompt, video_path."""
    return pd.DataFrame(
        {
            "video_id": ["vid_001", "vid_002", "vid_003", "vid_004"],
            "group": ["group_a", "group_a", "group_b", "group_b"],
            "prompt": [
                "a cat walking",
                "a dog running",
                "a bird flying",
                "a fish swimming",
            ],
            "video_path": [
                "/tmp/videos/group_a/vid_001.mp4",
                "/tmp/videos/group_a/vid_002.mp4",
                "/tmp/videos/group_b/vid_003.mp4",
                "/tmp/videos/group_b/vid_004.mp4",
            ],
        }
    )


@pytest.fixture
def sample_processed_metadata_df():
    """Processed metadata with num_frames and duration_sec."""
    return pd.DataFrame(
        {
            "video_id": ["vid_001", "vid_002", "vid_003", "vid_004"],
            "group": ["group_a", "group_a", "group_b", "group_b"],
            "prompt": [
                "a cat walking",
                "a dog running",
                "a bird flying",
                "a fish swimming",
            ],
            "video_path": [
                "/tmp/cache/group_a/vid_001.mp4",
                "/tmp/cache/group_a/vid_002.mp4",
                "/tmp/cache/group_b/vid_003.mp4",
                "/tmp/cache/group_b/vid_004.mp4",
            ],
            "num_frames": [16, 16, 16, 16],
            "duration_sec": [2.0, 2.0, 2.0, 2.0],
        }
    )


@pytest.fixture
def sample_flicker_df():
    """Sample flicker per-video results."""
    return pd.DataFrame(
        {
            "video_id": ["vid_001", "vid_002", "vid_003", "vid_004"],
            "group": ["group_a", "group_a", "group_b", "group_b"],
            "flicker_mean": [0.01, 0.02, 0.015, 0.025],
            "flicker_std": [0.005, 0.008, 0.006, 0.01],
        }
    )


@pytest.fixture
def sample_niqe_df():
    """Sample NIQE per-video results."""
    return pd.DataFrame(
        {
            "video_id": ["vid_001", "vid_002", "vid_003", "vid_004"],
            "group": ["group_a", "group_a", "group_b", "group_b"],
            "niqe_mean": [5.1, 5.3, 4.8, 5.0],
        }
    )


@pytest.fixture
def sample_clipvqa_df():
    """Sample CLIP/VQA per-video results."""
    return pd.DataFrame(
        {
            "video_id": ["vid_001", "vid_002", "vid_003", "vid_004"],
            "group": ["group_a", "group_a", "group_b", "group_b"],
            "clip_or_vqa_score": [0.28, 0.31, 0.25, 0.29],
            "score_type": ["clip", "clip", "clip", "clip"],
        }
    )


@pytest.fixture
def sample_vbench_df():
    """Sample VBench per-video results."""
    return pd.DataFrame(
        {
            "video_id": ["vid_001", "vid_002", "vid_003", "vid_004"],
            "group": ["group_a", "group_a", "group_b", "group_b"],
            "vbench_temporal_score": [0.85, 0.88, 0.82, 0.86],
            "subject_consistency": [0.9, 0.92, 0.88, 0.91],
            "motion_smoothness": [0.8, 0.84, 0.76, 0.81],
        }
    )


# ---------------------------------------------------------------------------
# Prompt file fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_prompt_csv(tmp_path):
    """Create a sample prompts.csv and return its path."""
    prompt_file = tmp_path / "prompts.csv"
    df = pd.DataFrame(
        {
            "video_id": ["vid_001", "vid_002", "vid_003"],
            "prompt": [
                "a cat walking on the street",
                "a dog running in the park",
                "a bird flying over the ocean",
            ],
        }
    )
    df.to_csv(prompt_file, index=False)
    return prompt_file
