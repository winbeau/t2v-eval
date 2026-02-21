"""
Tests for scripts/run_clip_or_vqa.py — CLIP/VQA evaluation.

Limited to functions testable without external submodules or GPU.

Covers:
  - check_t2v_metrics_installation()
  - setup_t2v_metrics_path()
  - load_config()
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_clip_or_vqa import (
    check_t2v_metrics_installation,
    load_config,
    setup_t2v_metrics_path,
)


# ---------------------------------------------------------------------------
# check_t2v_metrics_installation
# ---------------------------------------------------------------------------
class TestCheckT2vMetricsInstallation:
    def test_returns_true_when_init_exists(self, tmp_path):
        init_file = tmp_path / "t2v_metrics" / "__init__.py"
        init_file.parent.mkdir(parents=True)
        init_file.write_text("")

        with patch("scripts.run_clip_or_vqa.T2V_METRICS_ROOT", tmp_path):
            assert check_t2v_metrics_installation() is True

    def test_returns_true_when_single_file_exists(self, tmp_path):
        single_file = tmp_path / "t2v_metrics.py"
        single_file.write_text("")

        with patch("scripts.run_clip_or_vqa.T2V_METRICS_ROOT", tmp_path):
            assert check_t2v_metrics_installation() is True

    def test_returns_false_when_missing(self, tmp_path):
        with patch("scripts.run_clip_or_vqa.T2V_METRICS_ROOT", tmp_path):
            assert check_t2v_metrics_installation() is False


# ---------------------------------------------------------------------------
# setup_t2v_metrics_path
# ---------------------------------------------------------------------------
class TestSetupT2vMetricsPath:
    def test_adds_to_sys_path(self, tmp_path):
        with patch("scripts.run_clip_or_vqa.T2V_METRICS_ROOT", tmp_path):
            setup_t2v_metrics_path()
            assert str(tmp_path) in sys.path
            # Clean up
            sys.path.remove(str(tmp_path))

    def test_idempotent(self, tmp_path):
        with patch("scripts.run_clip_or_vqa.T2V_METRICS_ROOT", tmp_path):
            setup_t2v_metrics_path()
            setup_t2v_metrics_path()  # second call
            count = sys.path.count(str(tmp_path))
            assert count == 1
            sys.path.remove(str(tmp_path))


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_loads_clip_config(self, sample_config):
        config, path = sample_config
        loaded = load_config(str(path))
        clip_config = loaded["metrics"]["clip_or_vqa"]
        assert clip_config["mode"] == "clip"
        assert clip_config["model_name"] == "ViT-B-32"
