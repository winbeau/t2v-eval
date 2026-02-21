"""
Tests for scripts/run_eval_core.py — pipeline orchestrator.

Covers:
  - _submodule_key_file()
  - _submodule_ready()
  - check_submodules()
  - run_script()
  - copy_outputs_to_frontend()
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.run_eval_core import (
    _submodule_key_file,
    _submodule_ready,
    copy_outputs_to_frontend,
    run_script,
)


# ---------------------------------------------------------------------------
# _submodule_key_file
# ---------------------------------------------------------------------------
class TestSubmoduleKeyFile:
    def test_vbench_key_file(self, tmp_path):
        result = _submodule_key_file("VBench", tmp_path)
        assert result == tmp_path / "vbench" / "__init__.py"

    def test_t2v_metrics_key_file_package(self, tmp_path):
        """t2v_metrics as package (directory with __init__.py)."""
        init_file = tmp_path / "t2v_metrics" / "__init__.py"
        init_file.parent.mkdir(parents=True)
        init_file.write_text("")
        result = _submodule_key_file("t2v_metrics", tmp_path)
        assert result == init_file

    def test_t2v_metrics_key_file_single(self, tmp_path):
        """t2v_metrics as single file (t2v_metrics.py)."""
        result = _submodule_key_file("t2v_metrics", tmp_path)
        # When __init__.py doesn't exist, falls back to t2v_metrics.py
        assert result == tmp_path / "t2v_metrics.py"


# ---------------------------------------------------------------------------
# _submodule_ready
# ---------------------------------------------------------------------------
class TestSubmoduleReady:
    def test_ready_submodule(self, tmp_path):
        """Properly initialized submodule should return True."""
        (tmp_path / ".git").write_text("")
        (tmp_path / "t2v_metrics").mkdir()
        (tmp_path / "t2v_metrics" / "__init__.py").write_text("")
        assert _submodule_ready("t2v_metrics", tmp_path) is True

    def test_missing_directory(self, tmp_path):
        assert _submodule_ready("t2v_metrics", tmp_path / "nonexistent") is False

    def test_no_git_file(self, tmp_path):
        (tmp_path / "t2v_metrics").mkdir()
        (tmp_path / "t2v_metrics" / "__init__.py").write_text("")
        assert _submodule_ready("t2v_metrics", tmp_path) is False

    def test_no_key_file(self, tmp_path):
        (tmp_path / ".git").write_text("")
        assert _submodule_ready("t2v_metrics", tmp_path) is False


# ---------------------------------------------------------------------------
# run_script
# ---------------------------------------------------------------------------
class TestRunScript:
    def test_nonexistent_script(self, sample_config):
        _, config_path = sample_config
        result = run_script("nonexistent_script.py", str(config_path))
        assert result is False

    def test_run_script_timeout(self, sample_config):
        """Script that exceeds timeout should return False."""
        _, config_path = sample_config
        with patch("scripts.run_eval_core.subprocess.run") as mock_run:
            from subprocess import TimeoutExpired

            mock_run.side_effect = TimeoutExpired(cmd="test", timeout=1)
            result = run_script("run_flicker.py", str(config_path))
        assert result is False

    def test_run_script_generic_error(self, sample_config):
        _, config_path = sample_config
        with patch("scripts.run_eval_core.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Permission denied")
            result = run_script("run_flicker.py", str(config_path))
        assert result is False


# ---------------------------------------------------------------------------
# copy_outputs_to_frontend
# ---------------------------------------------------------------------------
class TestCopyOutputsToFrontend:
    def test_copies_group_summary(self, tmp_path):
        """Copies group_summary.csv and updates manifest.json."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        (output_dir / "group_summary.csv").write_text("group,score\na,1.0\n")

        frontend_dir = tmp_path / "frontend" / "public" / "data"

        config = {
            "paths": {
                "output_dir": str(output_dir),
                "group_summary": "group_summary.csv",
                "per_video_metrics": "per_video_metrics.csv",
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Patch PROJECT_ROOT so frontend dir is under tmp_path
        with patch("scripts.run_eval_core.PROJECT_ROOT", tmp_path):
            result = copy_outputs_to_frontend(str(config_path))

        assert result is True
        assert (frontend_dir / "group_summary.csv").exists()
        assert (frontend_dir / "manifest.json").exists()

        manifest = json.loads((frontend_dir / "manifest.json").read_text())
        assert "group_summary.csv" in manifest["files"]

    def test_updates_existing_manifest(self, tmp_path):
        """Existing manifest files are preserved when adding new ones."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        (output_dir / "group_summary.csv").write_text("data\n")

        frontend_dir = tmp_path / "frontend" / "public" / "data"
        frontend_dir.mkdir(parents=True)
        manifest = {"files": ["old_file.csv"]}
        (frontend_dir / "manifest.json").write_text(json.dumps(manifest))

        config = {
            "paths": {
                "output_dir": str(output_dir),
                "group_summary": "group_summary.csv",
                "per_video_metrics": "per_video_metrics.csv",
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("scripts.run_eval_core.PROJECT_ROOT", tmp_path):
            copy_outputs_to_frontend(str(config_path))

        updated = json.loads((frontend_dir / "manifest.json").read_text())
        assert "old_file.csv" in updated["files"]
        assert "group_summary.csv" in updated["files"]

    def test_handles_missing_output_dir(self, tmp_path):
        config = {
            "paths": {
                "output_dir": str(tmp_path / "nonexistent"),
                "group_summary": "group_summary.csv",
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("scripts.run_eval_core.PROJECT_ROOT", tmp_path):
            result = copy_outputs_to_frontend(str(config_path))
        # Should not crash — returns True even if no files copied
        assert result is True
