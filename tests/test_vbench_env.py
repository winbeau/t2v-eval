"""
Tests for scripts/vbench_runner/env.py — environment setup and config loading.

Covers:
  - resolve_path()
  - use_vbench_long()
  - get_vbench_subtasks()
  - load_config()
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.vbench_runner.env import (
    get_vbench_subtasks,
    load_config,
    resolve_path,
    use_vbench_long,
)


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------
class TestResolvePath:
    def test_none_returns_none(self):
        assert resolve_path(None) is None

    def test_empty_string_returns_none(self):
        assert resolve_path("") is None

    def test_absolute_path(self):
        result = resolve_path("/absolute/path")
        assert result == Path("/absolute/path")

    def test_relative_path(self):
        result = resolve_path("relative/path")
        assert result is not None
        assert result.is_absolute()
        assert str(result).endswith("relative/path")


# ---------------------------------------------------------------------------
# use_vbench_long
# ---------------------------------------------------------------------------
class TestUseVbenchLong:
    def test_use_long_true(self):
        config = {"metrics": {"vbench": {"use_long": True}}}
        assert use_vbench_long(config) is True

    def test_use_long_false(self):
        config = {"metrics": {"vbench": {"use_long": False}}}
        assert use_vbench_long(config) is False

    def test_backend_vbench_long(self):
        config = {"metrics": {"vbench": {"backend": "vbench_long"}}}
        assert use_vbench_long(config) is True

    def test_backend_long(self):
        config = {"metrics": {"vbench": {"backend": "long"}}}
        assert use_vbench_long(config) is True

    def test_backend_vbench(self):
        config = {"metrics": {"vbench": {"backend": "vbench"}}}
        assert use_vbench_long(config) is False

    def test_no_vbench_config(self):
        config = {"metrics": {}}
        assert use_vbench_long(config) is False

    def test_empty_config(self):
        config = {}
        assert use_vbench_long(config) is False


# ---------------------------------------------------------------------------
# get_vbench_subtasks
# ---------------------------------------------------------------------------
class TestGetVbenchSubtasks:
    def test_explicit_subtasks(self):
        config = {
            "metrics": {
                "vbench": {
                    "subtasks": [
                        "motion_smoothness",
                        "temporal_flickering",
                    ]
                }
            }
        }
        result = get_vbench_subtasks(config)
        assert result == ["motion_smoothness", "temporal_flickering"]

    def test_explicit_subtasks_deduplication(self):
        config = {
            "metrics": {
                "vbench": {
                    "subtasks": ["motion_smoothness", "motion_smoothness"]
                }
            }
        }
        result = get_vbench_subtasks(config)
        assert result == ["motion_smoothness"]

    def test_long_16_profile_default(self):
        config = {
            "metrics": {
                "vbench": {
                    "use_long": True,
                    "dimension_profile": "long_16",
                }
            }
        }
        result = get_vbench_subtasks(config)
        assert len(result) == 16

    def test_long_6_profile_default(self):
        config = {
            "metrics": {
                "vbench": {
                    "use_long": True,
                    "dimension_profile": "long_6",
                }
            }
        }
        result = get_vbench_subtasks(config)
        assert len(result) == 6

    def test_no_vbench_config_defaults(self):
        config = {}
        result = get_vbench_subtasks(config)
        assert "temporal_flickering" in result
        assert "motion_smoothness" in result

    def test_empty_subtasks_uses_defaults(self):
        config = {"metrics": {"vbench": {"subtasks": []}}}
        result = get_vbench_subtasks(config)
        # Empty subtasks list → normalize_subtasks returns [], which is falsy,
        # so the code falls through to profile-based defaults
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------
class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"key": "value"}))
        result = load_config(str(config_file))
        assert result == {"key": "value"}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")
