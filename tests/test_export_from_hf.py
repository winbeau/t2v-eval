"""
Tests for scripts/export_from_hf.py — HuggingFace dataset export and prompt loading.

Covers:
  - resolve_path()
  - extract_video_path()
  - infer_group_from_path()
  - load_prompt_file()
  - series_value()
  - build_prompt_indices()
  - find_prompt_row()
  - export_video_bytes()

Note: The `datasets` library (huggingface) is mocked at import time since it
may not be installed in the test environment.
"""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Mock the `datasets` module before importing export_from_hf,
# since `from datasets import load_dataset` runs at module level.
if "datasets" not in sys.modules:
    _mock_datasets = ModuleType("datasets")
    _mock_datasets.load_dataset = MagicMock()
    sys.modules["datasets"] = _mock_datasets

from scripts.export_from_hf import (
    build_prompt_indices,
    export_video_bytes,
    extract_video_path,
    find_prompt_row,
    infer_group_from_path,
    load_prompt_file,
    resolve_path,
    series_value,
)


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------
class TestResolvePath:
    def test_none_returns_none(self):
        assert resolve_path(None, Path("/base")) is None

    def test_absolute_path_unchanged(self):
        result = resolve_path("/absolute/path", Path("/base"))
        assert result == Path("/absolute/path")

    def test_relative_path_joined(self):
        result = resolve_path("relative/path", Path("/base"))
        assert result == Path("/base/relative/path")

    def test_dot_path(self):
        result = resolve_path(".", Path("/base"))
        assert result == Path("/base/.")


# ---------------------------------------------------------------------------
# extract_video_path
# ---------------------------------------------------------------------------
class TestExtractVideoPath:
    def test_dict_with_path(self):
        data = {"path": "/videos/test.mp4"}
        assert extract_video_path(data) == "/videos/test.mp4"

    def test_string_existing_file(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 10)
        assert extract_video_path(str(video)) == str(video)

    def test_string_nonexistent_file(self):
        assert extract_video_path("/nonexistent/video.mp4") is None

    def test_none_data(self):
        assert extract_video_path(None) is None

    def test_dict_without_path(self):
        assert extract_video_path({"bytes": b"\x00"}) is None


# ---------------------------------------------------------------------------
# infer_group_from_path
# ---------------------------------------------------------------------------
class TestInferGroupFromPath:
    def test_group_in_parts(self):
        result = infer_group_from_path(
            "/data/group_a/video.mp4", ["group_a", "group_b"]
        )
        assert result == "group_a"

    def test_group_in_filename(self):
        result = infer_group_from_path(
            "/data/videos/group_b_video.mp4", ["group_a", "group_b"]
        )
        assert result == "group_b"

    def test_no_match(self):
        result = infer_group_from_path(
            "/data/other/video.mp4", ["group_a", "group_b"]
        )
        assert result is None

    def test_none_path(self):
        assert infer_group_from_path(None, ["group_a"]) is None

    def test_empty_path(self):
        assert infer_group_from_path("", ["group_a"]) is None

    def test_empty_groups(self):
        assert infer_group_from_path("/data/group_a/video.mp4", []) is None


# ---------------------------------------------------------------------------
# load_prompt_file
# ---------------------------------------------------------------------------
class TestLoadPromptFile:
    def test_csv_with_standard_columns(self, tmp_path):
        prompt_file = tmp_path / "prompts.csv"
        pd.DataFrame(
            {"video_id": ["v1", "v2"], "prompt": ["cat", "dog"]}
        ).to_csv(prompt_file, index=False)

        df = load_prompt_file(prompt_file)
        assert "video_id" in df.columns
        assert "prompt" in df.columns
        assert len(df) == 2

    def test_csv_with_caption_alias(self, tmp_path):
        prompt_file = tmp_path / "prompts.csv"
        pd.DataFrame(
            {"uid": ["v1", "v2"], "caption": ["cat", "dog"]}
        ).to_csv(prompt_file, index=False)

        df = load_prompt_file(prompt_file)
        assert "video_id" in df.columns  # uid → video_id
        assert "prompt" in df.columns    # caption → prompt

    def test_jsonl_format(self, tmp_path):
        """JSONL file with prompt and video_id."""
        import json

        prompt_file = tmp_path / "prompts.jsonl"
        lines = [
            json.dumps({"video_id": "v1", "prompt": "a cat walking"}),
            json.dumps({"video_id": "v2", "prompt": "a dog running"}),
        ]
        prompt_file.write_text("\n".join(lines) + "\n")

        df = load_prompt_file(prompt_file)
        assert "video_id" in df.columns
        assert "prompt" in df.columns
        assert len(df) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prompt_file(tmp_path / "nonexistent.csv")

    def test_csv_with_path_alias(self, tmp_path):
        prompt_file = tmp_path / "prompts.csv"
        pd.DataFrame(
            {
                "path": ["/v1.mp4", "/v2.mp4"],
                "text": ["cat", "dog"],
            }
        ).to_csv(prompt_file, index=False)

        df = load_prompt_file(prompt_file)
        assert "video_path" in df.columns  # path → video_path
        assert "prompt" in df.columns      # text → prompt

    def test_csv_with_index_aliases(self, tmp_path):
        prompt_file = tmp_path / "prompts.csv"
        pd.DataFrame(
            {"idx": [0, 1], "prompt": ["cat", "dog"]}
        ).to_csv(prompt_file, index=False)

        df = load_prompt_file(prompt_file)
        assert "index" in df.columns  # idx → index


# ---------------------------------------------------------------------------
# series_value
# ---------------------------------------------------------------------------
class TestSeriesValue:
    def test_valid_key(self):
        row = pd.Series({"prompt": "a cat", "group": "g1"})
        assert series_value(row, "prompt") == "a cat"

    def test_missing_key(self):
        row = pd.Series({"prompt": "a cat"})
        assert series_value(row, "group") is None

    def test_nan_value(self):
        row = pd.Series({"prompt": float("nan")})
        assert series_value(row, "prompt") is None

    def test_none_row(self):
        assert series_value(None, "prompt") is None


# ---------------------------------------------------------------------------
# build_prompt_indices
# ---------------------------------------------------------------------------
class TestBuildPromptIndices:
    def test_with_video_id(self):
        df = pd.DataFrame({"video_id": ["v1", "v2"], "prompt": ["cat", "dog"]})
        by_id, by_stem, by_index = build_prompt_indices(df)
        assert "v1" in by_id
        assert "v2" in by_id

    def test_with_video_path(self):
        df = pd.DataFrame(
            {
                "video_path": ["/a/v1.mp4", "/b/v2.mp4"],
                "prompt": ["cat", "dog"],
            }
        )
        _, by_stem, _ = build_prompt_indices(df)
        assert "v1" in by_stem
        assert "v2" in by_stem

    def test_with_index_column(self):
        df = pd.DataFrame(
            {"index": [0, 1], "prompt": ["cat", "dog"]}
        )
        _, _, by_index = build_prompt_indices(df)
        assert 0 in by_index
        assert 1 in by_index

    def test_empty_df(self):
        df = pd.DataFrame(columns=["prompt"])
        by_id, by_stem, by_index = build_prompt_indices(df)
        assert len(by_id) == 0
        assert len(by_stem) == 0
        assert len(by_index) == 0


# ---------------------------------------------------------------------------
# find_prompt_row
# ---------------------------------------------------------------------------
class TestFindPromptRow:
    def test_find_by_video_id(self):
        df = pd.DataFrame({"video_id": ["v1", "v2"], "prompt": ["cat", "dog"]})
        by_id = {"v1": 0, "v2": 1}
        row = find_prompt_row(df, by_id, {}, {}, "v1", None, None, None)
        assert row is not None
        assert row["prompt"] == "cat"

    def test_find_by_stem(self):
        df = pd.DataFrame(
            {"video_path": ["/a/v1.mp4"], "prompt": ["cat"]}
        )
        by_stem = {"v1": 0}
        row = find_prompt_row(df, {}, by_stem, {}, "unknown", "/a/v1.mp4", None, None)
        assert row is not None
        assert row["prompt"] == "cat"

    def test_find_by_numeric_index(self):
        df = pd.DataFrame({"index": [42], "prompt": ["cat"]})
        by_index = {42: 0}
        row = find_prompt_row(df, {}, {}, by_index, "video_42", None, None, None)
        assert row is not None
        assert row["prompt"] == "cat"

    def test_find_by_positional(self):
        df = pd.DataFrame({"prompt": ["cat", "dog"]})
        by_pos = {0: 0, 1: 1}
        row = find_prompt_row(df, {}, {}, {}, "unknown", None, 0, by_pos)
        assert row is not None
        assert row["prompt"] == "cat"

    def test_returns_none_when_no_match(self):
        df = pd.DataFrame({"prompt": ["cat"]})
        row = find_prompt_row(df, {}, {}, {}, "nope", None, None, None)
        assert row is None

    def test_none_prompt_df(self):
        row = find_prompt_row(None, {}, {}, {}, "v1", None, None, None)
        assert row is None


# ---------------------------------------------------------------------------
# export_video_bytes
# ---------------------------------------------------------------------------
class TestExportVideoBytes:
    def test_bytes_data(self, tmp_path):
        output = tmp_path / "video.mp4"
        result = export_video_bytes(b"\x00\x01\x02", output)
        assert result is True
        assert output.read_bytes() == b"\x00\x01\x02"

    def test_dict_with_bytes(self, tmp_path):
        output = tmp_path / "video.mp4"
        result = export_video_bytes({"bytes": b"\xff\xfe"}, output)
        assert result is True
        assert output.read_bytes() == b"\xff\xfe"

    def test_dict_with_path(self, tmp_path):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"\xaa\xbb")
        output = tmp_path / "video.mp4"

        result = export_video_bytes({"path": str(source)}, output)
        assert result is True
        assert output.read_bytes() == b"\xaa\xbb"

    def test_string_path(self, tmp_path):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"\xcc\xdd")
        output = tmp_path / "video.mp4"

        result = export_video_bytes(str(source), output)
        assert result is True

    def test_unknown_format(self, tmp_path):
        output = tmp_path / "video.mp4"
        result = export_video_bytes(12345, output)
        assert result is False
