"""
Tests for scripts/vbench_runner/video_records.py — video metadata management.

Covers:
  - infer_group_from_path()
  - ensure_unique_video_ids()
  - load_prompt_map()
  - are_split_clips_ready()
  - get_input_video_files()
"""

from pathlib import Path

import pandas as pd

from scripts.vbench_runner.video_records import (
    are_split_clips_ready,
    build_video_list_from_local_dataset,
    ensure_unique_video_ids,
    get_input_video_files,
    infer_group_from_path,
    load_prompt_map,
)


# ---------------------------------------------------------------------------
# infer_group_from_path
# ---------------------------------------------------------------------------
class TestInferGroupFromPath:
    def test_group_in_folder(self):
        path = Path("/data/group_a/video.mp4")
        result = infer_group_from_path(path, ["group_a", "group_b"])
        assert result == "group_a"

    def test_group_in_filename(self):
        path = Path("/data/videos/group_b_001.mp4")
        result = infer_group_from_path(path, ["group_a", "group_b"])
        assert result == "group_b"

    def test_fallback_parent_name(self):
        path = Path("/data/unknown_group/video.mp4")
        result = infer_group_from_path(path, [])
        assert result == "unknown_group"

    def test_no_parent(self):
        path = Path("video.mp4")
        result = infer_group_from_path(path, [])
        assert result is None

    def test_no_match_with_groups(self):
        path = Path("/data/unknown/video.mp4")
        result = infer_group_from_path(path, ["group_a"])
        # Falls back to parent name since groups list doesn't match
        assert result == "unknown"


# ---------------------------------------------------------------------------
# ensure_unique_video_ids
# ---------------------------------------------------------------------------
class TestEnsureUniqueVideoIds:
    def test_already_unique(self):
        records = [
            {"video_id": "v1", "group": "g1", "video_path": "/a/v1.mp4"},
            {"video_id": "v2", "group": "g1", "video_path": "/a/v2.mp4"},
        ]
        config = {"groups": [{"name": "g1"}]}
        result = ensure_unique_video_ids(records, config)
        ids = [r["video_id"] for r in result]
        assert ids == ["v1", "v2"]

    def test_duplicates_remapped(self):
        records = [
            {"video_id": "vid_000", "group": "g1", "video_path": "/g1/vid_000.mp4"},
            {"video_id": "vid_000", "group": "g2", "video_path": "/g2/vid_000.mp4"},
        ]
        config = {"groups": [{"name": "g1"}, {"name": "g2"}]}
        result = ensure_unique_video_ids(records, config)
        ids = [r["video_id"] for r in result]
        assert len(set(ids)) == 2  # All unique
        assert ids[0] != ids[1]

    def test_empty_records(self):
        result = ensure_unique_video_ids([], {"groups": []})
        assert result == []

    def test_missing_video_id_inferred(self):
        records = [
            {"video_id": "", "group": "g1", "video_path": "/a/test_video.mp4"},
        ]
        config = {"groups": [{"name": "g1"}]}
        result = ensure_unique_video_ids(records, config)
        assert result[0]["video_id"] == "test_video"

    def test_triple_duplicates(self):
        records = [
            {"video_id": "v", "group": "g1", "video_path": "/g1/v.mp4"},
            {"video_id": "v", "group": "g2", "video_path": "/g2/v.mp4"},
            {"video_id": "v", "group": "g3", "video_path": "/g3/v.mp4"},
        ]
        config = {"groups": [{"name": "g1"}, {"name": "g2"}, {"name": "g3"}]}
        result = ensure_unique_video_ids(records, config)
        ids = [r["video_id"] for r in result]
        assert len(set(ids)) == 3

    def test_preserves_non_duplicated(self):
        """Non-duplicated IDs should remain unchanged."""
        records = [
            {"video_id": "unique_a", "group": "g1", "video_path": "/g1/unique_a.mp4"},
            {"video_id": "common", "group": "g1", "video_path": "/g1/common.mp4"},
            {"video_id": "common", "group": "g2", "video_path": "/g2/common.mp4"},
        ]
        config = {"groups": [{"name": "g1"}, {"name": "g2"}]}
        result = ensure_unique_video_ids(records, config)
        # unique_a should be unchanged
        assert result[0]["video_id"] == "unique_a"
        # common should be remapped
        ids = [r["video_id"] for r in result]
        assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# load_prompt_map
# ---------------------------------------------------------------------------
class TestLoadPromptMap:
    def test_basic_csv(self, tmp_path):
        csv = tmp_path / "prompts.csv"
        pd.DataFrame(
            {"video_id": ["v1", "v2"], "prompt": ["cat", "dog"]}
        ).to_csv(csv, index=False)

        by_id, by_stem, ordered = load_prompt_map(csv)
        assert by_id == {"v1": "cat", "v2": "dog"}
        assert ordered == ["cat", "dog"]

    def test_caption_alias(self, tmp_path):
        csv = tmp_path / "prompts.csv"
        pd.DataFrame(
            {"uid": ["v1"], "caption": ["a cat walking"]}
        ).to_csv(csv, index=False)

        by_id, _, ordered = load_prompt_map(csv)
        assert "v1" in by_id
        assert ordered == ["a cat walking"]

    def test_path_based_stem_lookup(self, tmp_path):
        csv = tmp_path / "prompts.csv"
        pd.DataFrame(
            {
                "video_path": ["/a/my_video.mp4"],
                "prompt": ["test prompt"],
            }
        ).to_csv(csv, index=False)

        _, by_stem, _ = load_prompt_map(csv)
        assert "my_video" in by_stem

    def test_single_column_fallback(self, tmp_path):
        """When only one column exists, it should become 'prompt'."""
        csv = tmp_path / "prompts.csv"
        csv.write_text("descriptions\ncat walking\ndog running\n")

        _, _, ordered = load_prompt_map(csv)
        assert len(ordered) == 2


# ---------------------------------------------------------------------------
# build_video_list_from_local_dataset
# ---------------------------------------------------------------------------
class TestBuildVideoListFromLocalDataset:
    def test_group_prompt_files_take_precedence_over_global(self, tmp_path):
        local_dir = tmp_path / "videos"
        (local_dir / "group_a").mkdir(parents=True)
        (local_dir / "group_b").mkdir(parents=True)
        (local_dir / "group_a" / "video_000.mp4").write_text("")
        (local_dir / "group_b" / "video_000.mp4").write_text("")

        global_prompt = tmp_path / "global_prompts.csv"
        pd.DataFrame({"video_id": ["video_000"], "prompt": ["global prompt"]}).to_csv(
            global_prompt, index=False
        )
        group_a_prompt = tmp_path / "group_a_prompts.csv"
        pd.DataFrame({"video_id": ["video_000"], "prompt": ["group a prompt"]}).to_csv(
            group_a_prompt, index=False
        )
        group_b_prompt = tmp_path / "group_b_prompts.csv"
        pd.DataFrame({"video_id": ["video_000"], "prompt": ["group b prompt"]}).to_csv(
            group_b_prompt, index=False
        )

        config = {
            "dataset": {
                "local_video_dir": str(local_dir),
                "prompt_file": str(global_prompt),
                "prompt_files_by_group": {
                    "group_a": str(group_a_prompt),
                    "group_b": str(group_b_prompt),
                },
            },
            "groups": [{"name": "group_a"}, {"name": "group_b"}],
        }
        records = build_video_list_from_local_dataset(config)

        by_group = {r["group"]: r["prompt"] for r in records}
        assert by_group["group_a"] == "group a prompt"
        assert by_group["group_b"] == "group b prompt"

    def test_group_positional_fallback_uses_group_local_index(self, tmp_path):
        local_dir = tmp_path / "videos"
        for group in ["group_a", "group_b"]:
            group_dir = local_dir / group
            group_dir.mkdir(parents=True)
            (group_dir / "video_000.mp4").write_text("")
            (group_dir / "video_001.mp4").write_text("")

        group_a_prompt = tmp_path / "group_a_prompts.csv"
        group_b_prompt = tmp_path / "group_b_prompts.csv"
        pd.DataFrame({"prompt": ["group_a_0", "group_a_1"]}).to_csv(group_a_prompt, index=False)
        pd.DataFrame({"prompt": ["group_b_0", "group_b_1"]}).to_csv(group_b_prompt, index=False)

        config = {
            "dataset": {
                "local_video_dir": str(local_dir),
                "prompt_files_by_group": {
                    "group_a": str(group_a_prompt),
                    "group_b": str(group_b_prompt),
                },
            },
            "groups": [{"name": "group_a"}, {"name": "group_b"}],
        }
        records = build_video_list_from_local_dataset(config)
        prompt_map = {(r["group"], r["video_id"]): r["prompt"] for r in records}

        assert prompt_map[("group_a", "video_000")] == "group_a_0"
        assert prompt_map[("group_a", "video_001")] == "group_a_1"
        assert prompt_map[("group_b", "video_000")] == "group_b_0"
        assert prompt_map[("group_b", "video_001")] == "group_b_1"

    def test_fallback_chain_group_to_global_to_video_id(self, tmp_path):
        local_dir = tmp_path / "videos"
        (local_dir / "group_a").mkdir(parents=True)
        (local_dir / "group_b").mkdir(parents=True)
        (local_dir / "group_a" / "video_000.mp4").write_text("")
        (local_dir / "group_b" / "video_001.mp4").write_text("")

        global_prompt = tmp_path / "global_prompts.csv"
        pd.DataFrame({"video_id": ["video_000"], "prompt": ["global prompt"]}).to_csv(
            global_prompt, index=False
        )

        config = {
            "dataset": {
                "local_video_dir": str(local_dir),
                "prompt_file": str(global_prompt),
                "prompt_files_by_group": {
                    "group_b": str(tmp_path / "missing_group_b_prompts.csv"),
                },
            },
            "groups": [{"name": "group_a"}, {"name": "group_b"}],
        }
        records = build_video_list_from_local_dataset(config)
        prompt_map = {(r["group"], r["video_id"]): r["prompt"] for r in records}

        assert prompt_map[("group_a", "video_000")] == "global prompt"
        assert prompt_map[("group_b", "video_001")] == "video_001"


# ---------------------------------------------------------------------------
# get_input_video_files
# ---------------------------------------------------------------------------
class TestGetInputVideoFiles:
    def test_finds_mp4_files(self, tmp_path):
        (tmp_path / "a.mp4").write_text("")
        (tmp_path / "b.mp4").write_text("")
        (tmp_path / "c.txt").write_text("")  # non-video
        result = get_input_video_files(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".mp4" for p in result)

    def test_finds_avi_and_mov(self, tmp_path):
        (tmp_path / "a.avi").write_text("")
        (tmp_path / "b.mov").write_text("")
        result = get_input_video_files(tmp_path)
        assert len(result) == 2

    def test_excludes_directories(self, tmp_path):
        (tmp_path / "subdir.mp4").mkdir()  # directory named .mp4
        (tmp_path / "real.mp4").write_text("")
        result = get_input_video_files(tmp_path)
        assert len(result) == 1

    def test_sorted_output(self, tmp_path):
        (tmp_path / "c.mp4").write_text("")
        (tmp_path / "a.mp4").write_text("")
        (tmp_path / "b.mp4").write_text("")
        result = get_input_video_files(tmp_path)
        names = [p.name for p in result]
        assert names == ["a.mp4", "b.mp4", "c.mp4"]

    def test_empty_dir(self, tmp_path):
        result = get_input_video_files(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# are_split_clips_ready
# ---------------------------------------------------------------------------
class TestAreSplitClipsReady:
    def test_ready(self, tmp_path):
        video_dir = tmp_path / "input"
        video_dir.mkdir()
        v1 = video_dir / "vid_001.mp4"
        v1.write_text("")

        split_dir = video_dir / "split_clip" / "vid_001"
        split_dir.mkdir(parents=True)
        (split_dir / "clip_001.mp4").write_text("")

        assert are_split_clips_ready(video_dir, [v1]) is True

    def test_no_split_dir(self, tmp_path):
        video_dir = tmp_path / "input"
        video_dir.mkdir()
        v1 = video_dir / "vid_001.mp4"
        v1.write_text("")
        assert are_split_clips_ready(video_dir, [v1]) is False

    def test_missing_clip_folder(self, tmp_path):
        video_dir = tmp_path / "input"
        video_dir.mkdir()
        v1 = video_dir / "vid_001.mp4"
        v1.write_text("")

        split_dir = video_dir / "split_clip"
        split_dir.mkdir()
        # No vid_001 subfolder
        assert are_split_clips_ready(video_dir, [v1]) is False

    def test_empty_clip_folder(self, tmp_path):
        video_dir = tmp_path / "input"
        video_dir.mkdir()
        v1 = video_dir / "vid_001.mp4"
        v1.write_text("")

        clip_dir = video_dir / "split_clip" / "vid_001"
        clip_dir.mkdir(parents=True)
        # Folder exists but no clips inside
        assert are_split_clips_ready(video_dir, [v1]) is False

    def test_empty_input_list(self, tmp_path):
        video_dir = tmp_path / "input"
        video_dir.mkdir()
        (video_dir / "split_clip").mkdir()
        assert are_split_clips_ready(video_dir, []) is False
