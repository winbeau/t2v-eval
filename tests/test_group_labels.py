"""
Tests for scripts/vbench_runner/group_labels.py.
"""

import pandas as pd

from scripts.vbench_runner.group_labels import build_group_alias_map, remap_group_column


class TestBuildGroupAliasMap:
    def test_builds_alias_map_from_groups(self):
        config = {
            "groups": [
                {"name": "k1", "alias": "K1 Baseline"},
                {"name": "k2", "alias": "K2 Variant"},
            ]
        }
        result = build_group_alias_map(config)
        assert result == {"k1": "K1 Baseline", "k2": "K2 Variant"}

    def test_ignores_missing_or_empty_alias(self):
        config = {
            "groups": [
                {"name": "k1"},
                {"name": "k2", "alias": ""},
                {"name": "k3", "alias": "  "},
                {"name": "k4", "alias": "K4"},
            ]
        }
        result = build_group_alias_map(config)
        assert result == {"k4": "K4"}


class TestRemapGroupColumn:
    def test_remaps_group_column(self):
        df = pd.DataFrame(
            {
                "video_id": ["v1", "v2", "v3"],
                "group": ["k1", "k2", "k3"],
            }
        )
        mapped = remap_group_column(df, {"k1": "K1", "k2": "K2"})
        assert list(mapped["group"]) == ["K1", "K2", "k3"]
        # Original df should stay unchanged
        assert list(df["group"]) == ["k1", "k2", "k3"]

    def test_noop_when_group_column_missing(self):
        df = pd.DataFrame({"video_id": ["v1"]})
        mapped = remap_group_column(df, {"k1": "K1"})
        assert mapped.equals(df)

    def test_preserves_nan_values(self):
        df = pd.DataFrame({"group": ["k1", None]})
        mapped = remap_group_column(df, {"k1": "K1"})
        assert mapped.loc[0, "group"] == "K1"
        assert pd.isna(mapped.loc[1, "group"])
