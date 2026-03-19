import argparse
from pathlib import Path

import pytest

from scripts.vbench_runner.core import _resolve_vbench_output_path
from scripts.vbench_runner.group_subset import (
    filter_records_to_groups,
    resolve_effective_group_subset,
    validate_output_file_name,
)


@pytest.fixture
def sample_group_config():
    return {
        "groups": [
            {"name": "g1"},
            {"name": "g2"},
            {"name": "g3"},
        ]
    }


def test_resolve_effective_group_subset_filters_config(sample_group_config):
    effective_names, effective_config = resolve_effective_group_subset(sample_group_config, "g2")

    assert effective_names == ["g1", "g3"]
    assert [item["name"] for item in effective_config["groups"]] == ["g1", "g3"]
    assert [item["name"] for item in sample_group_config["groups"]] == ["g1", "g2", "g3"]


def test_resolve_effective_group_subset_rejects_unknown(sample_group_config):
    with pytest.raises(ValueError, match="Unknown groups"):
        resolve_effective_group_subset(sample_group_config, "missing")


def test_resolve_effective_group_subset_rejects_empty_tokens(sample_group_config):
    with pytest.raises(ValueError, match="did not contain any valid group names"):
        resolve_effective_group_subset(sample_group_config, " , , ")


def test_filter_records_to_groups_keeps_only_allowed():
    records = [
        {"video_id": "v1", "group": "g1"},
        {"video_id": "v2", "group": "g2"},
        {"video_id": "v3", "group": "g3"},
    ]
    filtered = filter_records_to_groups(records, ["g1", "g3"])

    assert [record["group"] for record in filtered] == ["g1", "g3"]


def test_validate_output_file_name_rejects_paths():
    with pytest.raises(ValueError, match="file name only"):
        validate_output_file_name("nested/out.csv", arg_name="--vbench-output")


def test_resolve_vbench_output_path_requires_explicit_name_for_subset(tmp_path):
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(vbench_output="")

    with pytest.raises(SystemExit, match="2"):
        _resolve_vbench_output_path(
            args=args,
            parser=parser,
            output_dir=tmp_path,
            config_stem="demo",
            subset_active=True,
        )


def test_resolve_vbench_output_path_uses_explicit_name(tmp_path):
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(vbench_output="subset.csv")

    result = _resolve_vbench_output_path(
        args=args,
        parser=parser,
        output_dir=tmp_path,
        config_stem="demo",
        subset_active=True,
    )

    assert result == Path(tmp_path) / "subset.csv"
