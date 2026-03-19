import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.vbench_runner.group_runs import (
    build_group_run_file_map,
    load_group_run_cache,
    main as merge_group_runs_main,
    write_group_run_cache,
)


@pytest.fixture
def group_run_config(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    config = {
        "groups": [
            {"name": "group_a"},
            {"name": "group_b"},
        ],
        "paths": {
            "output_dir": str(output_dir),
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config, config_path


def sample_group_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": ["video_000", "video_001", "video_000", "video_001"],
            "group": ["group_a", "group_a", "group_b", "group_b"],
            "vbench_temporal_score": [81.0, 82.0, 71.0, 72.0],
            "subject_consistency": [91.0, 92.0, 81.0, 82.0],
        }
    )


def test_build_group_run_file_map_is_deterministic(group_run_config):
    config, _ = group_run_config
    output_dir = Path(config["paths"]["output_dir"])

    mapping = build_group_run_file_map(config, output_dir)

    assert mapping["group_a"].name == "01__group_a.csv"
    assert mapping["group_b"].name == "02__group_b.csv"


def test_write_group_run_cache_requires_force_when_target_exists(group_run_config):
    config, _ = group_run_config
    output_dir = Path(config["paths"]["output_dir"])
    df = sample_group_results()

    write_group_run_cache(df, config=config, output_dir=output_dir, force=False)
    with pytest.raises(FileExistsError, match="Per-group VBench cache already exists"):
        write_group_run_cache(df[df["group"] == "group_a"], config=config, output_dir=output_dir, force=False)


def test_load_group_run_cache_rejects_missing_group(group_run_config):
    config, _ = group_run_config
    output_dir = Path(config["paths"]["output_dir"])
    df = sample_group_results()

    write_group_run_cache(df[df["group"] == "group_a"], config=config, output_dir=output_dir, force=False)

    with pytest.raises(ValueError, match="Missing groups: group_b"):
        load_group_run_cache(config=config, output_dir=output_dir)


def test_load_group_run_cache_rejects_unknown_csv(group_run_config):
    config, _ = group_run_config
    output_dir = Path(config["paths"]["output_dir"])
    df = sample_group_results()

    write_group_run_cache(df, config=config, output_dir=output_dir, force=False)
    extra = output_dir / "vbench_group_runs" / "unexpected.csv"
    extra.write_text("video_id,group\nx,group_x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected CSV files"):
        load_group_run_cache(config=config, output_dir=output_dir)


def test_load_group_run_cache_rejects_wrong_group_content(group_run_config):
    config, _ = group_run_config
    output_dir = Path(config["paths"]["output_dir"])
    mapping = build_group_run_file_map(config, output_dir)
    mapping["group_a"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "video_id": ["video_000"],
            "group": ["group_b"],
            "vbench_temporal_score": [80.0],
        }
    ).to_csv(mapping["group_a"], index=False)
    pd.DataFrame(
        {
            "video_id": ["video_000"],
            "group": ["group_b"],
            "vbench_temporal_score": [70.0],
        }
    ).to_csv(mapping["group_b"], index=False)

    with pytest.raises(ValueError, match="group mismatch"):
        load_group_run_cache(config=config, output_dir=output_dir)


def test_merge_group_runs_main_creates_outputs_and_frontend_copy(group_run_config, monkeypatch, tmp_path):
    config, config_path = group_run_config
    output_dir = Path(config["paths"]["output_dir"])
    df = sample_group_results()

    write_group_run_cache(df, config=config, output_dir=output_dir, force=False)
    monkeypatch.setattr(
        "scripts.vbench_runner.video_records.PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["group_runs.py", "--config", str(config_path), "--force"],
    )

    rc = merge_group_runs_main()
    assert rc == 0

    merged_df = pd.read_csv(output_dir / f"vbench_{config_path.stem}.csv")
    summary_df = pd.read_csv(output_dir / f"group_summary_vbench_{config_path.stem}.csv")
    assert set(merged_df["group"]) == {"group_a", "group_b"}
    assert set(summary_df["group"]) == {"group_a", "group_b"}

    frontend_dir = tmp_path / "frontend" / "public" / "data"
    manifest = json.loads((frontend_dir / "manifest.json").read_text(encoding="utf-8"))
    assert f"vbench_{config_path.stem}.csv" in manifest["files"]
    assert f"group_summary_vbench_{config_path.stem}.csv" in manifest["files"]


def test_merge_group_runs_shell_script_syntax():
    result = subprocess.run(
        ["bash", "-n", "scripts/merge_vbench_all.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
