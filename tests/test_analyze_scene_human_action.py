"""Tests for scripts/analyze_scene_human_action.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analyze_scene_human_action import (
    build_variance_summary,
    filter_debug_rows,
    load_vbench_per_video,
    main,
)
from scripts.vbench_runner.auxiliary import explain_scene_from_prompt
from scripts.vbench_runner.compat import match_human_action_prompt


def test_explain_scene_from_prompt_modes() -> None:
    assert explain_scene_from_prompt("A beach with waves") == {
        "label": "beach",
        "mode": "keyword",
    }
    preposition = explain_scene_from_prompt("A cat near violin")
    assert preposition["mode"] == "preposition"
    assert preposition["label"]
    assert explain_scene_from_prompt("A spinning globe") == {
        "label": "globe",
        "mode": "fallback",
    }


def test_match_human_action_prompt_modes() -> None:
    categories = ["playing guitar", "walking the dog", "riding bicycle"]

    exact = match_human_action_prompt(
        "A musician is playing guitar on stage",
        categories=categories,
    )
    assert exact["label"] == "playing guitar"
    assert exact["mode"] == "exact"

    keyword = match_human_action_prompt(
        "A woman is walking dog in a park",
        categories=categories,
    )
    assert keyword["label"] == "walking the dog"
    assert keyword["mode"] == "keyword"

    unmatched = match_human_action_prompt(
        "A cat sleeping on a sofa",
        categories=categories,
    )
    assert unmatched["label"] is None
    assert unmatched["mode"] == "unmatched"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_run(output_dir: Path, group_suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "video_id": f"g1_video_000",
                "group": f"deep-forcing-{group_suffix}",
                "scene": 90.0,
                "human_action": 100.0,
            },
            {
                "video_id": f"g2_video_000",
                "group": f"pyramid-forcing-{group_suffix}",
                "scene": 60.0,
                "human_action": 0.0,
            },
        ]
    ).to_csv(output_dir / "vbench_per_video.csv", index=False)

    _write_json(
        output_dir / "vbench_results" / "scene_full_info.json",
        [
            {
                "prompt_en": "A surfer on the beach",
                "video_list": [f"/tmp/split_clip/g1_video_000-Scene-001/clip_a.mp4"],
            },
            {
                "prompt_en": "A cat in a shed",
                "video_list": [f"/tmp/split_clip/g2_video_000-Scene-001/clip_b.mp4"],
            },
        ],
    )
    _write_json(
        output_dir / "vbench_results" / "human_action_full_info.json",
        [
            {
                "prompt_en": "A woman playing guitar on stage",
                "video_list": [f"/tmp/split_clip/g1_video_000-Scene-001/clip_a.mp4"],
            },
            {
                "prompt_en": "A cat sleeping on a sofa",
                "video_list": [f"/tmp/split_clip/g2_video_000-Scene-001/clip_b.mp4"],
            },
        ],
    )
    _write_json(
        output_dir / "vbench_results" / "scene_eval_results.json",
        {
            "scene": [
                {
                    "video_path": f"/tmp/split_clip/g1_video_000-Scene-001/clip_a.mp4",
                    "video_results": 0.9,
                },
                {
                    "video_path": f"/tmp/split_clip/g2_video_000-Scene-001/clip_b.mp4",
                    "video_results": 0.6,
                },
            ]
        },
    )
    _write_json(
        output_dir / "vbench_results" / "human_action_eval_results.json",
        {
            "human_action": [
                {
                    "video_path": f"/tmp/split_clip/g1_video_000-Scene-001/clip_a.mp4",
                    "video_results": 1,
                },
                {
                    "video_path": f"/tmp/split_clip/g2_video_000-Scene-001/clip_b.mp4",
                    "video_results": 0,
                },
            ]
        },
    )
    (output_dir / "run_vbench.log").write_text(
        "human_action K-400 matching: exact=1 keyword=0 unmatched=1\n",
        encoding="utf-8",
    )


def test_main_generates_report_and_csvs(tmp_path, monkeypatch) -> None:
    run_30s = tmp_path / "outputs" / "prompts128-30s"
    run_60s = tmp_path / "outputs" / "prompts128-60s"
    _make_run(run_30s, "30s")
    _make_run(run_60s, "60s")

    monkeypatch.setattr(
        "scripts.analyze_scene_human_action.load_human_action_categories",
        lambda: ["playing guitar", "walking the dog", "riding bicycle"],
    )

    artifacts_dir = tmp_path / "artifacts"
    rc = main(
        [
            "--output-dir",
            str(run_30s),
            "--output-dir",
            str(run_60s),
            "--artifacts-dir",
            str(artifacts_dir),
        ]
    )

    assert rc == 0
    assert (artifacts_dir / "report.md").exists()
    assert (artifacts_dir / "per_video_diagnostics.csv").exists()
    assert (artifacts_dir / "group_dimension_summary.csv").exists()
    assert (artifacts_dir / "variance_distribution_summary.csv").exists()
    assert (artifacts_dir / "scene_suspicious_top20.csv").exists()
    assert (artifacts_dir / "human_action_suspicious_top20.csv").exists()

    diag_df = pd.read_csv(artifacts_dir / "per_video_diagnostics.csv")
    assert {"scene_mode", "scene_label", "human_action_mode", "human_action_label"}.issubset(
        diag_df.columns
    )
    assert set(diag_df["human_action_mode"]) == {"exact", "unmatched"}


def test_load_vbench_per_video_falls_back_to_named_csv(tmp_path) -> None:
    output_dir = tmp_path / "outputs" / "prompts128-30s"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "video_id": "g1_video_000",
                "group": "deep-forcing-30s",
                "scene": 80.0,
                "human_action": 90.0,
            }
        ]
    ).to_csv(output_dir / "vbench_prompts128-30s.csv", index=False)

    df = load_vbench_per_video(output_dir)

    assert list(df["video_id"]) == ["g1_video_000"]


def test_filter_debug_rows_only_drops_unmatched() -> None:
    df = pd.DataFrame(
        [
            {"video_id": "a", "human_action_mode": "exact", "group": "g", "human_action": 80.0},
            {"video_id": "b", "human_action_mode": "keyword", "group": "g", "human_action": 50.0},
            {"video_id": "c", "human_action_mode": "unmatched", "group": "g", "human_action": 100.0},
        ]
    )

    kept, removed = filter_debug_rows(df, "debug1")

    assert list(kept["video_id"]) == ["a", "b"]
    assert list(removed["video_id"]) == ["c"]
    assert list(removed["removed_reason"]) == ["human_action_unmatched"]


def test_build_variance_summary_reports_bimodal_bins() -> None:
    df = pd.DataFrame(
        [
            {"run": "prompts128-60s", "group": "deep-forcing-60s", "scene": 0.0, "human_action": 0.0},
            {"run": "prompts128-60s", "group": "deep-forcing-60s", "scene": 100.0, "human_action": 100.0},
            {"run": "prompts128-60s", "group": "deep-forcing-60s", "scene": 95.0, "human_action": 0.0},
        ]
    )

    summary = build_variance_summary(df)

    scene_row = summary[summary["metric"] == "scene"].iloc[0]
    action_row = summary[summary["metric"] == "human_action"].iloc[0]

    assert scene_row["eq0_count"] == 1
    assert scene_row["bin_95_100_count"] == 2
    assert round(scene_row["zero_ratio"], 4) == round(1 / 3, 4)
    assert round(action_row["zero_ratio"], 4) == round(2 / 3, 4)


def test_main_exports_debug_csvs_and_updates_frontend(tmp_path, monkeypatch) -> None:
    run_30s = tmp_path / "outputs" / "prompts128-30s"
    run_60s = tmp_path / "outputs" / "prompts128-60s"
    _make_run(run_30s, "30s")
    _make_run(run_60s, "60s")

    monkeypatch.setattr(
        "scripts.analyze_scene_human_action.load_human_action_categories",
        lambda: ["playing guitar", "walking the dog", "riding bicycle"],
    )

    artifacts_dir = tmp_path / "artifacts"
    frontend_dir = tmp_path / "frontend" / "public" / "data"
    rc = main(
        [
            "--output-dir",
            str(run_30s),
            "--output-dir",
            str(run_60s),
            "--artifacts-dir",
            str(artifacts_dir),
            "--emit-debug-filter",
            "debug1",
            "--copy-to-frontend",
            "--frontend-data-dir",
            str(frontend_dir),
        ]
    )

    assert rc == 0

    per_video_30s = run_30s / "vbench_prompts128-30s-debug1.csv"
    summary_30s = run_30s / "prompts128-30s-debug1.csv"
    removed_30s = run_30s / "prompts128-30s-debug1_removed.csv"
    per_video_60s = run_60s / "vbench_prompts128-60s-debug1.csv"
    summary_60s = run_60s / "prompts128-60s-debug1.csv"
    removed_60s = run_60s / "prompts128-60s-debug1_removed.csv"

    for path in [per_video_30s, summary_30s, removed_30s, per_video_60s, summary_60s, removed_60s]:
        assert path.exists()
        assert (frontend_dir / path.name).exists()

    filtered_30s = pd.read_csv(per_video_30s)
    filtered_60s = pd.read_csv(per_video_60s)
    assert list(filtered_30s["video_id"]) == ["g1_video_000"]
    assert list(filtered_60s["video_id"]) == ["g1_video_000"]

    summary_df = pd.read_csv(summary_60s)
    assert {"group", "n_videos", "scene_mean", "scene_std", "human_action_mean", "human_action_std"}.issubset(
        summary_df.columns
    )
    assert summary_df["n_videos"].tolist() == [1]

    removed_df = pd.read_csv(removed_60s)
    assert removed_df["removed_reason"].tolist() == ["human_action_unmatched"]

    manifest = json.loads((frontend_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "vbench_prompts128-60s-debug1.csv" in manifest["files"]
    assert "prompts128-60s-debug1.csv" in manifest["files"]
