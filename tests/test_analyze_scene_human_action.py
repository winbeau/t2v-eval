"""Tests for scripts/analyze_scene_human_action.py."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analyze_scene_human_action import load_vbench_per_video, main
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
