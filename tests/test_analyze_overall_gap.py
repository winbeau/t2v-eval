"""Tests for scripts/analyze_overall_gap.py."""

from pathlib import Path

import pandas as pd

from scripts.analyze_overall_gap import (
    collect_prompt_quality_by_group,
    compare_with_paper,
    is_video_like_prompt,
    parse_group_alias_map,
)


def test_is_video_like_prompt() -> None:
    assert is_video_like_prompt("video_000")
    assert is_video_like_prompt("g6_video_127")
    assert not is_video_like_prompt("A woman walking on beach")


def test_parse_group_alias_map(tmp_path: Path) -> None:
    log_path = tmp_path / "run_vbench.log"
    log_path.write_text(
        "INFO Group alias mapping for duplicate IDs: {'k1':'g1','k2':'g2'}\n",
        encoding="utf-8",
    )

    result = parse_group_alias_map(log_path)

    assert result == {"k1": "g1", "k2": "g2"}


def test_collect_prompt_quality_by_group(tmp_path: Path) -> None:
    full_info = tmp_path / "overall_consistency_full_info.json"
    full_info.write_text(
        """[
  {
    "prompt_en": "video_000",
    "video_list": ["/tmp/split_clip/g2_video_000/g2_video_000_000.mp4"]
  },
  {
    "prompt_en": "a cat running",
    "video_list": ["/tmp/split_clip/g1_video_000/g1_video_000_000.mp4"]
  }
]
""",
        encoding="utf-8",
    )

    alias_to_group = {"g1": "k1", "g2": "k2"}
    df = collect_prompt_quality_by_group(full_info, alias_to_group)

    assert set(df["group"].tolist()) == {"k1", "k2"}
    row_k2 = df[df["group"] == "k2"].iloc[0]
    row_k1 = df[df["group"] == "k1"].iloc[0]
    assert float(row_k2["video_like_ratio"]) == 1.0
    assert float(row_k1["video_like_ratio"]) == 0.0


def test_compare_with_paper_overall_consistency() -> None:
    summary_df = pd.DataFrame(
        {
            "group": ["k6_deep_forcing", "k7_native_self_forcing_static21_sink1"],
            "overall_consistency_mean": [4.7855, 5.2251],
        }
    )

    out = compare_with_paper(
        summary_df=summary_df,
        deep_group="k6_deep_forcing",
        self_group="k7_native_self_forcing_static21_sink1",
    )

    assert len(out) == 4
    assert set(out["model"].tolist()) == {"deep_forcing", "self_forcing"}
    deep_30 = out[(out["model"] == "deep_forcing") & (out["horizon"] == "30s")].iloc[0]
    assert abs(float(deep_30["abs_gap"]) - (4.7855 - 20.54)) < 1e-6
