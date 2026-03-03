# Deep-Forcing Overall Consistency Gap Root-Cause Analysis

## Executive Summary

- 结论主因：当前实验中 `overall_consistency` 低值主要来自 **prompt 语义退化**（多数样本 prompt 为 `video_000` 形式），而非聚合脚本误算。
- 关键证据：`overall_consistency` 与 `temporal_style` 在逐样本上完全一致（equal_ratio=1.0），且 `overall_consistency_full_info.json` 中仅 g1 组使用自然语言 prompt，其余 g2~g7 全为 video-id 风格 prompt。

## Current Config Context

| backend     | mode              | comparison_profile   |   subtasks |
|:------------|:------------------|:---------------------|-----------:|
| vbench_long | long_custom_input | deep_forcing_8d      |         16 |

## Paper vs Ours (Overall Consistency)

| model        | horizon   |   ours |   paper |   abs_gap |
|:-------------|:----------|-------:|--------:|----------:|
| deep_forcing | 30s       | 4.7855 |   20.54 |  -15.7545 |
| self_forcing | 30s       | 5.2251 |   20.5  |  -15.2749 |
| deep_forcing | 60s       | 4.7855 |   20.38 |  -15.5945 |
| self_forcing | 60s       | 5.2251 |   18.63 |  -13.4049 |

## Prompt Quality by Group (from overall_consistency_full_info.json)

| alias   | group                                    |   n_videos |   video_like_count |   video_like_ratio |   natural_prompt_ratio |
|:--------|:-----------------------------------------|-----------:|-------------------:|-------------------:|-----------------------:|
| g1      | k1_selfforcing_absolute21_baseline       |        128 |                  0 |                  0 |                      1 |
| g2      | k2_selfforcing_dynamic21_baseline        |        128 |                128 |                  1 |                      0 |
| g3      | k3_sta_posneg_osc_phase6_tail4           |        128 |                128 |                  1 |                      0 |
| g4      | k4_dynamic_sink1_af_quality              |        128 |                128 |                  1 |                      0 |
| g5      | k5_sta_posneg_osc_phase6_tail4_nodynrope |        128 |                128 |                  1 |                      0 |
| g6      | k6_deep_forcing                          |        128 |                128 |                  1 |                      0 |
| g7      | k7_native_self_forcing_static21_sink1    |        128 |                128 |                  1 |                      0 |

## Prompt Quality Bucket Effect

| prompt_quality_bucket   |   n_groups |   mean_overall |   mean_video_like_ratio |
|:------------------------|-----------:|---------------:|------------------------:|
| natural_prompt          |          1 |        23.3709 |                       0 |
| video_id_like_prompt    |          6 |         5.0145 |                       1 |

## Additional Diagnostics

- overall_consistency == temporal_style equal_ratio: `1.000000`
- group alias mapping (from log): `{'k1_selfforcing_absolute21_baseline': 'g1', 'k2_selfforcing_dynamic21_baseline': 'g2', 'k3_sta_posneg_osc_phase6_tail4': 'g3', 'k4_dynamic_sink1_af_quality': 'g4', 'k5_sta_posneg_osc_phase6_tail4_nodynrope': 'g5', 'k6_deep_forcing': 'g6', 'k7_native_self_forcing_static21_sink1': 'g7'}`

## Root-Cause Ranking

1. **Prompt mapping only covers one 128-row prompt file**: run log shows a single prompt file loaded with 128 rows, while total videos are 896.
2. **Most groups receive fallback prompt tokens (`video_000` etc.)**: prompt-quality table shows g2~g7 are 100% video-id-like prompts.
3. **Upstream dimension behavior overlap**: overall and temporal_style are effectively identical under current setup, so prompt degradation directly drags both down.
4. **Not a simple scaling bug**: k1 group has natural prompts and overall≈23.37 (already near paper 18~21 range), while degraded-prompt groups collapse to ≈4.6~5.6.

## Actionable Next Checks

1. 为每个组提供正确的 prompt 映射（或合并成 896 行统一 prompt 文件），再重跑 8维/12维。
2. 重跑后优先比较 K6/K7 的 overall_consistency 是否回升到 18~21 区间。
3. 若仍有差距，再切换到论文同口径 backend/mode 做 A/B（vbench_long vs vbench）。
