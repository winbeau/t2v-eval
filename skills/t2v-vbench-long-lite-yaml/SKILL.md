---
name: t2v-vbench-long-lite-yaml
description: Generate runnable `t2v-eval` VBench-Long Lite YAML configs (`configs/*.yaml`) from a pasted directory tree or local experiment layout. Use when the user asks for VBench-only, VBench-Long Lite, 12D Lite, long_12, 30s/60s/long-video temporal configs, or wants to create configs from `sutree`/directory-tree input while supporting different video lengths.
---

# T2V VBench-Long Lite YAML SOP

## Goal

Create repository-conformant `configs/*.yaml` files for local video experiments that run only VBench-Long Lite 12D evaluation. Use the directory tree to infer the experiment root and groups, then infer or apply the requested frame-length profile.

## SOP

1. Read nearby configs before writing when possible, especially:
   - `configs/strategy-ablation.yaml`
   - `configs/causal-forcing-rerun-30s.yaml`
   - `configs/prompts128-60s.yaml`
   - `configs/four-forcing-H200.yaml`
2. Parse the input tree:
   - Use the tree root as `dataset.local_video_dir`.
   - Use first-level child directories as `groups[*].name`.
   - Preserve the tree order unless the user asks to sort.
   - If the root itself is one group and contains `prompts.csv`, use the root basename as the single group.
3. Choose the config filename:
   - Default to `configs/<experiment-name>.yaml`, where `<experiment-name>` is the basename of the root path.
   - If the user names a config file, use that exact filename.
4. Infer frame profile from explicit user text first, then group/root names, then matching repository examples.
5. Build a VBench-only YAML; do not emit `clip_or_vqa`, `flicker`, or `niqe` blocks.
6. Validate YAML parseability before finishing.
7. If the local video directory exists, optionally count videos per group without running evaluation.
8. Report the run command: `uv run python scripts/run_vbench.py --config configs/<experiment-name>.yaml`.

## Frame Profiles

Use explicit user-provided `latent_frames`, `actual_frames`, or `num_frames` when present. Otherwise use these repository conventions:

| Signal | latent_frames | actual_frames | protocol.num_frames |
| --- | ---: | ---: | ---: |
| `21`, short baseline | 21 | 84 | 50 |
| `72` | 72 | 288 | 173 |
| `H200`, `120/480`, explicit four-forcing style | 120 | 480 | 288 |
| `30s`, recent 30-second local experiments | 123 | 492 | 295 |
| `60s`, recent 60-second local experiments | 243 | 972 | 583 |

When a new latent length is explicit but `num_frames` is not, set `actual_frames = latent_frames * 4` and `protocol.num_frames = round(actual_frames * 0.6)`. Add a short YAML comment stating the assumption.

For mixed-length groups, set each group’s `latent_frames` and `actual_frames` individually. Use one shared `protocol.num_frames`; prefer the user-specified value, otherwise use the smallest inferred `num_frames` and add a comment about the mixed-length assumption.

## Dataset Mapping

Use these defaults unless the user overrides them:

- `dataset.repo_id`: `"kv-compression/AdaHead"` even when `use_local_videos: true`
- `dataset.split`: `"train"`
- `dataset.use_local_videos`: `true`
- `dataset.video_dir`: `"videos/<experiment-name>"`
- `runtime`: `device: "cuda"`, `batch_size: 1`, `num_workers: 4`, `seed: 42`

For per-group directories, write:

```yaml
prompt_files_by_group:
  "<group>": "<root>/<group>/prompts.csv"
prompt_file: "<root>/<first-group>/prompts.csv"
```

For a single-group root with `prompts.csv` at the root, write:

```yaml
prompt_files_by_group:
  "<group>": "<root>/prompts.csv"
prompt_file: "<root>/prompts.csv"
```

Keep the fallback `prompt_file` because current pipeline entry points still expect one. If prompts may differ by group, add a concise comment that the fallback is only for compatibility and per-group files are declared in `prompt_files_by_group`.

## YAML Shape

Emit top-level sections in this order:

1. `dataset`
2. `groups`
3. `group_categories` only when analytically meaningful
4. `protocol`
5. `metrics`
6. `runtime`
7. `paths`
8. `logging`

Use 2-space indentation. Use double quotes for strings. Keep numbers and booleans unquoted. Use `frame_sampling: "uniform"` and `frame_padding: "loop"` unless the user overrides them.

## VBench-Long Lite Block

Always use this metric shape for this skill:

```yaml
metrics:
  enabled:
    - "vbench_temporal"

  vbench:
    enabled: true
    backend: "vbench_long"
    use_long: true
    strict_integrity: true
    require_full_coverage: true
    group_mismatch_warn_only: false
    preprocess_workers: 48
    slow_dims_fused: true
    slow_dims_decode_total_workers: 48
    slow_dims_decode_prefetch: 24
    slow_dims_decode_backend: "decord"
    slow_dims_shard_mode: "clip"
    slow_dims_perf_window_clips: 5
    slow_dims_stage_profile: true
    grit_batch_parallel_enable: false
    grit_batch_size: 1
    grit_batch_adaptive_fallback_enable: false
    appearance_style_batch_enable: true
    appearance_style_batch_size: 16
    dynamic_degree_batch_enable: true
    dynamic_degree_pair_batch_size: 8
    batch_accel_fp16_enable: false
    batch_accel_strict_fp32: true
    scale_to_percent: []
    profile_output: "group_summary_lite12d.csv"
    dimension_profile: "long_12"
    mode: "long_custom_input"
    use_semantic_splitting: false
    clip_length_config: "clip_length_mix.yaml"
    static_filter_flag: false
    scene_threshold: 35.0
    sb_clip2clip_feat_extractor: "dino"
    bg_clip2clip_feat_extractor: "clip"
    subtasks:
      - "subject_consistency"
      - "background_consistency"
      - "temporal_flickering"
      - "motion_smoothness"
      - "temporal_style"
      - "appearance_style"
      - "scene"
      - "human_action"
      - "overall_consistency"
      - "dynamic_degree"
      - "imaging_quality"
      - "aesthetic_quality"
```

## Paths

Use experiment-specific outputs to avoid overwrites:

```yaml
paths:
  cache_dir: "./eval_cache"
  output_dir: "./outputs/<experiment-name>"
  metadata_file: "metadata.csv"
  processed_metadata: "processed_metadata.csv"
  per_video_metrics: "per_video_metrics.csv"
  group_summary: "group_summary.csv"
  runtime_csv: "runtime.csv"
  figures_dir: "./outputs/<experiment-name>/figs"
  experiment_output: "<experiment-name>.csv"

logging:
  level: "INFO"
  log_file: "./outputs/<experiment-name>/eval.log"
  console: true
```

## Validation

After writing the config, run:

```bash
uv run python -c 'import pathlib, yaml; p = pathlib.Path("configs/<experiment-name>.yaml"); yaml.safe_load(p.read_text(encoding="utf-8")); print("YAML_OK")'
```

If the local root exists, count videos without starting evaluation:

```bash
find <root>/<group> -maxdepth 1 -name "*.mp4" | wc -l
```

Do not run the full VBench evaluation unless the user explicitly asks.

## Example Trigger Phrases

- “根据这个 sutree 写 VBench-Long Lite yaml”
- “给这个 30s/60s 本地视频目录生成 long_12 配置”
- “仿照 addition-30s 的 SOP 生成配置”
- “只跑 VBench，不要 CLIP/VQA/NIQE”
- “按 12维 Lite / VBench-only 写 configs”
