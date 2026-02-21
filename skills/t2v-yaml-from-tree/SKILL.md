---
name: t2v-yaml-from-tree
description: Generate runnable `t2v-eval` experiment YAML configs (`configs/*.yaml`) from a pasted directory tree (typically `sutree` output). Use when a user provides an experiment folder structure like `hf/AdaHead/Exp-*` with group subfolders and asks to "write yaml", "生成配置", "根据目录写配置", or similar. Convert the tree into repository-conformant sections (`dataset`, `groups`, `protocol`, `metrics`, `runtime`, `paths`, `logging`) with safe defaults and explicit assumptions.
---

# T2V YAML From Tree

## Goal

Create a valid config file for this repository from folder-structure input with minimal back-and-forth.  
Default to a full evaluation config unless the user explicitly asks for VBench-only.

## Parse Input

1. Parse the experiment root path from the first tree line (for example `hf/AdaHead/Exp-K_StaOscCompression`).
2. Parse first-level child directories as `groups[*].name`.
3. Assume each group folder contains `prompts.csv` and videos when the user says the dataset has already been normalized to project convention.
4. Keep user-provided group order unless user asks to sort.

## Infer Defaults

Apply these repository defaults unless the user overrides them:

- `dataset.repo_id`: `"kv-compression/AdaHead"`
- `dataset.split`: `"train"`
- `dataset.use_local_videos`: `true`
- `runtime`: `cuda`, `batch_size: 1`, `num_workers: 4`, `seed: 42`
- Metrics block: full set (`clip_or_vqa`, `vbench_temporal`, `flicker`, `niqe`)
- VBench block: `backend: "vbench_long"`, `use_long: true`, `dimension_profile: "long_16"`
- Output directory: `./outputs/<ExperimentName>`

Infer frame profile from group names:

- If most/all names contain `72`, set:
  - `latent_frames: 72`, `actual_frames: 288`
  - `protocol.num_frames: 173`
  - `clip_or_vqa.num_frames_for_score: 173`
  - `niqe.num_frames_for_niqe: 173`
- Otherwise set 21-frame profile:
  - `latent_frames: 21`, `actual_frames: 84`
  - `protocol.num_frames: 50`
  - `clip_or_vqa.num_frames_for_score: 50`
  - `niqe.num_frames_for_niqe: 50`
- If mixed 21/72 groups appear together, infer per-group `latent_frames/actual_frames` and keep protocol at `50` by default, then add a YAML note comment about the mixed-frame assumption.

## Build YAML

Emit sections in this fixed order:

1. `dataset`
2. `groups`
3. `group_categories` (optional; include only when grouping is meaningful)
4. `protocol`
5. `metrics`
6. `runtime`
7. `paths`
8. `logging`

Use this mapping logic:

- `dataset.local_video_dir`: root path from tree
- `dataset.prompt_file`: `<root>/<first_group>/prompts.csv`
- `dataset.video_dir`: `videos/<ExperimentName>`
- `paths.output_dir`: `./outputs/<ExperimentName>`
- `paths.figures_dir`: `./outputs/<ExperimentName>/figs`
- `paths.experiment_output`: `<ExperimentName>.csv`
- `logging.log_file`: `./outputs/<ExperimentName>/eval.log`

Write `groups[*].description` by humanizing the folder name (replace `_` with spaces and keep concise).

## Handle Prompt Strategy Explicitly

Always include this note near `dataset.prompt_file`:

- This pipeline currently accepts a single `prompt_file`.
- Assume all groups share the same prompt set keyed by `video_id`.
- If prompts differ by group, merge prompts first and then update `prompt_file`.

## Output Requirements

1. Write to `configs/<ExperimentName>.yaml` unless user asks for another filename.
2. Keep 2-space indentation and double-quoted strings.
3. Prefer comments only for assumptions; avoid noisy comments.
4. Ensure `num_frames` alignment across:
   - `protocol.num_frames`
   - `metrics.clip_or_vqa.num_frames_for_score`
   - `metrics.niqe.num_frames_for_niqe`
5. Validate YAML parseability before finishing.

## Example Trigger Phrases

- “根据这个 sutree 写一个 yaml”
- “给这个实验目录生成 configs”
- “我会贴目录树，你直接产出配置”
- “按项目规范自动写 t2v eval 配置”

## Refusal / Clarification Boundary

Ask a follow-up question only if one of these is missing:

- No experiment root path is present
- No group directory names are parseable
- User explicitly asks for a non-default metric profile but does not specify which one

Otherwise proceed directly and document assumptions in YAML comments.
