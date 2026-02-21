# Repository Guidelines

## Project Structure & Module Organization
- `configs/`: experiment YAMLs (per-run settings).
- `scripts/`: Python pipeline entry points (`run_all.py`, metric runners, preprocessing).
- `frontend/`: Vue 3 LaTeX table generator (`src/`, `public/data/`).
- `third_party/`: git submodules for official VBench and `t2v_metrics`.
- `outputs/`: generated CSVs and figures; `eval_cache/` stores preprocessed video cache.
- `docs/`: images and tutorial material.

## Build, Test, and Development Commands
- `uv venv --python 3.10` and `uv pip install -e .`: create env and install Python deps.
- `uv pip install -r third_party/VBench/requirements.txt`: VBench extras after submodules.
- `python scripts/run_all.py --config configs/<exp>.yaml`: run full evaluation.
- `python scripts/run_all.py --config configs/<exp>.yaml --skip-vbench`: skip a metric.
- `cd frontend && pnpm install && pnpm dev`: run the LaTeX table UI.
- `cd frontend && pnpm build`: build frontend for preview/distribution.

## Coding Style & Naming Conventions
- Python is formatted with `black` (line length 100) and linted with `ruff`.
- Follow existing patterns in `scripts/` (snake_case functions, explicit CLI args).
- Frontend uses TypeScript/Vue; keep component names in `PascalCase` and variables in `camelCase` as seen in `frontend/src/`.

## YAML 编写规范（`configs/*.yaml`）
- 适用范围：仅针对本仓库评测配置（`configs/`）；不要把 `third_party/` 下的 YAML 风格混入实验配置。
- 基本格式：
  - 使用 2 空格缩进，禁止 Tab。
  - 顶层字段顺序建议保持：`dataset` → `groups` → `group_categories`（可选）→ `protocol` → `metrics` → `runtime` → `paths` → `logging`。
  - 字符串使用双引号；数值和布尔值保持裸值（如 `173`、`true`）。
  - 列表逐行写（`- item`），避免行内复杂对象。
- 必填与强依赖字段：
  - 顶层至少要有：`dataset`、`paths`；若要跑预处理还必须有 `protocol`。
  - `dataset.repo_id` 为硬依赖（即使 `use_local_videos: true` 也保留）。
  - `paths` 至少包含：`cache_dir`、`output_dir`、`metadata_file`、`processed_metadata`、`per_video_metrics`、`group_summary`。
- `dataset` 约定：
  - 本地视频模式请设置：`use_local_videos: true`、`local_video_dir`、`prompt_file`。
  - `groups[*].name` 应与本地目录下的组名一致；不一致的视频会被跳过。
  - 若目录结构无法推断分组，提供 `default_group` 兜底。
- `groups` / `group_categories`：
  - `groups` 是对象列表，最少包含 `name`；`description`、`latent_frames`、`actual_frames` 为推荐元数据。
  - `group_categories` 仅用于分析分组，不参与核心执行逻辑。
- `protocol` 约定：
  - 常用固定键：`fps_eval`、`num_frames`、`resize`、`frame_sampling`、`frame_padding`。
  - `frame_sampling` 当前建议使用 `"uniform"`。
  - `frame_padding` 建议优先 `"loop"`（可选 `"repeat_last"`、`"truncate"`）。
- `metrics` 约定：
  - 推荐保持 `metrics.enabled` 与实际子块一致（虽然脚本主要由 CLI `--skip-*` 控制）。
  - `clip_or_vqa.num_frames_for_score` 与 `niqe.num_frames_for_niqe` 应与 `protocol.num_frames` 对齐。
  - VBench-Long 推荐显式设置 `metrics.vbench.backend: "vbench_long"` 或 `use_long: true`；16 维建议配 `dimension_profile: "long_16"` 并明确 `subtasks`。
- `paths` / `logging` 约定：
  - `paths.output_dir` 使用实验专属目录，避免覆盖其他实验结果。
  - `experiment_output` 使用唯一文件名（通常与配置名对应）。
  - `logging.log_file` 放在该实验输出目录下，便于归档。

## Testing Guidelines
- `pytest` is configured in `pyproject.toml` with `tests/` and `test_*.py` naming.
- The repo currently has no `tests/` directory; add tests there when introducing new logic.
- Optional coverage: `pytest --cov` (requires `pytest-cov`).

## Commit & Pull Request Guidelines
- Commit messages follow a lightweight Conventional Commits style: `feat:`, `fix:`, `docs:`, `style:`, `chores:`.
- PRs should include:
  - A short summary and the motivation.
  - The config used (`configs/<exp>.yaml`) and any new outputs.
  - Screenshots for frontend changes (table preview or UI updates).
  - Notes on submodule updates if applicable.

## Submodules & Reproducibility
- Initialize submodules before running metrics: `git submodule update --init --recursive`.
- If you update a submodule, commit the new pin with a clear message (e.g., `chore: update VBench`).
