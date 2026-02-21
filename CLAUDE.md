# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

T2V-Eval is a unified, no-reference evaluation pipeline for text-to-video (T2V) generation. It has two main parts:
- **Python backend** (`scripts/`) — runs evaluation metrics (CLIPScore, VQAScore, VBench-16D, Flicker, NIQE) on generated videos
- **Vue 3 frontend** (`frontend/`) — generates publication-ready LaTeX tables from evaluation CSV results

Official VBench and t2v_metrics implementations are integrated as pinned git submodules in `third_party/`.

## Build & Development Commands

### Python Backend

```bash
# Environment setup
uv venv --python 3.10
uv sync
source .venv/bin/activate

# Initialize submodules (required before running metrics)
git submodule update --init --recursive

# Install detectron2 for VBench 16D object-related dimensions
uv pip install --no-build-isolation "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"

# Package management — always use uv
uv add <package>              # Add a dependency
uv add --dev <package>        # Add a dev dependency

# Run evaluation (CLI entry point: t2v-eval = scripts.run_eval_core:main)
t2v-eval --config configs/<exp>.yaml
uv run python scripts/run_eval_core.py --config configs/<exp>.yaml
uv run python scripts/run_eval_core.py --config configs/<exp>.yaml --skip-vbench
uv run python scripts/run_vbench.py --config configs/Exp_OscStable_Head_Window_vbench16.yaml --force

# Linting & formatting
uv run black --check --line-length 100 scripts/
uv run ruff check scripts/

# Tests
uv run pytest
uv run pytest --cov
uv run pytest tests/test_flicker.py -v    # Run single test file
```

### Frontend (Vue 3 + TypeScript)

```bash
cd frontend
pnpm install
pnpm dev                    # Dev server (Vite)
pnpm build                  # Type-check (vue-tsc) + production build
pnpm preview                # Preview production build
```

## Architecture

### Data Flow

```
Input videos (per group/method)
  → preprocess_videos.py (unify to 8fps, 16 frames, 256px) → eval_cache/
  → Metric runners in parallel:
      run_clip_or_vqa.py → clipvqa_per_video.csv
      run_vbench.py      → vbench_per_video.csv
      run_flicker.py     → flicker_per_video.csv
      run_niqe.py        → niqe_per_video.csv
  → summarize.py (merge + aggregate mean ± std) → outputs/group_summary.csv
  → Copy to frontend/public/data/ for LaTeX table generation
```

### Python Pipeline (`scripts/`)

Entry points are independent metric runners that read YAML configs from `configs/`:
- `run_eval_core.py` — main pipeline orchestrator (non-VBench metrics), also the CLI entry point (`t2v-eval`)
- `run_vbench.py` — VBench temporal quality (supports 2D and 16D profiles)
- `run_clip_or_vqa.py` — CLIPScore/VQAScore via official t2v_metrics submodule
- `run_flicker.py` — temporal flicker detection (custom implementation)
- `run_niqe.py` — NIQE image quality via pyiqa
- `summarize.py` — aggregates per-video CSVs into group summaries (mean ± std)

### VBench Dimension Registry (`scripts/vbench_runner/`)

Custom VBench runner using a **dimension registry pattern**. To add a new dimension:

1. Create `scripts/vbench_runner/dimensions/my_dimension.py` with a `SPEC = DimensionSpec(...)` (see `base.py` for fields: key, description, requires_clip, requires_pyiqa, long_mode_only)
2. Import and add the SPEC to the preset lists in `dimensions/registry.py`

Two presets exist: `LONG_DIMENSIONS_16` (all 16, slow) and `LONG_DIMENSIONS_6_RECOMMENDED` (fast subset, skips 4 GrIT-based dimensions: object_class, multiple_objects, spatial_relationship, color).

Key runner modules: `core.py` (orchestrator with multi-GPU support), `distributed.py` (GPU distribution), `results.py` (result extraction), `env.py` (dependency checks).

### YAML Config Structure (`configs/`)

Configs have these sections: `dataset` (source videos/prompts), `groups` (methods to compare), `protocol` (fps, frames, resolution), `metrics` (which metrics + per-metric params), `runtime` (device, batch_size), `paths` (output dirs), `logging`. See `configs/Exp_.yaml.example` for a full template.

### Frontend (`frontend/src/`)

Vue 3 Composition API with `<script setup>`, styled with Tailwind CSS.

- `App.vue` — top-level state management (reactive data, selected metrics, computed LaTeX)
- `composables/useMetrics.ts` — core logic: CSV loading/parsing, LaTeX generation, clipboard
- `utils/csvParser.ts` — smart auto-detection: recognizes group summary vs per-video CSV, auto-aggregates per-video data
- `utils/latexGenerator.ts` — generates booktabs-format LaTeX tables with best-value highlighting
- `types/metrics.ts` — TypeScript interfaces (`VideoMetric`, `GroupSummary`, `MetricConfig`, `LatexTableOptions`)

### Metric Direction Convention

Higher is better: CLIPScore, VQAScore, all 16 VBench dimensions. Lower is better: Flicker, NIQE. This is encoded in `MetricConfig.direction` in the frontend and affects best-value highlighting in LaTeX output.

## Coding Conventions

- **Python**: `black` (line-length 100), `ruff` (rules: E, F, W, I, N, UP, B, C4; E501 ignored). snake_case functions, explicit CLI args.
- **Package management**: Use `uv add <pkg>` to add dependencies (not `pip install`). Use `uv run` to run Python scripts (e.g. `uv run python scripts/run_vbench.py ...`).
- **Frontend**: TypeScript, PascalCase components, camelCase variables. Vue 3 Composition API.
- **Commits**: Conventional Commits — `feat:`, `fix:`, `docs:`, `style:`, `chores:`
- **Python version**: 3.10 (strict — `>=3.10,<3.11` in pyproject.toml)

## Reference

Detailed operational tutorial (setup, troubleshooting, multi-GPU usage, CLI flags): `docs/USAGE.md`

## Key Version Constraints

Torch ecosystem is pinned for VBench-16D stability: `torch==2.4.*`, `torchvision==0.19.*`, `timm<=1.0.12`, `transformers==4.49.0`. Do not bump these without testing all 16 VBench dimensions.

## Submodules

Pinned in `third_party/` (VBench @ `98b1951`, t2v_metrics @ `0bd9bfc`). Always initialize before running metrics. When updating a submodule, commit the new pin with a message like `chore: update VBench`.
