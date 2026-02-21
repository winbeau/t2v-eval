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

# Run evaluation — always use uv run
uv run python scripts/run_vbench.py --config configs/Exp_OscStable_Head_Window_vbench16.yaml --force
uv run python scripts/run_eval_core.py --config configs/<exp>.yaml
uv run python scripts/run_eval_core.py --config configs/<exp>.yaml --skip-vbench

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

### Python Pipeline (`scripts/`)

Entry points are independent metric runners that read YAML configs from `configs/`:
- `run_eval_core.py` — main pipeline orchestrator (non-VBench metrics), also the CLI entry point (`t2v-eval`)
- `run_vbench.py` — VBench temporal quality (supports 2D and 16D profiles)
- `run_clip_or_vqa.py` — CLIPScore/VQAScore via official t2v_metrics submodule
- `run_flicker.py` — temporal flicker detection (custom implementation)
- `run_niqe.py` — NIQE image quality via pyiqa
- `summarize.py` — aggregates per-video CSVs into group summaries (mean ± std)

`scripts/vbench_runner/` contains a custom VBench runner with a **dimension registry pattern**: `dimensions/registry.py` registers dimension implementations, and each dimension has its own module under `dimensions/`. New VBench dimensions are added by creating a new dimension file and registering it.

Data flow: Input videos → preprocess to `eval_cache/` → run metrics → per-video CSVs in `outputs/` → `summarize.py` → `group_summary.csv`

### Frontend (`frontend/src/`)

Vue 3 Composition API with `<script setup>`, styled with Tailwind CSS.

- `App.vue` — top-level state management (reactive data, selected metrics, computed LaTeX)
- `composables/useMetrics.ts` — core logic: CSV loading/parsing, LaTeX generation, clipboard
- `utils/csvParser.ts` — smart auto-detection: recognizes group summary vs per-video CSV, auto-aggregates per-video data
- `utils/latexGenerator.ts` — generates booktabs-format LaTeX tables with best-value highlighting
- `types/metrics.ts` — TypeScript interfaces (`VideoMetric`, `GroupSummary`, `MetricConfig`, `LatexTableOptions`)
- `components/` — UI: FileUpload (drag-drop CSV), TablePreview (KaTeX rendering), CodeBlock (LaTeX output), MetricSelector, OptionsPanel, LocalDataModal

### Metric Direction Convention

Some metrics are "higher is better" (CLIPScore, VQAScore, VBench dimensions), others are "lower is better" (Flicker, NIQE). This is encoded in `MetricConfig.direction` in the frontend and affects best-value highlighting in LaTeX output.

## Coding Conventions

- **Python**: `black` (line-length 100), `ruff` (rules: E, F, W, I, N, UP, B, C4; E501 ignored). snake_case functions, explicit CLI args.
- **Frontend**: TypeScript, PascalCase components, camelCase variables. Vue 3 Composition API.
- **Commits**: Conventional Commits — `feat:`, `fix:`, `docs:`, `style:`, `chores:`
- **Python version**: 3.10 (strict — `>=3.10,<3.11` in pyproject.toml)

## Submodules

Pinned in `third_party/`. Always initialize before running metrics. When updating a submodule, commit the new pin with a message like `chore: update VBench`.
