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
