# T2V-Eval: Text-to-Video Evaluation with Official Implementations

[中文文档](README_zh.md) | English

A unified, no-reference evaluation pipeline for **text-to-video (T2V) generation**, with a focus on comparing different inference and acceleration strategies.

> **Key Feature**: This repository uses **official implementations** of VBench and t2v_metrics via git submodules for reproducibility and credibility.

---

## Official Implementations

This evaluation suite integrates the following official repositories as git submodules:

| Repository | Purpose | Version |
|------------|---------|---------|
| [VBench](https://github.com/Vchitect/VBench) | Temporal quality evaluation | `98b1951` |
| [t2v_metrics](https://github.com/linzhiqiu/t2v_metrics) | CLIPScore / VQAScore | `0bd9bfc` |

**Why submodules?**
- Ensures reproducibility with pinned commit hashes
- Uses original, peer-reviewed implementations
- Allows independent updates while maintaining version control
- Avoids code duplication and potential implementation drift

---

## Quick Start

### 1. Clone with Submodules

```bash
# Clone with submodules (recommended)
git clone --recurse-submodules https://github.com/YOUR_USERNAME/t2v-eval.git
cd t2v-eval

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### 2. Install Dependencies (using uv)

We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv --python 3.10
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install project dependencies
uv pip install -e .

# Install VBench dependencies
uv pip install -r third_party/VBench/requirements.txt

# (Optional) Install development dependencies
uv pip install -e ".[dev]"
```

<details>
<summary>Alternative: Using pip</summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r third_party/VBench/requirements.txt
```

</details>

### 3. Configure Dataset

Create a config file for your experiment in `configs/`:

```bash
# Example: configs/Exp_OscStable_Head_Window.yaml
```

```yaml
dataset:
  repo_id: "winbeau/AdaHead"  # Your HuggingFace dataset
  video_dir: "videos/Exp_OscStable_Head_Window"

groups:
  - name: "frame_baseline_21"
    description: "Frame-level baseline, 21 frames"
  # ... more groups
```

### 4. Run Full Pipeline

```bash
# One-click evaluation (specify your config)
python scripts/run_all.py --config configs/Exp_OscStable_Head_Window.yaml

# Or with auto submodule initialization
python scripts/run_all.py --config configs/Exp_OscStable_Head_Window.yaml --auto-init-submodules

# Skip specific metrics if needed
python scripts/run_all.py --config configs/Exp_OscStable_Head_Window.yaml --skip-vbench
```

### 5. View Results

```
outputs/
├── per_video_metrics.csv    # Per-video scores
├── group_summary.csv        # Group-level mean ± std
└── figs/                    # (Optional) Visualizations
```

---

## Metrics Overview

| Metric | Aspect | Direction | Implementation |
|--------|--------|-----------|----------------|
| CLIPScore / VQAScore | Text-video alignment | ↑ Higher is better | Official t2v_metrics |
| VBench (Temporal) | Temporal quality | ↑ Higher is better | Official VBench |
| Temporal Flicker | Frame stability | ↓ Lower is better | Custom (this repo) |
| NIQE | Visual quality | ↓ Lower is better | pyiqa |
| #Frames / Duration | Generation scale | — | Metadata |
| FPS | Inference efficiency | ↑ Higher is better | User-provided |

---

## Project Structure

```
t2v-eval/
├── configs/
│   └── Exp_*.yaml            # Per-experiment configs
├── scripts/
│   ├── export_from_hf.py      # Export dataset from HuggingFace
│   ├── preprocess_videos.py   # Unify video format
│   ├── run_clip_or_vqa.py     # CLIPScore/VQAScore via t2v_metrics
│   ├── run_vbench.py          # VBench temporal evaluation
│   ├── run_flicker.py         # Temporal flicker score
│   ├── run_niqe.py            # NIQE image quality
│   ├── summarize.py           # Aggregate results
│   └── run_all.py             # One-click pipeline entry
├── third_party/               # Git submodules
│   ├── VBench/                # Official VBench repo
│   └── t2v_metrics/           # Official t2v_metrics repo
├── outputs/                   # Evaluation results
├── eval_cache/                # Preprocessed video cache
├── pyproject.toml             # Project config (uv/pip)
└── README.md
```

---

## Evaluation Protocol

All videos are preprocessed to a unified format:

```yaml
protocol:
  fps_eval: 8            # Evaluation FPS
  num_frames: 16         # Fixed frame count
  resize: 256            # Spatial resolution
  frame_sampling: uniform
  frame_padding: loop    # Handle short videos
```

---

## Reproducibility

### Pinned Submodule Versions

| Submodule | Commit Hash |
|-----------|-------------|
| VBench | `98b19513678e99c80d8377fda25ba53b81a491a6` |
| t2v_metrics | `0bd9bfc68032ce4f9d5da80d646fa5ceb3b9bb1b` |

### Updating/Locking Submodules

```bash
# Update to latest
cd third_party/VBench && git pull origin main && cd ../..
git add third_party/VBench && git commit -m "Update VBench"

# Lock to specific commit
git -C third_party/VBench checkout <commit_hash>
git add third_party/VBench && git commit -m "Pin VBench to <hash>"
```

---

## Citation

```bibtex
@article{huang2023vbench,
  title={VBench: Comprehensive Benchmark Suite for Video Generative Models},
  author={Huang, Ziqi and others},
  journal={arXiv preprint arXiv:2311.17982},
  year={2023}
}

@article{lin2024evaluating,
  title={Evaluating Text-to-Visual Generation with Image-to-Text Generation},
  author={Lin, Zhiqiu and others},
  journal={arXiv preprint arXiv:2404.01291},
  year={2024}
}
```

---

## Paper Statement Template

> We evaluate temporal quality using the **official VBench implementation** (Huang et al., 2023) and text-video alignment using **t2v_metrics** (Lin et al., 2024). Both implementations are integrated as git submodules with pinned commit hashes to ensure reproducibility. Specifically, we use VBench commit `98b1951` and t2v_metrics commit `0bd9bfc`.

---

## License

MIT License. VBench and t2v_metrics have their own licenses.
