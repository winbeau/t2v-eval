# T2V-Eval Configuration & Output Tutorial

This tutorial explains the YAML configuration format and CSV output logic for the T2V-Eval (Text-to-Video Evaluation) pipeline.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [YAML Configuration Structure](#yaml-configuration-structure)
3. [Configuration Sections](#configuration-sections)
4. [CSV Output Files](#csv-output-files)
5. [Frame Sampling Logic](#frame-sampling-logic)
6. [Best Practices](#best-practices)

---

## Quick Start

Use the example template to create your experiment configuration:

```bash
# Copy the template
cp configs/Exp_.yaml.example configs/Exp_MyExperiment.yaml

# Edit with your settings
nano configs/Exp_MyExperiment.yaml

# Run evaluation
python scripts/run_all.py --config configs/Exp_MyExperiment.yaml
```

The template file `configs/Exp_.yaml.example` contains detailed comments explaining every configuration option.

---

## YAML Configuration Structure

The configuration file controls all aspects of the evaluation pipeline. Here's the complete structure:

```yaml
# Dataset settings
dataset:
  repo_id: "..."
  split: "..."
  use_local_videos: true/false
  local_video_dir: "..."
  prompt_file: "..."
  video_dir: "..."

# Experiment groups
groups:
  - name: "group_name"
    description: "..."
    latent_frames: N
    actual_frames: N

# Group categorization
group_categories:
  by_attention_type: {...}
  by_num_frames: {...}

# Evaluation protocol
protocol:
  fps_eval: N
  num_frames: N
  resize: N
  frame_sampling: "uniform"
  frame_padding: "loop"

# Metrics configuration
metrics:
  enabled: [...]
  clip_or_vqa: {...}
  vbench: {...}
  flicker: {...}
  niqe: {...}

# Runtime settings
runtime:
  device: "cuda"
  batch_size: N
  num_workers: N
  seed: N

# Output paths
paths:
  cache_dir: "..."
  output_dir: "..."
  experiment_output: "..."

# Logging
logging:
  level: "INFO"
  log_file: "..."
  console: true
```

---

## Configuration Sections

### 1. Dataset Configuration

```yaml
dataset:
  repo_id: "hf/AdaHead"              # HuggingFace dataset repository ID
  split: "train"                      # Dataset split to use
  use_local_videos: true              # Use local videos instead of downloading
  local_video_dir: "hf/AdaHead/videos/Exp_OscStable_Head_Window"
  prompt_file: "hf/AdaHead/videos/Exp_OscStable_Head_Window/prompts.csv"
  video_dir: "videos/Exp_OscStable_Head_Window"  # Relative path for video lookup
```

**Key Points:**
- `use_local_videos: true` - Load videos from local filesystem
- `prompt_file` - CSV file containing video prompts (columns: `video_id`, `prompt`)
- `video_dir` - Directory structure: `{video_dir}/{group_name}/*.mp4`

### 2. Experiment Groups

```yaml
groups:
  - name: "frame_baseline_21"
    description: "Frame-level baseline, 21 latent frames (84 actual)"
    latent_frames: 21
    actual_frames: 84

  - name: "osc_long_72"
    description: "Oscillating attention (long window), 72 latent frames"
    latent_frames: 72
    actual_frames: 288
```

**Important Concept: Latent vs Actual Frames**

| Latent Frames | Actual Frames (MP4) | Relationship |
|---------------|---------------------|--------------|
| 21 | 84 | actual = latent × 4 |
| 72 | 288 | actual = latent × 4 |

> **Why ×4?** Video diffusion models (like AR-DiT) operate in latent space. After VAE decoding, each latent frame expands to 4 actual video frames.

### 3. Group Categories (Optional)

```yaml
group_categories:
  by_attention_type:
    frame: ["frame_baseline_21", "frame_baseline_72"]
    head: ["head_baseline_21", "head_baseline_72"]
    oscillating: ["osc_long_21", "osc_long_72", "osc_short_21", "osc_short_72"]
    stable: ["stable_long_21", "stable_long_72", "stable_short_21", "stable_short_72"]

  by_num_frames:
    short_21: ["frame_baseline_21", "head_baseline_21", ...]
    long_72: ["frame_baseline_72", "head_baseline_72", ...]

  by_window_size:
    long_window: ["osc_long_21", "osc_long_72", "stable_long_21", "stable_long_72"]
    short_window: ["osc_short_21", "osc_short_72", "stable_short_21", "stable_short_72"]
```

Used for downstream analysis and visualization grouping.

### 4. Evaluation Protocol

```yaml
protocol:
  fps_eval: 8                    # Evaluation FPS (frames per second)
  num_frames: 173                # Number of frames to SAMPLE for evaluation
  resize: 256                    # Spatial resolution (width & height)
  frame_sampling: "uniform"      # Sampling strategy
  frame_padding: "loop"          # Handling when video < num_frames
```

**Frame Sampling Calculation (60% Coverage):**

| Video Type | Actual Frames | 60% Coverage | Recommended `num_frames` |
|------------|---------------|--------------|--------------------------|
| 21 latent | 84 | 50.4 | 50 |
| 72 latent | 288 | 172.8 | 173 |

**Frame Padding Options:**
- `loop` - Loop video from beginning when frames exhausted
- `repeat_last` - Repeat the last frame
- `truncate` - Stop at available frames (may cause errors if too few)

### 5. Metrics Configuration

#### 5.1 Enabled Metrics

```yaml
metrics:
  enabled:
    - "clip_or_vqa"      # Text-video alignment (CLIP score)
    - "vbench_temporal"  # Temporal quality (VBench)
    - "flicker"          # Frame-to-frame consistency
    - "niqe"             # Perceptual image quality
```

#### 5.2 CLIP/VQA Settings

```yaml
clip_or_vqa:
  mode: "clip"                 # Options: clip, vqa
  model_name: "ViT-B-32"       # CLIP model variant
  pretrained: "openai"         # Pretrained weights source
  num_frames_for_score: 173    # Should match protocol.num_frames
  aggregation: "mean"          # How to aggregate: mean, max
```

#### 5.3 VBench Settings

```yaml
vbench:
  enabled: true
  temporal_only: true          # Focus on temporal metrics
  subtasks:
    - "temporal_flickering"    # Measures visual consistency
    - "motion_smoothness"      # Measures motion quality
```

> **Warning:** `subject_consistency` and `temporal_style` often fail with `ZeroDivisionError` on custom videos. Stick to the recommended subtasks.

#### 5.4 Flicker Settings

```yaml
flicker:
  method: "l1"                 # Difference method: l1, l2
  normalize: true              # Normalize pixel values to [0, 1]
  compute_std: true            # Also compute std of frame differences
  grayscale: false             # Convert to grayscale before computing
```

#### 5.5 NIQE Settings

```yaml
niqe:
  enabled: true
  num_frames_for_niqe: 173     # Should match protocol.num_frames
  block_size: 96               # NIQE block size
  alternative: "niqe"          # Options: niqe, brisque
```

### 6. Runtime Configuration

```yaml
runtime:
  device: "cuda"               # Device: cuda, cpu
  batch_size: 1                # Batch size for inference
  num_workers: 4               # DataLoader workers
  seed: 42                     # Random seed for reproducibility
```

### 7. Paths Configuration

```yaml
paths:
  cache_dir: "./eval_cache"                    # Cache for model weights
  output_dir: "./outputs"                       # Main output directory
  metadata_file: "metadata.csv"                 # Raw video metadata
  processed_metadata: "processed_metadata.csv"  # Processed metadata
  per_video_metrics: "per_video_metrics.csv"    # Per-video results
  group_summary: "group_summary.csv"            # Aggregated results
  runtime_csv: "runtime.csv"                    # Timing information
  figures_dir: "./outputs/figs"                 # Visualization output

  # Custom experiment output filename
  experiment_output: "Exp_72frames_OscStable_Head_Win.csv"
```

### 8. Logging Configuration

```yaml
logging:
  level: "INFO"                # DEBUG, INFO, WARNING, ERROR
  log_file: "./outputs/eval.log"
  console: true                # Print to console
```

---

## CSV Output Files

### 1. Per-Video Metrics (`per_video_metrics.csv`)

Contains individual scores for each video:

| Column | Description | Example |
|--------|-------------|---------|
| `video_id` | Unique video identifier | `prompt_001_frame_baseline_21` |
| `group` | Experiment group name | `frame_baseline_21` |
| `prompt` | Text prompt used | `A cat walking...` |
| `clip_score` | CLIP alignment score | `0.2847` |
| `temporal_flickering` | VBench temporal flickering | `0.9823` |
| `motion_smoothness` | VBench motion smoothness | `0.9567` |
| `flicker_mean` | Mean frame difference | `0.0234` |
| `flicker_std` | Std of frame differences | `0.0089` |
| `niqe_score` | NIQE quality score | `4.567` |

### 2. Group Summary (`group_summary.csv`)

Aggregated statistics per experiment group:

| Column | Description |
|--------|-------------|
| `group` | Group name |
| `{metric}_mean` | Mean value across all videos in group |
| `{metric}_std` | Standard deviation |
| `video_count` | Number of videos in group |

**Example:**
```csv
group,clip_score_mean,clip_score_std,temporal_flickering_mean,temporal_flickering_std,video_count
frame_baseline_21,0.2847,0.0234,0.9823,0.0156,50
head_baseline_21,0.2912,0.0198,0.9845,0.0143,50
osc_long_21,0.2756,0.0267,0.9789,0.0178,50
```

### 3. Experiment Output (`experiment_output`)

When `paths.experiment_output` is set, the group summary is also saved with this custom filename:

```yaml
paths:
  experiment_output: "Exp_72frames_OscStable_Head_Win.csv"
```

This file is identical to `group_summary.csv` but uses your specified filename for easy identification.

### 4. VBench Results (`vbench_per_video.csv`)

Separate file for VBench-specific metrics:

| Column | Description |
|--------|-------------|
| `video_id` | Video identifier |
| `temporal_flickering` | Temporal flickering score (0-1, higher=better) |
| `motion_smoothness` | Motion smoothness score (0-1, higher=better) |
| `vbench_temporal_score` | Average of temporal metrics |
| `group` | Experiment group |

---

## Frame Sampling Logic

### Sampling Process

1. **Load Video**: Read all frames from MP4 file
2. **Apply Sampling Strategy**:
   - `uniform`: Evenly spaced frames across video
   - Formula: `indices = linspace(0, total_frames-1, num_frames)`
3. **Handle Short Videos**:
   - If `total_frames < num_frames`, apply padding strategy
4. **Resize Frames**: Resize to `protocol.resize × protocol.resize`

### Example: 60% Coverage Calculation

For 72 latent frames (288 actual frames):
```
coverage_ratio = 0.60
actual_frames = 288
num_frames = round(288 × 0.60) = 173
```

For 21 latent frames (84 actual frames):
```
coverage_ratio = 0.60
actual_frames = 84
num_frames = round(84 × 0.60) = 50
```

### Why 60% Coverage?

- **Computational Efficiency**: Reduces processing time while maintaining quality
- **Exposure Bias Mitigation**: AR-DiT models may have quality degradation in later frames
- **Standardization**: Consistent evaluation across different video lengths

---

## Best Practices

### 1. Consistent Frame Counts

Ensure these values match across your config:
```yaml
protocol:
  num_frames: 173

metrics:
  clip_or_vqa:
    num_frames_for_score: 173  # Match!
  niqe:
    num_frames_for_niqe: 173   # Match!
```

### 2. Group Naming Convention

Use descriptive, consistent naming:
```
{method}_{variant}_{latent_frames}
```

Examples:
- `frame_baseline_21` - Frame-level baseline, 21 latent frames
- `osc_long_72` - Oscillating attention, long window, 72 latent frames
- `stable_short_21` - Stable attention, short window, 21 latent frames

### 3. Separate Configs for Different Experiments

Create dedicated YAML files for each experiment setup:
- `Exp_21frames_OscStable_Head_Win.yaml` - 21 latent frame experiments
- `Exp_72frames_OscStable_Head_Win.yaml` - 72 latent frame experiments

### 4. VBench Subtask Selection

Avoid problematic subtasks:
```yaml
# Recommended (stable)
subtasks:
  - "temporal_flickering"
  - "motion_smoothness"

# Avoid (may fail)
# - "subject_consistency"  # ZeroDivisionError
# - "temporal_style"       # Content-dependent failures
```

### 5. Output File Organization

```
outputs/
├── eval.log                           # Execution log
├── metadata.csv                       # Raw metadata
├── processed_metadata.csv             # Processed metadata
├── per_video_metrics.csv              # Individual scores
├── group_summary.csv                  # Aggregated results
├── Exp_72frames_OscStable_Head_Win.csv # Custom experiment output
├── vbench_per_video.csv               # VBench results
├── vbench_results/                    # VBench intermediate files
│   ├── input_videos/                  # Symlinked videos
│   └── *_eval_results.json            # Raw VBench outputs
└── figs/                              # Visualizations
```

---

## Quick Reference

### Metric Interpretation

| Metric | Range | Better |
|--------|-------|--------|
| `clip_score` | 0-1 | Higher |
| `temporal_flickering` | 0-1 | Higher |
| `motion_smoothness` | 0-1 | Higher |
| `flicker_mean` | 0+ | Lower |
| `niqe_score` | 0+ | Lower |

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| VBench ZeroDivisionError | Remove `subject_consistency` from subtasks |
| Empty CLIP scores | Check `prompt_file` path and format |
| Frame padding warnings | Increase `num_frames` or check video length |
| CUDA out of memory | Reduce `batch_size` to 1 |

---

## Creating a New Experiment

### Step 1: Copy the Template

```bash
cp configs/Exp_.yaml.example configs/Exp_YourExperiment.yaml
```

### Step 2: Modify Key Settings

Edit the following sections based on your experiment:

| Section | What to Change |
|---------|----------------|
| `dataset.local_video_dir` | Path to your video directory |
| `dataset.prompt_file` | Path to your prompts CSV |
| `groups` | Define your experiment groups |
| `protocol.num_frames` | Set based on 60% coverage |
| `paths.experiment_output` | Custom output filename |

### Step 3: Run Evaluation

```bash
python scripts/run_all.py --config configs/Exp_YourExperiment.yaml
```

### Step 4: View Results

Results will be saved to:
- `outputs/per_video_metrics.csv` - Individual video scores
- `outputs/group_summary.csv` - Aggregated statistics
- `frontend/public/data/{experiment_output}` - For visualization

---

*Last updated: 2025-01-25*
