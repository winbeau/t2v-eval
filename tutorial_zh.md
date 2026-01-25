# T2V-Eval 配置与输出教程

本教程详细讲解 T2V-Eval（文本到视频评估）流水线的 YAML 配置格式和 CSV 输出逻辑。

---

## 目录

1. [快速开始](#快速开始)
2. [YAML 配置结构](#yaml-配置结构)
3. [配置模块详解](#配置模块详解)
4. [CSV 输出文件](#csv-输出文件)
5. [帧采样逻辑](#帧采样逻辑)
6. [最佳实践](#最佳实践)

---

## 快速开始

使用示例模板创建你的实验配置：

```bash
# 复制模板
cp configs/Exp_.yaml.example configs/Exp_MyExperiment.yaml

# 编辑配置
nano configs/Exp_MyExperiment.yaml

# 运行评估
python scripts/run_all.py --config configs/Exp_MyExperiment.yaml
```

模板文件 `configs/Exp_.yaml.example` 包含详细注释，解释了每个配置选项。

---

## YAML 配置结构

配置文件控制评估流水线的所有方面。完整结构如下：

```yaml
# 数据集设置
dataset:
  repo_id: "..."
  split: "..."
  use_local_videos: true/false
  local_video_dir: "..."
  prompt_file: "..."
  video_dir: "..."

# 实验组
groups:
  - name: "group_name"
    description: "..."
    latent_frames: N
    actual_frames: N

# 分组分类
group_categories:
  by_attention_type: {...}
  by_num_frames: {...}

# 评估协议
protocol:
  fps_eval: N
  num_frames: N
  resize: N
  frame_sampling: "uniform"
  frame_padding: "loop"

# 指标配置
metrics:
  enabled: [...]
  clip_or_vqa: {...}
  vbench: {...}
  flicker: {...}
  niqe: {...}

# 运行时设置
runtime:
  device: "cuda"
  batch_size: N
  num_workers: N
  seed: N

# 输出路径
paths:
  cache_dir: "..."
  output_dir: "..."
  experiment_output: "..."

# 日志
logging:
  level: "INFO"
  log_file: "..."
  console: true
```

---

## 配置模块详解

### 1. 数据集配置 (dataset)

```yaml
dataset:
  repo_id: "hf/AdaHead"              # HuggingFace 数据集仓库 ID
  split: "train"                      # 数据集划分
  use_local_videos: true              # 使用本地视频而非下载
  local_video_dir: "hf/AdaHead/videos/Exp_OscStable_Head_Window"
  prompt_file: "hf/AdaHead/videos/Exp_OscStable_Head_Window/prompts.csv"
  video_dir: "videos/Exp_OscStable_Head_Window"  # 视频查找的相对路径
```

**关键说明：**
- `use_local_videos: true` - 从本地文件系统加载视频
- `prompt_file` - 包含视频提示词的 CSV 文件（列：`video_id`, `prompt`）
- `video_dir` - 目录结构：`{video_dir}/{group_name}/*.mp4`

### 2. 实验组配置 (groups)

```yaml
groups:
  - name: "frame_baseline_21"
    description: "帧级别基线，21 潜在帧（84 实际帧）"
    latent_frames: 21
    actual_frames: 84

  - name: "osc_long_72"
    description: "振荡注意力（长窗口），72 潜在帧"
    latent_frames: 72
    actual_frames: 288
```

**重要概念：潜在帧 vs 实际帧**

| 潜在帧 (Latent) | 实际帧 (MP4) | 关系 |
|----------------|--------------|------|
| 21 | 84 | 实际帧 = 潜在帧 × 4 |
| 72 | 288 | 实际帧 = 潜在帧 × 4 |

> **为什么 ×4？** 视频扩散模型（如 AR-DiT）在潜在空间中操作。经过 VAE 解码后，每个潜在帧会扩展为 4 个实际视频帧。

### 3. 分组分类 (group_categories)（可选）

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

用于下游分析和可视化的分组依据。

### 4. 评估协议 (protocol)

```yaml
protocol:
  fps_eval: 8                    # 评估帧率（每秒帧数）
  num_frames: 173                # 用于评估的采样帧数
  resize: 256                    # 空间分辨率（宽度和高度）
  frame_sampling: "uniform"      # 采样策略
  frame_padding: "loop"          # 视频帧数不足时的处理方式
```

**帧采样计算（60% 覆盖率）：**

| 视频类型 | 实际帧数 | 60% 覆盖 | 推荐 `num_frames` |
|----------|----------|----------|-------------------|
| 21 潜在帧 | 84 | 50.4 | 50 |
| 72 潜在帧 | 288 | 172.8 | 173 |

**帧填充选项 (frame_padding)：**
- `loop` - 视频帧耗尽时从头循环播放
- `repeat_last` - 重复最后一帧
- `truncate` - 在可用帧处停止（帧数过少时可能导致错误）

### 5. 指标配置 (metrics)

#### 5.1 启用的指标

```yaml
metrics:
  enabled:
    - "clip_or_vqa"      # 文本-视频对齐度（CLIP 分数）
    - "vbench_temporal"  # 时间质量（VBench）
    - "flicker"          # 帧间一致性
    - "niqe"             # 感知图像质量
```

#### 5.2 CLIP/VQA 设置

```yaml
clip_or_vqa:
  mode: "clip"                 # 选项：clip, vqa
  model_name: "ViT-B-32"       # CLIP 模型变体
  pretrained: "openai"         # 预训练权重来源
  num_frames_for_score: 173    # 应与 protocol.num_frames 一致
  aggregation: "mean"          # 聚合方式：mean, max
```

#### 5.3 VBench 设置

```yaml
vbench:
  enabled: true
  temporal_only: true          # 专注于时间维度指标
  subtasks:
    - "temporal_flickering"    # 测量视觉一致性
    - "motion_smoothness"      # 测量运动质量
```

> **警告：** `subject_consistency` 和 `temporal_style` 在自定义视频上经常出现 `ZeroDivisionError` 错误。建议使用推荐的子任务。

#### 5.4 Flicker（闪烁）设置

```yaml
flicker:
  method: "l1"                 # 差异计算方法：l1, l2
  normalize: true              # 将像素值归一化到 [0, 1]
  compute_std: true            # 同时计算帧差异的标准差
  grayscale: false             # 计算前转换为灰度图
```

#### 5.5 NIQE 设置

```yaml
niqe:
  enabled: true
  num_frames_for_niqe: 173     # 应与 protocol.num_frames 一致
  block_size: 96               # NIQE 块大小
  alternative: "niqe"          # 选项：niqe, brisque
```

### 6. 运行时配置 (runtime)

```yaml
runtime:
  device: "cuda"               # 设备：cuda, cpu
  batch_size: 1                # 推理批次大小
  num_workers: 4               # DataLoader 工作进程数
  seed: 42                     # 随机种子，确保可复现性
```

### 7. 路径配置 (paths)

```yaml
paths:
  cache_dir: "./eval_cache"                    # 模型权重缓存
  output_dir: "./outputs"                       # 主输出目录
  metadata_file: "metadata.csv"                 # 原始视频元数据
  processed_metadata: "processed_metadata.csv"  # 处理后的元数据
  per_video_metrics: "per_video_metrics.csv"    # 单视频结果
  group_summary: "group_summary.csv"            # 聚合结果
  runtime_csv: "runtime.csv"                    # 计时信息
  figures_dir: "./outputs/figs"                 # 可视化输出

  # 自定义实验输出文件名
  experiment_output: "Exp_72frames_OscStable_Head_Win.csv"
```

### 8. 日志配置 (logging)

```yaml
logging:
  level: "INFO"                # DEBUG, INFO, WARNING, ERROR
  log_file: "./outputs/eval.log"
  console: true                # 输出到控制台
```

---

## CSV 输出文件

### 1. 单视频指标 (`per_video_metrics.csv`)

包含每个视频的独立评分：

| 列名 | 描述 | 示例 |
|------|------|------|
| `video_id` | 唯一视频标识符 | `prompt_001_frame_baseline_21` |
| `group` | 实验组名称 | `frame_baseline_21` |
| `prompt` | 使用的文本提示词 | `A cat walking...` |
| `clip_score` | CLIP 对齐分数 | `0.2847` |
| `temporal_flickering` | VBench 时间闪烁 | `0.9823` |
| `motion_smoothness` | VBench 运动平滑度 | `0.9567` |
| `flicker_mean` | 平均帧差异 | `0.0234` |
| `flicker_std` | 帧差异标准差 | `0.0089` |
| `niqe_score` | NIQE 质量分数 | `4.567` |

### 2. 分组汇总 (`group_summary.csv`)

每个实验组的聚合统计：

| 列名 | 描述 |
|------|------|
| `group` | 组名 |
| `{metric}_mean` | 组内所有视频的平均值 |
| `{metric}_std` | 标准差 |
| `video_count` | 组内视频数量 |

**示例：**
```csv
group,clip_score_mean,clip_score_std,temporal_flickering_mean,temporal_flickering_std,video_count
frame_baseline_21,0.2847,0.0234,0.9823,0.0156,50
head_baseline_21,0.2912,0.0198,0.9845,0.0143,50
osc_long_21,0.2756,0.0267,0.9789,0.0178,50
```

### 3. 实验输出 (`experiment_output`)

当设置了 `paths.experiment_output` 时，分组汇总会以此自定义文件名另存一份：

```yaml
paths:
  experiment_output: "Exp_72frames_OscStable_Head_Win.csv"
```

此文件与 `group_summary.csv` 内容相同，但使用您指定的文件名以便识别。

### 4. VBench 结果 (`vbench_per_video.csv`)

VBench 专用指标的独立文件：

| 列名 | 描述 |
|------|------|
| `video_id` | 视频标识符 |
| `temporal_flickering` | 时间闪烁分数（0-1，越高越好） |
| `motion_smoothness` | 运动平滑度分数（0-1，越高越好） |
| `vbench_temporal_score` | 时间指标的平均值 |
| `group` | 实验组 |

---

## 帧采样逻辑

### 采样流程

1. **加载视频**：读取 MP4 文件的所有帧
2. **应用采样策略**：
   - `uniform`：在视频中均匀间隔采样
   - 公式：`indices = linspace(0, total_frames-1, num_frames)`
3. **处理短视频**：
   - 如果 `total_frames < num_frames`，应用填充策略
4. **调整帧大小**：调整为 `protocol.resize × protocol.resize`

### 示例：60% 覆盖率计算

对于 72 潜在帧（288 实际帧）：
```
覆盖率 = 0.60
实际帧数 = 288
num_frames = round(288 × 0.60) = 173
```

对于 21 潜在帧（84 实际帧）：
```
覆盖率 = 0.60
实际帧数 = 84
num_frames = round(84 × 0.60) = 50
```

### 为什么选择 60% 覆盖率？

- **计算效率**：在保持质量的同时减少处理时间
- **曝光偏差缓解**：AR-DiT 模型在后期帧可能存在质量下降（exposure bias）
- **标准化**：不同视频长度的一致评估标准

---

## 最佳实践

### 1. 保持帧数一致

确保配置中这些值保持一致：
```yaml
protocol:
  num_frames: 173

metrics:
  clip_or_vqa:
    num_frames_for_score: 173  # 保持一致！
  niqe:
    num_frames_for_niqe: 173   # 保持一致！
```

### 2. 组命名规范

使用描述性的一致命名：
```
{方法}_{变体}_{潜在帧数}
```

示例：
- `frame_baseline_21` - 帧级别基线，21 潜在帧
- `osc_long_72` - 振荡注意力，长窗口，72 潜在帧
- `stable_short_21` - 稳定注意力，短窗口，21 潜在帧

### 3. 为不同实验创建独立配置

为每个实验设置创建专用 YAML 文件：
- `Exp_21frames_OscStable_Head_Win.yaml` - 21 潜在帧实验
- `Exp_72frames_OscStable_Head_Win.yaml` - 72 潜在帧实验

### 4. VBench 子任务选择

避免有问题的子任务：
```yaml
# 推荐（稳定）
subtasks:
  - "temporal_flickering"
  - "motion_smoothness"

# 避免使用（可能失败）
# - "subject_consistency"  # ZeroDivisionError
# - "temporal_style"       # 内容相关的失败
```

### 5. 输出文件组织

```
outputs/
├── eval.log                           # 执行日志
├── metadata.csv                       # 原始元数据
├── processed_metadata.csv             # 处理后的元数据
├── per_video_metrics.csv              # 单视频分数
├── group_summary.csv                  # 聚合结果
├── Exp_72frames_OscStable_Head_Win.csv # 自定义实验输出
├── vbench_per_video.csv               # VBench 结果
├── vbench_results/                    # VBench 中间文件
│   ├── input_videos/                  # 软链接的视频
│   └── *_eval_results.json            # VBench 原始输出
└── figs/                              # 可视化图表
```

---

## 快速参考

### 指标解读

| 指标 | 范围 | 越好 |
|------|------|------|
| `clip_score` | 0-1 | 越高 |
| `temporal_flickering` | 0-1 | 越高 |
| `motion_smoothness` | 0-1 | 越高 |
| `flicker_mean` | 0+ | 越低 |
| `niqe_score` | 0+ | 越低 |

### 常见问题与解决方案

| 问题 | 解决方案 |
|------|----------|
| VBench ZeroDivisionError | 从 subtasks 中移除 `subject_consistency` |
| CLIP 分数为空 | 检查 `prompt_file` 路径和格式 |
| 帧填充警告 | 增加 `num_frames` 或检查视频长度 |
| CUDA 内存不足 | 将 `batch_size` 减小到 1 |

---

## 创建新实验

### 步骤 1：复制模板

```bash
cp configs/Exp_.yaml.example configs/Exp_YourExperiment.yaml
```

### 步骤 2：修改关键设置

根据你的实验编辑以下部分：

| 配置项 | 需要修改 |
|--------|----------|
| `dataset.local_video_dir` | 视频目录路径 |
| `dataset.prompt_file` | 提示词 CSV 路径 |
| `groups` | 定义实验组 |
| `protocol.num_frames` | 基于 60% 覆盖率设置 |
| `paths.experiment_output` | 自定义输出文件名 |

### 步骤 3：运行评估

```bash
python scripts/run_all.py --config configs/Exp_YourExperiment.yaml
```

### 步骤 4：查看结果

结果将保存至：
- `outputs/per_video_metrics.csv` - 单视频分数
- `outputs/group_summary.csv` - 聚合统计
- `frontend/public/data/{experiment_output}` - 用于可视化

---

*最后更新：2025-01-25*
