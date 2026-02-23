# T2V-Eval 简要使用说明

## 0. Node.js / pnpm 安装（前端必需）

> 说明：为避免系统仓库里的 `npm` 版本过旧，建议直接安装 Node.js LTS（自带较新 `npm`）。

```bash
# 安装 Node.js LTS（Ubuntu/Debian，NodeSource）
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs
node -v
npm -v

# 安装并激活 pnpm（通过 Corepack）
corepack enable
corepack prepare pnpm@latest --activate
pnpm -v
```

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 ffmpeg（预处理写视频需要）
apt update && apt install -y ffmpeg

# 安装 Python 头文件与编译工具（避免 cryptacular 等编译失败）
apt install -y python3.10-dev build-essential
```

## 1. 环境与依赖（常见问题修复）
- VBench 16 维推荐稳定栈（已在 `pyproject.toml` 固定）：
  - `torch==2.1.2`
  - `torchvision==0.16.2`
  - `timm<=1.0.12`
  - 执行：`uv sync`
- 16维里含检测相关维度时，安装 `detectron2`：
  - `uv pip install --no-build-isolation "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"`
- `ModuleNotFoundError: yaml/pandas/PIL/decord/cv2`：
  - 运行 `uv sync` 后仍缺依赖时，补装：`uv pip install PyYAML pandas pillow decord opencv-python`
  - 服务器建议用无头版：`opencv-python-headless`
- `No module named 'clip'`（VBench 的 `background_consistency` / `aesthetic_quality` 常见）：
  - 注意：`clip` 不是 `pkg_resources`；`pkg_resources` 来自 `setuptools`
  - 先装：`uv pip install -U setuptools`
  - 先确认在当前虚拟环境内安装：`uv pip install openai-clip`
  - 若镜像源没有该包，使用：`python -m pip install git+https://github.com/openai/CLIP.git`
- `No module named 'pkg_resources'`：
  - 安装/升级 `setuptools`：`uv pip install -U setuptools`
- `ImportError: PyAV is not installed`：
  - 安装：`uv pip install av`
- `No module named 'fairscale'`（16维里的 `scene` 常见）：
  - 安装：`uv pip install fairscale`
- `No module named 'detectron2'`（`object_class`/`multiple_objects`/`spatial_relationship`/`color`）：
  - 先装构建后端：`uv pip install hatchling`
  - 再装：`uv pip install --no-build-isolation "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"`
- `ffmpeg` 不存在：预处理写视频会失败（安装命令见上）
- 编译失败 `Python.h` 缺失：安装 Python 头文件与编译工具（命令见上）
- HF 访问报 401：
  - 确认 repo_id，例如：`kv-compression/AdaHead`
  - 登录：`huggingface-cli login` 或传 `--token`
- 本地数据路径变更：
  - 同步更新 `configs/*.yaml` 中的 `local_video_dir` / `prompt_file` / `prompt_files_by_group`

## 前置准备（推荐本地下载 HF 数据集）
1. 注册子模块（首次克隆后执行）
```bash
git submodule update --init --recursive
```
2. 创建并同步依赖（uv）
```bash
uv sync
source .venv/bin/activate
```
3. 登录 HuggingFace（私有/受限数据集需要）
```bash
huggingface-cli login
```

## 本地下载 HF 数据集（推荐）
使用脚本按子目录下载数据，避免每次跑都拉取网络数据。
```bash
python scripts/download_hf_subdir.py \
  --repo-id kv-compression/AdaHead \
  --subdir Exp_OscStable_Head_Window \
  --output-dir hf/AdaHead/Exp_OscStable_Head_Window
```

## 配置并运行
1. 使用 16 维 VBench 配置文件（已内置本地路径）：
```yaml
dataset:
  use_local_videos: true
  local_video_dir: "hf/AdaHead/Exp_OscStable_Head_Window"
  prompt_files_by_group:
    group_a: "hf/AdaHead/Exp_OscStable_Head_Window/group_a/prompts.csv"
    group_b: "hf/AdaHead/Exp_OscStable_Head_Window/group_b/prompts.csv"
  # 可选：全局回退（当某组未配置或组内未命中时使用）
  prompt_file: "hf/AdaHead/Exp_OscStable_Head_Window/prompts.csv"
```
2. 运行核心评测流程（含导出/预处理/CLIP(or VQA)/Flicker/NIQE）并开启并行预处理：
```bash
python scripts/run_all.py \
    --config configs/Exp-K_StaOscCompression.yaml \
    --preprocess-workers 48 \
    --ffmpeg-threads 1
```

等价推荐入口（非别名）：
```bash
python scripts/run_eval_core.py \
    --config configs/Exp-K_StaOscCompression.yaml \
    --preprocess-workers 48 \
    --ffmpeg-threads 1
```

仅运行预处理（便于单独压测）：
```bash
python scripts/preprocess_videos.py \
    --config configs/Exp-K_StaOscCompression.yaml \
    --preprocess-workers 48 \
    --ffmpeg-threads 1 \
    --force
```

Core 预处理相关 CLI 参数（`run_all.py` / `run_eval_core.py`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--preprocess-workers` | `1` | 预处理进程数（显式设置，不做自动探测） |
| `--ffmpeg-threads` | `1` | 每个预处理进程内 ffmpeg 线程数 |

> 建议：大机器上优先使用 `--preprocess-workers 48 --ffmpeg-threads 1`，避免总线程数过高导致争用。

3. 运行 VBench-Long 评测（**推荐 12 维，跳过 4 个 GrIT 慢维度**）：
`run_vbench.py` 会在本地视频模式下自动做一次 `split_clip` 预处理（仅 rank0），后续维度复用缓存。
同时会在评测前自动预取并校验 CLIP 权重（`ViT-B-32.pt` / `ViT-L-14.pt`），若检测到损坏会自动修复，避免 `invalid load key` 类报错。
`configs/Exp-C_OscHead_RadicalKV_vbench.yaml` 已内置：
```yaml
metrics:
  vbench:
    preprocess_workers: 48
```
若 `use_semantic_splitting: true`，当前会回退到 VBench 原生预处理流程。

直接运行（不改你原命令）：
```bash
python scripts/run_vbench.py \
    --config configs/Exp-C_OscHead_RadicalKV_vbench.yaml \
    --skip-on-error \
    --skip color,object_class,multiple_objects,spatial_relationship \
    --force
```

如需临时覆盖并发度（CLI 优先级高于 YAML）：
```bash
python scripts/run_vbench.py \
    --config configs/Exp-C_OscHead_RadicalKV_vbench.yaml \
    --preprocess-workers 48 \
    --skip-on-error \
    --skip color,object_class,multiple_objects,spatial_relationship \
    --force
```

> **说明**：`color`、`object_class`、`multiple_objects`、`spatial_relationship` 依赖 GrIT 密集描述模型（每帧 beam search 文本生成约 6s），这 4 维占总时间 80% 以上。跳过后 4 卡约 20 分钟完成，其余 12 维不受影响。

可选资产预取参数（默认均开启）：
```yaml
metrics:
  vbench:
    prefetch_assets: true
    verify_asset_sha256: true
    repair_corrupted_assets: true
    # 可选：多卡聚合时等待各 rank partial 的超时与轮询间隔
    partial_collect_timeout_sec: 43200
    partial_collect_poll_sec: 2.0
    # 可选：论文对齐口径（Deep-Forcing 8维）
    comparison_profile: "deep_forcing_8d"
    # 0-1 指标转百分制（x100）；imaging_quality 保持原量纲
    scale_to_percent:
      - "dynamic_degree"
      - "motion_smoothness"
      - "overall_consistency"
      - "aesthetic_quality"
      - "subject_consistency"
      - "background_consistency"
    # 额外导出 profile 摘要（不影响主 group_summary.csv）
    profile_output: "group_summary_deep_forcing_8d.csv"
```
也可用 CLI 临时关闭：
`--no-prefetch-assets` / `--no-verify-asset-sha256` / `--no-repair-corrupted-assets`

如需运行全部 16 维（极慢，4×L40 约数小时）：
```bash
python scripts/run_vbench.py \
    --config configs/Exp-C_OscHead_RadicalKV_vbench.yaml \
    --skip-on-error \
    --force
```

运行结束后会自动把结果同步到 `frontend/public/data/`（并更新 `manifest.json`）。
脚本会在开始时只做一次切片预处理，后续维度复用 `split_clip`，不再重复预处理。

诊断“实现问题 vs 口径问题”（只读）：
```bash
python scripts/diagnose_vbench_alignment.py \
    --output-dir outputs/Exp-K_StaOscCompression \
    --config configs/Exp-K_StaOscCompression.yaml \
    --pair overall_consistency,temporal_style \
    --report-out outputs/Exp-K_StaOscCompression/alignment_report.md
```
该脚本会同时检查：
1) `vbench_*.csv` 聚合后的统计；
2) `vbench_results/*_eval_results.json` 原始子任务输出；
3) `overall_consistency` 与 `temporal_style` 的逐样本一致度；
4) `*_full_info.json` 输入（`prompt_en`/`video_list`）是否一致；
5) `third_party/VBench/vbench` 中两个维度实现的源码相似度。

若出现 `overall_consistency` 显著偏低，同时日志显示大量 `fallback_video_id`，
优先检查 `dataset.prompt_files_by_group` 是否覆盖所有组并指向各组 `prompts.csv`。

### CLI 参数说明

| 参数 | 说明 |
|------|------|
| `--config` | YAML 配置文件路径 |
| `--force` | 覆盖已有结果，强制重新计算 |
| `--skip-on-error` | 某个维度失败时跳过而非终止，聚合已成功的部分结果 |
| `--skip <dims>` | 跳过指定维度（逗号分隔），如 `--skip color,object_class` |
| `--preprocess-workers` | VBench-Long 一次性切片预处理进程数（CLI 覆盖 `metrics.vbench.preprocess_workers`） |
| `--no-prefetch-assets` | 关闭评测前 CLIP 权重预取与校验 |
| `--no-verify-asset-sha256` | 关闭 CLIP 权重 SHA256 校验 |
| `--no-repair-corrupted-assets` | 检测到损坏权重时不自动修复（直接报错） |
| `--no-auto-multi-gpu` | 禁用自动多卡并行 |

多卡并行（例如 4×L40）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/run_vbench.py \
    --config configs/Exp-C_OscHead_RadicalKV_vbench.yaml \
    --skip-on-error \
    --skip color,object_class,multiple_objects,spatial_relationship \
    --force
```
说明：脚本会自动按"维度"均分到可见 GPU（如 12 维 / 4 卡 => 每卡 3 维，全视频），最终由 CPU 聚合输出 `outputs/<experiment>/vbench_per_video.csv`。

4. 官方 VBench-Long 直跑 16 维度命令（可选）：
```bash
cd third_party/VBench
python vbench2_beta_long/eval_long.py \
  --videos_path <VIDEO_DIR> \
  --dimension subject_consistency background_consistency temporal_flickering motion_smoothness temporal_style appearance_style scene object_class multiple_objects spatial_relationship human_action color overall_consistency dynamic_degree imaging_quality aesthetic_quality \
  --mode long_custom_input \
  --dev_flag
```

## 推理后启动前端（Vite）

```bash
cd frontend
pnpm install
pnpm exec vite --host 0.0.0.0 --port 5173 --strictPort
```

等价写法：`pnpm dev`（已在 `vite.config.ts` 固定 `host=0.0.0.0`、`port=5173`）。

## 常见提示
- 若提示缺少依赖（如 `pandas`/`pyyaml`），在虚拟环境中补装：`uv pip install pandas PyYAML`。
- VBench 现为独立步骤，推荐用法：
  ```bash
  python scripts/run_vbench.py \
      --config configs/Exp-C_OscHead_RadicalKV_vbench.yaml \
      --skip-on-error \
      --skip color,object_class,multiple_objects,spatial_relationship \
      --force
  ```
