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
uv run python scripts/download_hf_subdir.py \
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

groups:
  - name: "k6_deep_forcing"           # 执行主键（通常与文件夹名一致）
    alias: "K6 Deep Forcing"          # 输出 CSV 里的组名（可选）
  - name: "k7_native_self_forcing_static21_sink1"
    alias: "K7 Self Forcing"
```
说明：`groups[].alias` 仅影响输出 CSV 的 `group` 展示名；未配置 alias 时默认使用原组名（本地模式下通常就是文件夹名）。
2. 运行核心评测流程（含导出/预处理/CLIP(or VQA)/Flicker/NIQE）并开启并行预处理：
```bash
uv run python scripts/run_all.py \
    --config configs/Exp-K_StaOscCompression.yaml \
    --preprocess-workers 48 \
    --ffmpeg-threads 1
```

等价推荐入口（非别名）：
```bash
uv run python scripts/run_eval_core.py \
    --config configs/Exp-K_StaOscCompression.yaml \
    --preprocess-workers 48 \
    --ffmpeg-threads 1
```

仅运行预处理（便于单独压测）：
```bash
uv run python scripts/preprocess_videos.py \
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

3. 运行 VBench（`-H200` 12 维 YAML）

常用规则：
- `-H200` 这类 12 维 YAML 已经只保留 temporal 维度，不需要再手动 `--skip color,object_class,multiple_objects,spatial_relationship`
- 4 卡运行时直接设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`
- 子集运行后会同时写：
  - 本次子集总 CSV
  - `outputs/<exp>/vbench_group_runs/` 下的小组级 CSV
- 最终合并时只认小组级 CSV，并严格检查是否全覆盖 YAML 中的全部组

1. 4 卡运行全部：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run python scripts/run_vbench.py \
    --config configs/Exp_pyramid_forcing_groups-H200.yaml \
    --force
```

2. 4 卡运行子集：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
uv run python scripts/run_vbench.py \
    --config configs/Exp_pyramid_forcing_groups-H200.yaml \
    --skip-groups g3,g4,g5,g6 \
    --vbench-output vbench_Exp_pyramid_forcing_groups-H200_g1_g2.csv \
    --force
```

3. 合并所有子小组：
```bash
scripts/merge_vbench_all.sh \
    --config configs/Exp_pyramid_forcing_groups-H200.yaml \
    --force
```

说明：
- `--skip-groups` 只接受 YAML 中 `groups[].name` 的精确名称
- 子集模式下必须显式传 `--vbench-output`
- 若某个小组缓存已经存在而未传 `--force`，脚本会直接报错，避免覆盖旧结果
- `merge_vbench_all.sh` 会严格检查小组缓存是否全覆盖；缺组、重复组、额外脏 CSV 都会直接失败
- 合并完成后会自动复制最终 VBench CSV 和 group summary 到 `frontend/public/data/`

4. 官方 VBench-Long 直跑 16 维度命令（可选）：
```bash
cd third_party/VBench
uv run python vbench2_beta_long/eval_long.py \
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
  uv run python scripts/run_vbench.py \
      --config configs/Exp-C_OscHead_RadicalKV_vbench.yaml \
      --skip-on-error \
      --skip color,object_class,multiple_objects,spatial_relationship \
      --force
  ```
