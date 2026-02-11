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
- `ffmpeg` 不存在：预处理写视频会失败（安装命令见上）
- 编译失败 `Python.h` 缺失：安装 Python 头文件与编译工具（命令见上）
- HF 访问报 401：
  - 确认 repo_id，例如：`kv-compression/AdaHead`
  - 登录：`huggingface-cli login` 或传 `--token`
- 本地数据路径变更：
  - 同步更新 `configs/*.yaml` 中的 `local_video_dir` / `prompt_file`

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
  prompt_file: "hf/AdaHead/Exp_OscStable_Head_Window/prompts.csv"
```
2. 仅跑 VBench-Long 16 维度（当前配置）：
```bash
# 下载并准备好本地数据集后即可直接运行（不要求先预处理）
python scripts/run_vbench.py --config configs/Exp_OscStable_Head_Window_vbench16.yaml --force
```
运行结束后会自动把结果同步到 `frontend/public/data/`（并更新 `manifest.json`）。
脚本会在开始时只做一次切片预处理，后续 16 个维度复用 `split_clip`，不再重复预处理。

多卡并行（例如 4×L40）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/run_vbench.py --config configs/Exp_OscStable_Head_Window_vbench16.yaml --force
```
说明：脚本会自动按“维度”均分到可见 GPU（如 16 维 / 4 卡 => 每卡 4 维，全视频），最终由 CPU 聚合输出 `outputs/Exp_OscStable_Head_Window_vbench16/vbench_per_video.csv`。

如需手动关闭自动多卡：
```bash
python scripts/run_vbench.py --config configs/Exp_OscStable_Head_Window_vbench16.yaml --force --no-auto-multi-gpu
```

3. 官方 VBench-Long 直跑 16 维度命令（可选）：
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
- VBench 现为独立步骤，请单独运行：`python scripts/run_vbench.py --config <your_config>.yaml --force`。
