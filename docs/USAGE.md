# T2V-Eval 简要使用说明

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 ffmpeg（预处理写视频需要）
apt update && apt install -y ffmpeg

# 安装 Python 头文件与编译工具（避免 cryptacular 等编译失败）
apt install -y python3.10-dev build-essential
```

## 0. 环境与依赖（常见问题修复）
- `ModuleNotFoundError: yaml/pandas/PIL/decord/cv2`：
  - 运行 `uv sync` 后仍缺依赖时，补装：`uv pip install PyYAML pandas pillow decord opencv-python`
  - 服务器建议用无头版：`opencv-python-headless`
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
1. 在 `configs/Exp_OscStable_Head_Window.yaml` 中启用本地模式：
```yaml
dataset:
  use_local_videos: true
  local_video_dir: "hf/AdaHead/Exp_OscStable_Head_Window"
  prompt_file: "hf/AdaHead/Exp_OscStable_Head_Window/prompts.csv"
```
2. 运行评测：
```bash
python scripts/run_all.py --config configs/Exp_OscStable_Head_Window.yaml
```

## 常见提示
- 若提示缺少依赖（如 `pandas`/`pyyaml`），在虚拟环境中补装：`uv pip install pandas PyYAML`。
- 需要跳过 VBench 时可加 `--skip-vbench`。
