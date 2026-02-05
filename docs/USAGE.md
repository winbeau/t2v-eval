# T2V-Eval 简要使用说明

## 0. 环境与依赖（常见问题修复）
- `ModuleNotFoundError: yaml/pandas/PIL/decord/cv2`：
  - 运行 `uv sync` 后仍缺依赖时，补装：`uv pip install PyYAML pandas pillow decord opencv-python`
  - 服务器建议用无头版：`opencv-python-headless`
- `ffmpeg` 不存在（预处理写视频失败）：
  - Debian/Ubuntu：`apt-get update && apt-get install -y ffmpeg`
  - Conda：`conda install -y -c conda-forge ffmpeg`
- 编译失败 `Python.h` 缺失（如 `cryptacular`）：
  - Debian/Ubuntu：`apt-get install -y python3.10-dev build-essential`
- HF 访问报 401：
  - 确认 repo_id，例如：`kv-compression/AdaHead`
  - 登录：`huggingface-cli login` 或传 `--token`
- 本地数据路径变更：
  - 同步更新 `configs/*.yaml` 中的 `local_video_dir` / `prompt_file`

## 前置准备（推荐本地下载 HF 数据集）
1. 克隆并初始化子模块
```bash
git clone --recurse-submodules <repo>
git submodule update --init --recursive
```
2. 创建并同步依赖（uv）
```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
uv pip install -r third_party/VBench/requirements.txt
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
