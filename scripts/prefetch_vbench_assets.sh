#!/usr/bin/env bash
set -euo pipefail

cache_dir="${VBENCH_CACHE_DIR:-$HOME/.cache/vbench}"
dry_run=0

usage() {
  cat <<'EOF'
Usage:
  scripts/prefetch_vbench_assets.sh [--cache-dir DIR] [--dry-run]

Prefetch all model assets that may be downloaded lazily by `scripts/run_vbench.py`.

Options:
  --cache-dir DIR  Override VBench cache dir. Defaults to $VBENCH_CACHE_DIR or ~/.cache/vbench
  --dry-run        Print planned actions without downloading
  -h, --help       Show this help

This script prepares assets for all VBench-Long dimensions supported in this repo:
  - subject_consistency
  - background_consistency
  - temporal_flickering
  - motion_smoothness
  - temporal_style
  - appearance_style
  - scene
  - object_class
  - multiple_objects
  - spatial_relationship
  - human_action
  - color
  - overall_consistency
  - dynamic_degree
  - imaging_quality
  - aesthetic_quality
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --cache-dir" >&2; exit 2; }
      cache_dir="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_file() {
  local url="$1"
  local dest="$2"
  local dest_dir
  dest_dir="$(dirname "$dest")"

  mkdir -p "$dest_dir"

  if [[ -f "$dest" ]]; then
    echo "[skip] $dest"
    return 0
  fi

  echo "[download] $url -> $dest"
  if [[ $dry_run -eq 1 ]]; then
    return 0
  fi

  if have_cmd wget; then
    wget -O "$dest" "$url"
  elif have_cmd curl; then
    curl -L "$url" -o "$dest"
  else
    echo "Need either wget or curl to download assets." >&2
    exit 1
  fi
}

download_to_dir() {
  local url="$1"
  local dir="$2"
  local filename="${3:-$(basename "$url")}"
  download_file "$url" "$dir/$filename"
}

ensure_git_clone() {
  local repo_url="$1"
  local dest_dir="$2"

  if [[ -d "$dest_dir/.git" ]]; then
    echo "[skip] git repo $dest_dir"
    return 0
  fi

  if [[ -e "$dest_dir" ]]; then
    echo "Path exists but is not a git repo: $dest_dir" >&2
    exit 1
  fi

  echo "[clone] $repo_url -> $dest_dir"
  if [[ $dry_run -eq 1 ]]; then
    return 0
  fi

  git clone --depth 1 "$repo_url" "$dest_dir"
}

ensure_raft_model() {
  local raft_ckpt="$cache_dir/raft_model/models/raft-things.pth"
  local zip_path="$cache_dir/raft_model/models.zip"

  if [[ -f "$raft_ckpt" ]]; then
    echo "[skip] $raft_ckpt"
    return 0
  fi

  mkdir -p "$cache_dir/raft_model"
  echo "[download] RAFT archive -> $zip_path"
  echo "[extract] $zip_path -> $cache_dir/raft_model"
  if [[ $dry_run -eq 1 ]]; then
    return 0
  fi

  if have_cmd wget; then
    wget -O "$zip_path" "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
  elif have_cmd curl; then
    curl -L "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" -o "$zip_path"
  else
    echo "Need either wget or curl to download RAFT assets." >&2
    exit 1
  fi

  unzip -o "$zip_path" -d "$cache_dir/raft_model"
  rm -f "$zip_path"
}

if [[ $dry_run -eq 0 ]]; then
  have_cmd git || { echo "Missing required command: git" >&2; exit 1; }
  have_cmd unzip || { echo "Missing required command: unzip" >&2; exit 1; }
fi

echo "Using VBENCH cache dir: $cache_dir"

# Shared CLIP checkpoints.
download_to_dir \
  "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt" \
  "$cache_dir/clip_model" \
  "ViT-B-32.pt"
download_to_dir \
  "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" \
  "$cache_dir/clip_model" \
  "ViT-L-14.pt"

# human_action
download_to_dir \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth" \
  "$cache_dir/umt_model" \
  "l16_ptk710_ftk710_ftk400_f16_res224.pth"

# motion_smoothness
download_to_dir \
  "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth" \
  "$cache_dir/amt_model" \
  "amt-s.pth"

# dynamic_degree
ensure_raft_model

# subject_consistency
ensure_git_clone \
  "https://github.com/facebookresearch/dino" \
  "$cache_dir/dino_model/facebookresearch_dino_main"
download_to_dir \
  "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth" \
  "$cache_dir/dino_model" \
  "dino_vitbase16_pretrain.pth"

# aesthetic_quality
download_to_dir \
  "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth" \
  "$cache_dir/aesthetic_model/emb_reader" \
  "sa_0_4_vit_l_14_linear.pth"

# imaging_quality
download_to_dir \
  "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth" \
  "$cache_dir/pyiqa_model" \
  "musiq_spaq_ckpt-358bb6af.pth"

# GrIT-backed dimensions: object_class / multiple_objects / spatial_relationship / color
download_to_dir \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth" \
  "$cache_dir/grit_model" \
  "grit_b_densecap_objectdet.pth"

# scene
download_to_dir \
  "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth" \
  "$cache_dir/caption_model" \
  "tag2text_swin_14m.pth"

# temporal_style / overall_consistency
download_to_dir \
  "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth" \
  "$cache_dir/ViCLIP" \
  "ViClip-InternVid-10M-FLT.pth"
download_to_dir \
  "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz" \
  "$cache_dir/ViCLIP" \
  "bpe_simple_vocab_16e6.txt.gz"

echo "VBench asset prefetch complete."
