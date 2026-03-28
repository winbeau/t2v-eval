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

hf_cli_cmd() {
  if have_cmd hf; then
    echo "hf"
    return 0
  fi
  if have_cmd huggingface-cli; then
    echo "huggingface-cli"
    return 0
  fi
  return 1
}

is_hf_resolve_url() {
  local url="$1"
  [[ "$url" =~ ^https://huggingface\.co/ ]]
}

download_hf_file() {
  local url="$1"
  local dest="$2"
  local dest_dir repo_id revision filename repo_type cli

  if ! cli="$(hf_cli_cmd)"; then
    return 1
  fi

  dest_dir="$(dirname "$dest")"
  if [[ "$url" =~ ^https://huggingface\.co/spaces/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$ ]]; then
    repo_type="space"
    repo_id="${BASH_REMATCH[1]}"
    revision="${BASH_REMATCH[2]}"
    filename="${BASH_REMATCH[3]}"
  elif [[ "$url" =~ ^https://huggingface\.co/datasets/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$ ]]; then
    repo_type="dataset"
    repo_id="${BASH_REMATCH[1]}"
    revision="${BASH_REMATCH[2]}"
    filename="${BASH_REMATCH[3]}"
  elif [[ "$url" =~ ^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$ ]]; then
    repo_type="model"
    repo_id="${BASH_REMATCH[1]}"
    revision="${BASH_REMATCH[2]}"
    filename="${BASH_REMATCH[3]}"
  else
    return 1
  fi

  echo "[hf] $repo_type:$repo_id@$revision $filename -> $dest"
  if [[ $dry_run -eq 1 ]]; then
    return 0
  fi

  mkdir -p "$dest_dir"
  "$cli" download \
    --repo-type "$repo_type" \
    --revision "$revision" \
    --local-dir "$dest_dir" \
    "$repo_id" \
    "$filename"

  if [[ ! -f "$dest" ]]; then
    echo "HF download finished but expected file is missing: $dest" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local dest="$2"
  local dest_dir
  dest_dir="$(dirname "$dest")"

  if [[ -f "$dest" ]]; then
    echo "[skip] $dest"
    return 0
  fi

  echo "[download] $url -> $dest"
  if [[ $dry_run -eq 1 ]]; then
    return 0
  fi

  mkdir -p "$dest_dir"
  if is_hf_resolve_url "$url"; then
    if download_hf_file "$url" "$dest"; then
      return 0
    fi
  fi

  if have_cmd wget; then
    wget -c -O "$dest" "$url"
  elif have_cmd curl; then
    curl -L -C - "$url" -o "$dest"
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

  echo "[download] RAFT archive -> $zip_path"
  echo "[extract] $zip_path -> $cache_dir/raft_model"
  if [[ $dry_run -eq 1 ]]; then
    return 0
  fi

  mkdir -p "$cache_dir/raft_model"
  if have_cmd wget; then
    wget -c -O "$zip_path" "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
  elif have_cmd curl; then
    curl -L -C - "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip" -o "$zip_path"
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
