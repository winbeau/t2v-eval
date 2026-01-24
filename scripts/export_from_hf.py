#!/usr/bin/env python3
"""
export_from_hf.py
Export video dataset from HuggingFace Hub to local storage.

Generates metadata.csv with columns:
  - video_id: unique identifier
  - group: experiment group name
  - prompt: text prompt
  - video_path: local path to downloaded video
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str | None, base_dir: Path) -> Path | None:
    """Resolve a possibly-relative path against base_dir."""
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path)


def extract_video_path(video_data) -> str | None:
    """Try to extract a file path from HF video payloads."""
    if isinstance(video_data, dict) and "path" in video_data:
        return video_data["path"]
    if isinstance(video_data, str) and os.path.exists(video_data):
        return video_data
    return None


def infer_group_from_path(video_path: str | None, group_names: list[str]) -> str | None:
    """Infer group name from video path using known group names."""
    if not video_path:
        return None
    parts = set(Path(video_path).parts)
    for name in group_names:
        if name in parts:
            return name
    filename = Path(video_path).name
    for name in group_names:
        if name in filename:
            return name
    return None


def load_prompt_file(prompt_path: Path) -> pd.DataFrame:
    """
    Load prompts from txt/csv/tsv/jsonl.
    Supports columns: prompt, video_id, group (case-insensitive),
    plus common aliases: text/caption, id/uid/filename/path.
    """
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    ext = prompt_path.suffix.lower()
    df: pd.DataFrame
    if ext in {".jsonl", ".json"}:
        df = pd.read_json(prompt_path, lines=(ext == ".jsonl"))
    else:
        try:
            df = pd.read_csv(prompt_path, sep=None, engine="python")
        except Exception:
            lines = [
                line.strip()
                for line in prompt_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return pd.DataFrame({"prompt": lines})

    df.columns = [str(c).strip().lower() for c in df.columns]
    if "prompt" not in df.columns and len(df.columns) >= 2:
        # Likely headerless CSV where first row became header; re-read with header=None.
        if any((" " in c) or (len(c) > 24) or c.isdigit() for c in df.columns):
            df = pd.read_csv(prompt_path, header=None)
            df.columns = [str(c).strip().lower() for c in df.columns]
            df = df.rename(columns={df.columns[0]: "index", df.columns[1]: "prompt"})

    rename = {}
    if "caption" in df.columns:
        rename["caption"] = "prompt"
    if "text" in df.columns:
        rename["text"] = "prompt"
    if "id" in df.columns:
        rename["id"] = "video_id"
    if "uid" in df.columns:
        rename["uid"] = "video_id"
    if "filename" in df.columns:
        rename["filename"] = "video_id"
    if "file" in df.columns:
        rename["file"] = "video_id"
    if "path" in df.columns:
        rename["path"] = "video_path"
    if "idx" in df.columns:
        rename["idx"] = "index"
    if "no" in df.columns:
        rename["no"] = "index"
    if "num" in df.columns:
        rename["num"] = "index"
    if "number" in df.columns:
        rename["number"] = "index"
    if rename:
        df = df.rename(columns=rename)
    return df


def series_value(row: pd.Series | None, key: str) -> str | None:
    """Safe accessor for prompt file rows."""
    if row is None or key not in row:
        return None
    val = row[key]
    if pd.isna(val):
        return None
    return str(val)


def build_prompt_indices(prompt_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[int, int]]:
    """Build indices for prompt lookup."""
    prompt_index_by_id: dict[str, int] = {}
    prompt_index_by_stem: dict[str, int] = {}
    prompt_index_by_index: dict[int, int] = {}

    if "video_id" in prompt_df.columns:
        prompt_index_by_id = {
            str(row["video_id"]): idx for idx, row in prompt_df.iterrows()
        }
        for key, idx in prompt_index_by_id.items():
            try:
                prompt_index_by_index[int(key)] = idx
            except Exception:
                continue
    if "video_path" in prompt_df.columns:
        for idx, row in prompt_df.iterrows():
            stem = Path(str(row["video_path"])).stem
            if stem:
                prompt_index_by_stem[stem] = idx
    if "index" in prompt_df.columns:
        for idx, row in prompt_df.iterrows():
            try:
                prompt_index_by_index[int(row["index"])] = idx
            except Exception:
                continue

    return prompt_index_by_id, prompt_index_by_stem, prompt_index_by_index


def find_prompt_row(
    prompt_df: pd.DataFrame | None,
    prompt_index_by_id: dict[str, int],
    prompt_index_by_stem: dict[str, int],
    prompt_index_by_index: dict[int, int],
    candidate_id: str,
    video_path_hint: str | None,
    dataset_index: int | None,
    prompt_index_by_pos: dict[int, int] | None,
) -> pd.Series | None:
    """Find the prompt row using multiple strategies."""
    prompt_row = None
    if prompt_df is None:
        return None
    if prompt_index_by_id and str(candidate_id) in prompt_index_by_id:
        prompt_row = prompt_df.iloc[prompt_index_by_id[str(candidate_id)]]
    if prompt_row is None and video_path_hint and prompt_index_by_stem:
        stem = Path(video_path_hint).stem
        if stem in prompt_index_by_stem:
            prompt_row = prompt_df.iloc[prompt_index_by_stem[stem]]
    if prompt_row is None and prompt_index_by_index:
        import re
        match = re.search(r"(\d+)(?!.*\d)", str(candidate_id))
        if match:
            idx_key = int(match.group(1))
            if idx_key in prompt_index_by_index:
                prompt_row = prompt_df.iloc[prompt_index_by_index[idx_key]]
    if prompt_row is None and prompt_index_by_pos is not None and dataset_index is not None:
        if dataset_index in prompt_index_by_pos:
            prompt_row = prompt_df.iloc[prompt_index_by_pos[dataset_index]]
    return prompt_row


def export_video_bytes(video_data, output_path: Path) -> bool:
    """
    Export video data to file.
    Handles both bytes and file path formats from HF datasets.
    """
    try:
        if isinstance(video_data, bytes):
            output_path.write_bytes(video_data)
        elif isinstance(video_data, dict) and "bytes" in video_data:
            output_path.write_bytes(video_data["bytes"])
        elif isinstance(video_data, dict) and "path" in video_data:
            # Video is stored as a path reference
            import shutil
            shutil.copy(video_data["path"], output_path)
        elif isinstance(video_data, str) and os.path.exists(video_data):
            import shutil
            shutil.copy(video_data, output_path)
        else:
            logger.warning(f"Unknown video format: {type(video_data)}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to export video: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export HF dataset to local storage")
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for raw videos",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of samples (for testing)"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    config = load_config(str(config_path))
    dataset_config = config["dataset"]
    paths_config = config["paths"]

    repo_id = dataset_config["repo_id"]
    split = dataset_config.get("split", "test")
    prompt_file = dataset_config.get("prompt_file")
    default_group = dataset_config.get("default_group")
    use_local_videos = dataset_config.get("use_local_videos", False)
    local_video_dir = dataset_config.get("local_video_dir") or dataset_config.get("video_dir")

    # Setup directories
    cache_dir = resolve_path(args.output_dir or paths_config["cache_dir"], project_root)
    raw_videos_dir = cache_dir / "raw"
    raw_videos_dir.mkdir(parents=True, exist_ok=True)

    output_dir = resolve_path(paths_config["output_dir"], project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / paths_config["metadata_file"]

    prompt_df = None
    prompt_index_by_id: dict[str, int] = {}
    prompt_index_by_stem: dict[str, int] = {}
    prompt_index_by_index: dict[int, int] = {}
    prompt_index_by_pos: dict[int, int] | None = None

    if prompt_file:
        prompt_path = resolve_path(prompt_file, project_root)
        try:
            prompt_df = load_prompt_file(prompt_path)
            logger.info(f"Loaded prompt file: {prompt_path} ({len(prompt_df)} rows)")
        except Exception as e:
            logger.error(f"Failed to load prompt file: {e}")
            sys.exit(1)
        prompt_index_by_id, prompt_index_by_stem, prompt_index_by_index = build_prompt_indices(prompt_df)

    if use_local_videos:
        local_dir = resolve_path(local_video_dir, project_root)
        if local_dir is None or not local_dir.exists():
            logger.error(f"Local video dir not found: {local_video_dir}")
            sys.exit(1)

        logger.info(f"Using local videos from: {local_dir}")
        group_names = [g["name"] for g in config.get("groups", []) if isinstance(g, dict) and "name" in g]
        video_files = sorted(local_dir.rglob("*.mp4"))
        if not video_files:
            logger.error(f"No videos found under: {local_dir}")
            sys.exit(1)

        metadata_records = []
        for idx, video_path in enumerate(tqdm(video_files, desc="Indexing local videos")):
            video_path_hint = str(video_path)
            relative = video_path.relative_to(local_dir)
            group = relative.parts[0] if len(relative.parts) > 1 else None
            group = group or infer_group_from_path(video_path_hint, group_names) or default_group
            if group is None:
                logger.error("Missing group information. Provide folder structure or default_group in config.")
                sys.exit(1)

            candidate_id = video_path.stem
            prompt_row = find_prompt_row(
                prompt_df,
                prompt_index_by_id,
                prompt_index_by_stem,
                prompt_index_by_index,
                candidate_id,
                video_path_hint,
                idx,
                prompt_index_by_pos,
            )
            prompt = series_value(prompt_row, "prompt") if prompt_row is not None else None
            if prompt is None:
                logger.error(f"Missing prompt for video: {video_path}")
                sys.exit(1)

            metadata_records.append(
                {
                    "video_id": candidate_id,
                    "group": group,
                    "prompt": prompt,
                    "video_path": str(video_path),
                }
            )

        df = pd.DataFrame(metadata_records)
        df.to_csv(metadata_path, index=False)
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Total videos indexed: {len(df)}")
        logger.info("Group distribution:")
        for group, count in df["group"].value_counts().items():
            logger.info(f"  {group}: {count}")
        return

    logger.info(f"Loading dataset from HuggingFace: {repo_id}")
    logger.info(f"Split: {split}")
    hf_cache_dir = resolve_path(dataset_config.get("hf_cache_dir"), project_root) or (cache_dir / "hf")
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"HF cache dir: {hf_cache_dir}")

    try:
        dataset = load_dataset(repo_id, split=split, cache_dir=str(hf_cache_dir))
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(
            "Please check:\n"
            "  1. Dataset repo_id is correct in configs/eval.yaml\n"
            "  2. You have access to the dataset (may need `huggingface-cli login`)\n"
            "  3. Network connection is available"
        )
        sys.exit(1)

    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Validate required columns
    required_columns = ["video"]
    available_columns = dataset.column_names
    missing = [c for c in required_columns if c not in available_columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        logger.error(f"Available columns: {available_columns}")
        sys.exit(1)

    # Optional prompt file (used when prompt/group/video_id are missing in HF dataset)
    if prompt_df is not None and len(prompt_df) == len(dataset):
        prompt_index_by_pos = {idx: idx for idx in range(len(prompt_df))}
    elif prompt_df is not None and not prompt_index_by_id and not prompt_index_by_stem and not prompt_index_by_index:
        logger.warning(
            "Prompt file has no video_id/video_path/index columns and length doesn't match dataset; "
            "will not be able to align prompts."
        )

    if "prompt" not in available_columns and prompt_df is None:
        logger.error("Missing 'prompt' column and no prompt_file provided.")
        sys.exit(1)

    # Export videos and collect metadata
    metadata_records = []
    samples = dataset if args.limit is None else dataset.select(range(min(args.limit, len(dataset))))
    group_names = [g["name"] for g in config.get("groups", []) if isinstance(g, dict) and "name" in g]

    for idx, sample in enumerate(tqdm(samples, desc="Exporting videos")):
        video_data = sample["video"]
        video_path_hint = extract_video_path(video_data)
        candidate_id = sample.get("video_id")
        if not candidate_id and video_path_hint:
            candidate_id = Path(video_path_hint).stem
        if not candidate_id:
            candidate_id = f"{idx:06d}"

        prompt_row = find_prompt_row(
            prompt_df,
            prompt_index_by_id,
            prompt_index_by_stem,
            prompt_index_by_index,
            str(candidate_id),
            video_path_hint,
            idx,
            prompt_index_by_pos,
        )

        video_id = sample.get("video_id") or series_value(prompt_row, "video_id") or candidate_id
        group = sample.get("group") or series_value(prompt_row, "group")
        if group is None:
            group = infer_group_from_path(video_path_hint, group_names) or default_group
        prompt = sample.get("prompt") or series_value(prompt_row, "prompt")

        if group is None:
            logger.error("Missing group information. Provide 'group' column, prompt_file with group, or default_group in config.")
            sys.exit(1)
        if prompt is None:
            logger.error("Missing prompt information. Provide 'prompt' column or prompt_file.")
            sys.exit(1)

        # Create group subdirectory
        group_dir = raw_videos_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)

        # Export video
        video_filename = f"{video_id}.mp4"
        video_path = group_dir / video_filename

        if video_path.exists() and not args.force:
            logger.debug(f"Skipping existing video: {video_path}")
        else:
            success = export_video_bytes(video_data, video_path)
            if not success:
                logger.warning(f"Failed to export video_id={video_id}, skipping")
                continue

        metadata_records.append(
            {
                "video_id": video_id,
                "group": group,
                "prompt": prompt,
                "video_path": str(video_path),
            }
        )

    # Save metadata
    df = pd.DataFrame(metadata_records)
    df.to_csv(metadata_path, index=False)
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info(f"Total videos exported: {len(df)}")

    # Print group distribution
    logger.info("Group distribution:")
    for group, count in df["group"].value_counts().items():
        logger.info(f"  {group}: {count}")


if __name__ == "__main__":
    main()
