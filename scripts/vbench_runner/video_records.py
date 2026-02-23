"""
Video record loading, metadata management, and frontend output.
"""

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

try:
    from .env import PROJECT_ROOT, logger, resolve_path
except ImportError:
    from vbench_runner.env import PROJECT_ROOT, logger, resolve_path


def get_video_list(metadata_path: Path) -> list:
    """Get list of video paths from metadata."""
    df = pd.read_csv(metadata_path)
    return df.to_dict("records")


def infer_group_from_path(video_path: Path, group_names: list[str]) -> str | None:
    """Infer group by folder name or filename."""
    if group_names:
        parts = set(video_path.parts)
        for name in group_names:
            if name in parts:
                return name
        filename = video_path.name
        for name in group_names:
            if name in filename:
                return name
    if len(video_path.parts) > 1:
        return video_path.parent.name
    return None


def load_prompt_map(prompt_path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Load prompt mapping by video_id/stem and ordered fallback list."""
    suffix = prompt_path.suffix.lower()
    try:
        if suffix == ".tsv":
            prompt_df = pd.read_csv(prompt_path, sep="\t")
        else:
            prompt_df = pd.read_csv(prompt_path)
    except Exception:
        prompt_df = pd.read_csv(prompt_path, sep=None, engine="python")
    prompt_df.columns = [str(c).strip().lower() for c in prompt_df.columns]

    rename = {}
    if "caption" in prompt_df.columns:
        rename["caption"] = "prompt"
    if "text" in prompt_df.columns:
        rename["text"] = "prompt"
    if "id" in prompt_df.columns:
        rename["id"] = "video_id"
    if "uid" in prompt_df.columns:
        rename["uid"] = "video_id"
    if "filename" in prompt_df.columns:
        rename["filename"] = "video_id"
    if "path" in prompt_df.columns:
        rename["path"] = "video_path"
    if rename:
        prompt_df = prompt_df.rename(columns=rename)

    if "prompt" not in prompt_df.columns and len(prompt_df.columns) >= 1:
        prompt_df = prompt_df.rename(columns={prompt_df.columns[-1]: "prompt"})

    prompt_by_id = {}
    prompt_by_stem = {}
    ordered_prompts = []

    if "prompt" in prompt_df.columns:
        ordered_prompts = [str(v) for v in prompt_df["prompt"].fillna("").tolist()]
        if "video_id" in prompt_df.columns:
            for _, row in prompt_df.iterrows():
                video_id = str(row.get("video_id", "")).strip()
                prompt = str(row.get("prompt", "")).strip()
                if video_id and prompt:
                    prompt_by_id[video_id] = prompt
        if "video_path" in prompt_df.columns:
            for _, row in prompt_df.iterrows():
                video_path = str(row.get("video_path", "")).strip()
                prompt = str(row.get("prompt", "")).strip()
                if video_path and prompt:
                    prompt_by_stem[Path(video_path).stem] = prompt

    return prompt_by_id, prompt_by_stem, ordered_prompts


def _load_prompt_source(prompt_path_str: str | None, label: str) -> dict[str, object]:
    """Load one prompt source and return normalized lookup structures."""
    source = {
        "by_id": {},
        "by_stem": {},
        "ordered": [],
        "path": None,
    }
    if not prompt_path_str:
        return source

    prompt_path = resolve_path(prompt_path_str)
    if prompt_path and prompt_path.exists():
        by_id, by_stem, ordered = load_prompt_map(prompt_path)
        source = {
            "by_id": by_id,
            "by_stem": by_stem,
            "ordered": ordered,
            "path": prompt_path,
        }
        logger.info("Loaded %s prompts from: %s (%d rows)", label, prompt_path, len(ordered))
        return source

    logger.warning("%s prompt file not found, skipping: %s", label, prompt_path_str)
    return source


def _pick_prompt(
    *,
    candidate_id: str,
    group_name: str,
    global_index: int,
    group_index: int,
    group_prompt_sources: dict[str, dict[str, object]],
    global_prompt_source: dict[str, object],
) -> tuple[str, str]:
    """Resolve prompt and return (prompt, source_tag)."""
    group_source = group_prompt_sources.get(group_name)
    if group_source:
        by_id = group_source.get("by_id", {})
        by_stem = group_source.get("by_stem", {})
        ordered = group_source.get("ordered", [])

        prompt = by_id.get(candidate_id)
        if prompt:
            return str(prompt), "group_id"

        prompt = by_stem.get(candidate_id)
        if prompt:
            return str(prompt), "group_stem"

        if group_index < len(ordered):
            prompt = str(ordered[group_index]).strip()
            if prompt:
                return prompt, "group_positional"

    by_id = global_prompt_source.get("by_id", {})
    by_stem = global_prompt_source.get("by_stem", {})
    ordered = global_prompt_source.get("ordered", [])

    prompt = by_id.get(candidate_id)
    if prompt:
        return str(prompt), "global_id"

    prompt = by_stem.get(candidate_id)
    if prompt:
        return str(prompt), "global_stem"

    if global_index < len(ordered):
        prompt = str(ordered[global_index]).strip()
        if prompt:
            return prompt, "global_positional"

    return candidate_id, "fallback_video_id"


def build_video_list_from_local_dataset(config: dict) -> list:
    """Build video records from local dataset folders (no preprocess required)."""
    dataset_config = config.get("dataset", {})
    local_video_dir = dataset_config.get("local_video_dir") or dataset_config.get("video_dir")
    local_dir = resolve_path(local_video_dir)
    if local_dir is None or not local_dir.exists():
        raise FileNotFoundError(f"Local video dir not found: {local_video_dir}")

    groups = config.get("groups", [])
    group_names = [g["name"] for g in groups if isinstance(g, dict) and "name" in g]
    default_group = dataset_config.get("default_group")

    global_prompt_source = _load_prompt_source(
        dataset_config.get("prompt_file"),
        label="global",
    )

    group_prompt_sources: dict[str, dict[str, object]] = {}
    prompt_files_by_group = dataset_config.get("prompt_files_by_group") or {}
    if prompt_files_by_group and not isinstance(prompt_files_by_group, dict):
        logger.warning("Ignoring dataset.prompt_files_by_group because it is not a mapping.")
        prompt_files_by_group = {}
    for group_name, prompt_path_str in prompt_files_by_group.items():
        if not isinstance(group_name, str) or not group_name.strip():
            logger.warning("Skipping invalid group key in dataset.prompt_files_by_group: %r", group_name)
            continue
        group_prompt_sources[group_name] = _load_prompt_source(
            prompt_path_str,
            label=f"group[{group_name}]",
        )

    video_files = sorted(local_dir.rglob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No mp4 videos found under: {local_dir}")

    records = []
    group_indices: Counter[str] = Counter()
    prompt_sources_total: Counter[str] = Counter()
    prompt_sources_by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for idx, video_path in enumerate(video_files):
        candidate_id = video_path.stem

        relative = video_path.relative_to(local_dir)
        group = relative.parts[0] if len(relative.parts) > 1 else None
        if not group or (group_names and group not in group_names):
            group = infer_group_from_path(video_path, group_names)
        if group_names and group not in group_names:
            continue
        if group is None:
            group = default_group or "default"

        group_idx = group_indices[group]
        group_indices[group] += 1
        prompt, source_tag = _pick_prompt(
            candidate_id=candidate_id,
            group_name=group,
            global_index=idx,
            group_index=group_idx,
            group_prompt_sources=group_prompt_sources,
            global_prompt_source=global_prompt_source,
        )
        prompt_sources_total[source_tag] += 1
        prompt_sources_by_group[group][source_tag] += 1

        records.append(
            {
                "video_id": candidate_id,
                "group": group,
                "prompt": prompt,
                "video_path": str(video_path),
            }
        )

    if not records:
        raise RuntimeError(
            "No usable videos matched configured groups under local dataset directory."
        )
    logger.info("Prompt source summary: %s", dict(prompt_sources_total))
    if prompt_sources_total.get("fallback_video_id", 0) > 0:
        logger.warning(
            "Prompt fallback to video_id occurred for %d videos.",
            prompt_sources_total["fallback_video_id"],
        )
    ordered_group_names = group_names + sorted(
        name for name in prompt_sources_by_group.keys() if name not in set(group_names)
    )
    for group_name in ordered_group_names:
        if group_name in prompt_sources_by_group:
            logger.info(
                "Prompt source by group [%s]: %s",
                group_name,
                dict(prompt_sources_by_group[group_name]),
            )
    return records


def load_video_records_for_vbench(config: dict, output_dir: Path, paths_config: dict) -> list:
    """Load records from processed metadata / raw metadata / local dataset."""
    processed_metadata = output_dir / paths_config.get(
        "processed_metadata", "processed_metadata.csv"
    )
    metadata_file = output_dir / paths_config.get("metadata_file", "metadata.csv")

    if processed_metadata.exists():
        logger.info(f"Using processed metadata: {processed_metadata}")
        return get_video_list(processed_metadata)

    if metadata_file.exists():
        logger.info(f"Using raw metadata: {metadata_file}")
        return get_video_list(metadata_file)

    logger.info("Metadata not found, building video list from local dataset paths...")
    return build_video_list_from_local_dataset(config)


def ensure_unique_video_ids(video_records: list[dict], config: dict) -> list[dict]:
    """
    Ensure video_id is globally unique across groups.

    Some experiments store videos as {group}/video_000.mp4, causing duplicate
    video_id values across groups. VBench parsing/merge expects unique ids.
    """
    if not video_records:
        return video_records

    groups = config.get("groups", [])
    group_names = [g["name"] for g in groups if isinstance(g, dict) and "name" in g]

    normalized: list[dict] = []
    for record in video_records:
        rec = dict(record)
        video_id = str(rec.get("video_id", "")).strip()
        video_path = str(rec.get("video_path", "")).strip()
        group = str(rec.get("group", "")).strip()

        if not video_id and video_path:
            video_id = Path(video_path).stem
        if not group and video_path:
            group = infer_group_from_path(Path(video_path), group_names) or ""

        rec["video_id"] = video_id
        rec["video_path"] = video_path
        rec["group"] = group
        normalized.append(rec)

    counts = Counter(rec["video_id"] for rec in normalized if rec["video_id"])
    duplicated_ids = [video_id for video_id, count in counts.items() if count > 1]
    if not duplicated_ids:
        return normalized

    logger.warning(
        "Detected duplicate video_id values across groups (%d duplicated ids). "
        "Applying group-prefixed remap for VBench run.",
        len(duplicated_ids),
    )

    ordered_groups: list[str] = []
    for name in group_names:
        if name and name not in ordered_groups:
            ordered_groups.append(name)
    for rec in normalized:
        group_name = str(rec.get("group", "")).strip()
        if group_name and group_name not in ordered_groups:
            ordered_groups.append(group_name)
    group_alias = {name: f"g{idx + 1}" for idx, name in enumerate(ordered_groups)}

    used_ids: set[str] = set()
    remapped_records = 0
    for rec in normalized:
        original_id = rec["video_id"]
        if not original_id:
            original_id = Path(rec["video_path"]).stem if rec.get("video_path") else "video"

        unique_id = original_id
        if counts.get(original_id, 0) > 1:
            group_name = str(rec.get("group") or "").strip()
            alias = group_alias.get(group_name, "g0")
            unique_id = f"{alias}_{original_id}"

        if unique_id in used_ids:
            suffix = 2
            candidate = f"{unique_id}_{suffix}"
            while candidate in used_ids:
                suffix += 1
                candidate = f"{unique_id}_{suffix}"
            unique_id = candidate

        if unique_id != rec.get("video_id"):
            remapped_records += 1
        rec["video_id"] = unique_id
        used_ids.add(unique_id)

    logger.warning("Remapped %d records to unique video_id.", remapped_records)
    if group_alias:
        logger.info("Group alias mapping for duplicate IDs: %s", group_alias)
    return normalized


# =============================================================================
# Frontend output
# =============================================================================
def copy_outputs_to_frontend(output_dir: Path, paths_config: dict, vbench_output: Path) -> bool:
    """Copy outputs to frontend/public/data and update manifest."""
    try:
        frontend_data_dir = PROJECT_ROOT / "frontend" / "public" / "data"
        frontend_data_dir.mkdir(parents=True, exist_ok=True)

        copied_files = []

        if vbench_output.exists():
            dst = frontend_data_dir / vbench_output.name
            shutil.copy2(vbench_output, dst)
            copied_files.append(vbench_output.name)
            logger.info(f"  Copied: {vbench_output.name}")

        for key, default_name in [
            ("group_summary", "group_summary.csv"),
            ("per_video_metrics", "per_video_metrics.csv"),
            ("experiment_output", None),
        ]:
            file_name = paths_config.get(key, default_name)
            if not file_name:
                continue
            src = output_dir / file_name
            if src.exists():
                dst = frontend_data_dir / src.name
                shutil.copy2(src, dst)
                copied_files.append(src.name)
                logger.info(f"  Copied: {src.name}")

        manifest_path = frontend_data_dir / "manifest.json"
        existing_files = set()
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    existing_files = set(manifest.get("files", []))
            except (json.JSONDecodeError, KeyError):
                existing_files = set()

        discovered_csv_files = {p.name for p in frontend_data_dir.glob("*.csv") if p.is_file()}
        all_files = sorted(
            {
                file_name
                for file_name in existing_files.union(set(copied_files)).union(discovered_csv_files)
                if str(file_name).endswith(".csv")
            }
        )
        with open(manifest_path, "w") as f:
            json.dump({"files": all_files}, f, indent=2)
        logger.info(f"  Updated manifest.json with {len(all_files)} files")
        return True
    except Exception as e:
        logger.warning(f"Failed to copy outputs to frontend: {e}")
        return False


# =============================================================================
# Video file helpers
# =============================================================================
def get_input_video_files(video_dir: Path) -> list[Path]:
    """List direct input videos under the VBench input directory."""
    valid_suffixes = {".mp4", ".avi", ".mov"}
    return sorted(
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_suffixes
    )


def are_split_clips_ready(video_dir: Path, input_videos: list[Path]) -> bool:
    """
    Check whether split clips are already prepared for all input videos.
    """
    split_dir = video_dir / "split_clip"
    if not split_dir.exists() or not split_dir.is_dir():
        return False
    if not input_videos:
        return False

    valid_suffixes = {".mp4", ".avi", ".mov"}
    for video_path in input_videos:
        clip_dir = split_dir / video_path.stem
        if not clip_dir.exists() or not clip_dir.is_dir():
            return False
        has_clip = any(
            clip_file.is_file() and clip_file.suffix.lower() in valid_suffixes
            for clip_file in clip_dir.iterdir()
        )
        if not has_clip:
            return False
    return True
