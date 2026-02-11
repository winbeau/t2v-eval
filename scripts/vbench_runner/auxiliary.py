"""
Prompt processing, auxiliary info inference, and VBench-Long patching.
"""

import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from .env import logger
except ImportError:
    from vbench_runner.env import logger

# =============================================================================
# Constants (used only by prompt/auxiliary functions)
# =============================================================================
AUXILIARY_REQUIRED_LONG_DIMS: set[str] = {
    "object_class",
    "multiple_objects",
    "scene",
    "appearance_style",
    "color",
    "spatial_relationship",
}

COLOR_WORDS = (
    "white",
    "red",
    "pink",
    "blue",
    "silver",
    "purple",
    "orange",
    "green",
    "gray",
    "grey",
    "yellow",
    "black",
    "brown",
)

SPATIAL_RELATIONS = (
    "on the left of",
    "on the right of",
    "on the top of",
    "on the bottom of",
)

PROMPT_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "with",
    "for",
    "and",
    "or",
    "from",
    "by",
    "style",
    "view",
    "shot",
    "camera",
    "front",
    "back",
    "side",
    "left",
    "right",
    "top",
    "bottom",
}


# =============================================================================
# Prompt text helpers
# =============================================================================
def normalize_prompt_text(text: str) -> str:
    """Normalize prompt for robust key lookup."""
    return " ".join(str(text or "").strip().lower().split())


def simplify_prompt_text(text: str) -> str:
    """Normalize prompt and remove punctuation for fallback key lookup."""
    normalized = normalize_prompt_text(text)
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    return " ".join(normalized.split())


def tokenize_prompt_words(text: str) -> list[str]:
    """Tokenize lowercase words from prompt."""
    return re.findall(r"[a-z]+", str(text or "").lower())


def extract_object_token(text: str, default: str = "object") -> str:
    """Extract one object-like token from free-form text."""
    tokens = [
        token
        for token in tokenize_prompt_words(text)
        if token not in PROMPT_STOPWORDS and token not in COLOR_WORDS
    ]
    return tokens[-1] if tokens else default


# =============================================================================
# Auxiliary info inference
# =============================================================================
def infer_auxiliary_from_prompt(dimension: str, prompt_text: str) -> dict | None:
    """Infer missing auxiliary info for long custom input dimensions."""
    prompt_text = str(prompt_text or "")
    prompt_simple = simplify_prompt_text(prompt_text)

    if dimension == "appearance_style":
        style_match = re.search(
            r"(?:in the style of|style of)\s+([a-z0-9\s-]+)",
            prompt_simple,
        )
        if style_match:
            style = style_match.group(1).strip()
        else:
            style_tail = re.search(r"([a-z0-9\s-]+ style)\b", prompt_simple)
            style = style_tail.group(1).strip() if style_tail else ""
        if not style:
            style = "realistic style"
        return {"appearance_style": style}

    if dimension == "scene":
        first_clause = re.split(r"[,.;]", prompt_simple, maxsplit=1)[0]
        return {"scene": extract_object_token(first_clause, default="outdoor")}

    if dimension == "object_class":
        first_clause = re.split(r"[,.;]", prompt_simple, maxsplit=1)[0]
        return {"object": extract_object_token(first_clause, default="person")}

    if dimension == "multiple_objects":
        pattern = re.search(
            r"(?:^|\b)(.+?)\s+and\s+(.+?)(?:$|,|;|\.)",
            prompt_simple,
        )
        if pattern:
            obj_a = extract_object_token(pattern.group(1), default="person")
            obj_b = extract_object_token(pattern.group(2), default="object")
        else:
            words = [
                token
                for token in tokenize_prompt_words(prompt_simple)
                if token not in PROMPT_STOPWORDS and token not in COLOR_WORDS
            ]
            obj_a = words[0] if len(words) >= 1 else "person"
            obj_b = words[1] if len(words) >= 2 else "object"
        return {"object": f"{obj_a} and {obj_b}"}

    if dimension == "spatial_relationship":
        for relation in SPATIAL_RELATIONS:
            if relation in prompt_simple:
                left, right = prompt_simple.split(relation, 1)
                obj_a = extract_object_token(left, default="object")
                obj_b = extract_object_token(right, default="object")
                return {
                    "spatial_relationship": {
                        "object_a": obj_a,
                        "object_b": obj_b,
                        "relationship": relation,
                    }
                }
        return {
            "spatial_relationship": {
                "object_a": extract_object_token(prompt_simple, default="object"),
                "object_b": "object",
                "relationship": "on the left of",
            }
        }

    if dimension == "color":
        for color in COLOR_WORDS:
            if re.search(rf"\b{re.escape(color)}\b", prompt_simple):
                normalized_color = "gray" if color == "grey" else color
                return {"color": normalized_color}
        return {"color": "red"}

    return None


# =============================================================================
# Auxiliary lookup from official VBench full-info
# =============================================================================
def build_auxiliary_prompt_lookup(
    full_info_path: Path,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    """
    Build auxiliary lookup maps from official VBench full-info.

    Returns:
        (exact_prompt_lookup, simplified_prompt_lookup)
    """
    if not full_info_path.exists():
        return {}, {}

    with open(full_info_path, "r", encoding="utf-8") as f:
        full_info_data = json.load(f)

    exact: dict[str, dict[str, dict]] = {dim: {} for dim in AUXILIARY_REQUIRED_LONG_DIMS}
    simple: dict[str, dict[str, dict]] = {dim: {} for dim in AUXILIARY_REQUIRED_LONG_DIMS}

    for item in full_info_data:
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt_en", "")).strip()
        aux_all = item.get("auxiliary_info")
        if not prompt or not isinstance(aux_all, dict):
            continue

        norm_key = normalize_prompt_text(prompt)
        simple_key = simplify_prompt_text(prompt)
        for dim in AUXILIARY_REQUIRED_LONG_DIMS:
            payload = aux_all.get(dim)
            if not isinstance(payload, dict):
                continue
            exact[dim].setdefault(norm_key, deepcopy(payload))
            simple[dim].setdefault(simple_key, deepcopy(payload))

    return exact, simple


def resolve_auxiliary_payload(
    dimension: str,
    prompt_text: str,
    exact_lookup: dict[str, dict[str, dict]],
    simple_lookup: dict[str, dict[str, dict]],
) -> tuple[dict | None, str]:
    """Resolve auxiliary payload by official lookup first, then heuristic fallback."""
    norm_key = normalize_prompt_text(prompt_text)
    simple_key = simplify_prompt_text(prompt_text)

    by_exact = exact_lookup.get(dimension, {})
    by_simple = simple_lookup.get(dimension, {})
    if norm_key in by_exact:
        return deepcopy(by_exact[norm_key]), "exact"
    if simple_key in by_simple:
        return deepcopy(by_simple[simple_key]), "simplified"

    inferred = infer_auxiliary_from_prompt(dimension, prompt_text)
    if inferred is not None:
        return inferred, "heuristic"
    return None, "missing"


# =============================================================================
# VBench-Long patching
# =============================================================================
def patch_long_custom_full_info_builder(
    vbench_runner: Any,
    video_records: list[dict],
    exact_lookup: dict[str, dict[str, dict]],
    simple_lookup: dict[str, dict[str, dict]],
) -> None:
    """
    Patch VBench-Long `build_full_info_json` for long_custom_input to support all 16 dimensions.

    Upstream `long_custom_input` uses video-id as prompt and omits auxiliary_info, which makes
    6 dimensions fail. This patch keeps custom mode but injects:
      - true `prompt_en` from metadata
      - `auxiliary_info` for required dimensions
    """
    original_build = vbench_runner.build_full_info_json
    prompt_by_video_id = {
        str(record.get("video_id", "")): str(record.get("prompt", "")).strip()
        for record in video_records
    }

    def _patched_build_full_info_json(
        self,
        videos_path,
        name,
        dimension_list,
        prompt_list=[],
        special_str="",
        verbose=False,
        mode="vbench_standard",
        **kwargs,
    ):
        if str(mode).strip().lower() != "long_custom_input":
            return original_build(
                videos_path,
                name,
                dimension_list,
                prompt_list,
                special_str,
                verbose,
                mode,
                **kwargs,
            )

        split_root = Path(videos_path) / "split_clip"
        if not split_root.exists():
            return original_build(
                videos_path,
                name,
                dimension_list,
                prompt_list,
                special_str,
                verbose,
                mode,
                **kwargs,
            )

        valid_suffixes = {".mp4", ".avi", ".mov"}
        clips_by_video: dict[str, list[str]] = {}
        for folder in sorted(split_root.iterdir()):
            if not folder.is_dir():
                continue
            base_video_id = folder.name.split("-Scene")[0]
            clip_files = sorted(
                str(path)
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in valid_suffixes
            )
            if clip_files:
                clips_by_video.setdefault(base_video_id, []).extend(clip_files)

        if not clips_by_video:
            raise RuntimeError(f"No split clips found under: {split_root}")

        aux_dims = sorted(set(dimension_list) & AUXILIARY_REQUIRED_LONG_DIMS)
        source_counter: Counter[str] = Counter()
        full_info_payload = []
        for video_id in sorted(clips_by_video.keys()):
            prompt_text = prompt_by_video_id.get(video_id) or video_id
            item = {
                "prompt_en": prompt_text,
                "dimension": list(dimension_list),
                "video_list": clips_by_video[video_id],
            }
            if aux_dims:
                aux_map: dict[str, dict] = {}
                for dim in aux_dims:
                    aux_payload, source = resolve_auxiliary_payload(
                        dimension=dim,
                        prompt_text=prompt_text,
                        exact_lookup=exact_lookup,
                        simple_lookup=simple_lookup,
                    )
                    if aux_payload is None:
                        raise ValueError(
                            f"Cannot build auxiliary_info for dimension `{dim}` and prompt: "
                            f"{prompt_text!r} (video_id={video_id})."
                        )
                    aux_map[dim] = aux_payload
                    source_counter[source] += 1
                item["auxiliary_info"] = aux_map
            full_info_payload.append(item)

        output_path = Path(self.output_path) / f"{name}_full_info.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(full_info_payload, f, ensure_ascii=False, indent=2)

        if aux_dims:
            logger.info(
                "[%s] auxiliary_info ready for %d videos (%s)",
                name,
                len(full_info_payload),
                ", ".join(f"{k}={v}" for k, v in sorted(source_counter.items())),
            )

        print(f"Evaluation meta data saved to {output_path}")
        return str(output_path)

    vbench_runner.build_full_info_json = _patched_build_full_info_json.__get__(
        vbench_runner, type(vbench_runner)
    )
