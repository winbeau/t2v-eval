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

PERSON_TOKENS = {
    "person",
    "people",
    "woman",
    "women",
    "man",
    "men",
    "girl",
    "boy",
    "lady",
    "gentleman",
    "child",
    "children",
}

SMALL_COLOR_OBJECT_TOKENS = {
    "lipstick",
    "earring",
    "earrings",
    "ring",
    "rings",
    "nail",
    "nails",
    "bracelet",
    "bracelets",
    "necklace",
    "necklaces",
}

VISUAL_OBJECT_ALIASES = {
    "woman": ["woman", "person", "lady", "girl"],
    "women": ["women", "woman", "person", "people", "ladies", "girls"],
    "man": ["man", "person", "gentleman", "boy"],
    "men": ["men", "man", "person", "people", "gentlemen", "boys"],
    "person": ["person", "woman", "man", "lady", "gentleman", "girl", "boy"],
    "people": ["people", "persons", "women", "men"],
    "purse": ["purse", "bag", "handbag"],
    "handbag": ["handbag", "bag", "purse"],
    "bag": ["bag", "handbag", "purse"],
    "dress": ["dress", "gown"],
    "shirt": ["shirt", "top", "blouse"],
    "blouse": ["blouse", "shirt", "top"],
    "jacket": ["jacket", "coat"],
    "coat": ["coat", "jacket"],
    "boots": ["boots", "boot"],
    "boot": ["boot", "boots"],
}

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

# Common verbs / adjectives that GrIT cannot detect as objects.
# Used to filter fallback words in multiple_objects extraction.
COMMON_ACTION_WORDS = {
    "walking",
    "running",
    "sitting",
    "standing",
    "talking",
    "eating",
    "playing",
    "dancing",
    "swimming",
    "flying",
    "driving",
    "reading",
    "writing",
    "sleeping",
    "looking",
    "moving",
    "jumping",
    "climbing",
    "falling",
    "turning",
    "holding",
    "wearing",
    "smiling",
    "laughing",
    "crying",
    "singing",
    "working",
    "making",
    "going",
    "coming",
    "spinning",
    "floating",
    "rolling",
    "flowing",
    "growing",
    "glowing",
    "blowing",
    "shining",
    "burning",
    "melting",
    "raining",
    "snowing",
    "waving",
    "painting",
    "cooking",
    "baking",
    "drinking",
    "surfing",
    "skating",
    "skiing",
    "fishing",
    "hunting",
    "blooming",
    "wilting",
    "pouring",
    "dripping",
    "crashing",
    "breaking",
    "bouncing",
    "exploding",
    "beautiful",
    "small",
    "large",
    "big",
    "old",
    "young",
    "new",
    "fast",
    "slow",
    "long",
    "short",
    "tall",
    "dark",
    "bright",
    "through",
    "down",
    "into",
    "over",
    "across",
    "around",
    "along",
    "toward",
    "towards",
    "away",
    "together",
    "slowly",
    "quickly",
    "gently",
}

# Common scene/location words that tag2text is likely to caption
SCENE_KEYWORDS = {
    "beach",
    "ocean",
    "sea",
    "lake",
    "river",
    "mountain",
    "forest",
    "park",
    "garden",
    "city",
    "street",
    "road",
    "highway",
    "room",
    "kitchen",
    "bedroom",
    "bathroom",
    "living",
    "office",
    "classroom",
    "studio",
    "stage",
    "field",
    "farm",
    "desert",
    "snow",
    "rain",
    "sunset",
    "sunrise",
    "night",
    "sky",
    "underwater",
    "space",
    "jungle",
    "cave",
    "bridge",
    "castle",
    "church",
    "temple",
    "market",
    "restaurant",
    "cafe",
    "bar",
    "hospital",
    "school",
    "library",
    "museum",
    "airport",
    "station",
    "pool",
    "gym",
    "stadium",
    "arena",
    "courtyard",
    "balcony",
    "rooftop",
    "hallway",
    "corridor",
    "basement",
    "attic",
    "garage",
    "yard",
    "lawn",
    "meadow",
    "valley",
    "cliff",
    "waterfall",
    "island",
    "harbor",
    "dock",
    "pier",
    "alley",
    "plaza",
    "square",
    "mall",
    "store",
    "shop",
    "warehouse",
    "factory",
    "laboratory",
    "countryside",
    "village",
    "town",
    "suburb",
    "woodland",
    "savanna",
    "tundra",
    "swamp",
    "marsh",
    "canyon",
    "volcano",
    "glacier",
    "campsite",
    "trail",
    "pathway",
    "sidewalk",
    "playground",
    "carnival",
    "circus",
    "concert",
    "theater",
    "cinema",
    "gallery",
    "lobby",
    "staircase",
    "elevator",
    "train",
    "bus",
    "airplane",
    "boat",
    "ship",
    "outdoor",
    "indoor",
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


def normalize_color_token(color_key: str) -> str:
    """Normalize color token for comparisons."""
    normalized = str(color_key or "").strip().lower()
    return "gray" if normalized == "grey" else normalized


def _color_aliases(color_key: str) -> set[str]:
    normalized = normalize_color_token(color_key)
    aliases = {normalized} if normalized else set()
    if normalized == "gray":
        aliases.add("grey")
    return aliases


def _meaningful_tokens(text: str) -> list[str]:
    return [
        token
        for token in tokenize_prompt_words(text)
        if token not in PROMPT_STOPWORDS
        and token not in COLOR_WORDS
        and token not in COMMON_ACTION_WORDS
    ]


def extract_primary_subject_token(text: str, default: str = "object") -> str:
    """Extract a likely main visible subject from prompt text."""
    tokens = _meaningful_tokens(text)
    if not tokens:
        return default
    for token in tokens[:12]:
        if token in PERSON_TOKENS:
            return token
    return tokens[0]


def expand_visual_object_aliases(token: str) -> list[str]:
    """Expand a canonical token to a small explicit alias set."""
    normalized = str(token or "").strip().lower()
    if not normalized:
        return []
    aliases = VISUAL_OBJECT_ALIASES.get(normalized)
    if aliases is not None:
        return list(dict.fromkeys(aliases))
    return [normalized]


def _candidate_priority(token: str) -> tuple[int, str]:
    normalized = str(token or "").strip().lower()
    if normalized in SMALL_COLOR_OBJECT_TOKENS:
        return (2, normalized)
    if normalized in PERSON_TOKENS:
        return (1, normalized)
    return (0, normalized)


def extract_color_object_candidates(prompt_text: str, color_key: str) -> list[str]:
    """
    Extract ordered object candidates for a color prompt.

    Priority:
      1. Objects directly following the target color word.
      2. Main visible subject inferred from prompt.
      3. Unique, visually plausible alternatives from the same prompt.
    """
    prompt_simple = simplify_prompt_text(prompt_text)
    tokens = tokenize_prompt_words(prompt_simple)
    color_aliases = _color_aliases(color_key)

    explicit_candidates: list[str] = []
    for idx, token in enumerate(tokens):
        if token not in color_aliases:
            continue
        for look_ahead in range(idx + 1, min(len(tokens), idx + 5)):
            candidate = tokens[look_ahead]
            if candidate in PROMPT_STOPWORDS or candidate in COLOR_WORDS:
                continue
            explicit_candidates.append(candidate)
            break

    fallback_candidates: list[str] = []
    primary_subject = extract_primary_subject_token(prompt_simple, default="object")
    if primary_subject:
        fallback_candidates.append(primary_subject)
    for token in _meaningful_tokens(prompt_simple):
        fallback_candidates.append(token)

    ordered: list[str] = []
    seen: set[str] = set()
    for token in explicit_candidates + fallback_candidates:
        for alias in expand_visual_object_aliases(token):
            if alias and alias not in seen:
                ordered.append(alias)
                seen.add(alias)

    if not ordered:
        return ["object"]

    explicit_set = set(explicit_candidates)
    non_small_explicit_set = {
        token for token in explicit_set if token not in SMALL_COLOR_OBJECT_TOKENS
    }
    primary_subject_aliases = set(expand_visual_object_aliases(primary_subject))
    ranked = sorted(
        ordered,
        key=lambda token: (
            (
                0
                if token in non_small_explicit_set
                or any(
                    base in non_small_explicit_set for base in expand_visual_object_aliases(token)
                )
                else (
                    1
                    if token in primary_subject_aliases and token not in SMALL_COLOR_OBJECT_TOKENS
                    else 2 if token in SMALL_COLOR_OBJECT_TOKENS else 1
                )
            ),
            0 if token not in SMALL_COLOR_OBJECT_TOKENS else 1,
            ordered.index(token),
        ),
    )
    return list(dict.fromkeys(ranked))


def build_color_object_key(prompt_text: str, color_key: str, default: str = "object") -> str:
    """
    Build a safe object-matching text for the color dimension.

    Unlike the upstream prompt.replace(...) approach, this only removes standalone
    article/color tokens and keeps words like "woman", "paired", and "captured" intact.
    """
    prompt_simple = simplify_prompt_text(prompt_text)
    color_aliases = _color_aliases(color_key)

    filtered_tokens = [
        token
        for token in tokenize_prompt_words(prompt_simple)
        if token not in {"a", "an", "the"} and token not in color_aliases
    ]
    object_key = " ".join(filtered_tokens).strip()
    if object_key:
        return object_key
    return extract_object_token(prompt_simple, default=default)


def normalize_color_auxiliary_payload(
    payload: dict | None,
    prompt_text: str,
) -> dict | None:
    """Return a color auxiliary payload with normalized structured fields."""
    if not isinstance(payload, dict):
        return payload
    normalized = deepcopy(payload)
    color_value = normalize_color_token(normalized.get("color", ""))
    normalized["color"] = color_value or "red"

    candidates = normalized.get("object_candidates")
    if not isinstance(candidates, list) or not candidates:
        candidates = extract_color_object_candidates(prompt_text, normalized["color"])

    deduped_candidates: list[str] = []
    seen: set[str] = set()
    for token in candidates:
        raw = str(token or "").strip().lower()
        if not raw:
            continue
        for alias in expand_visual_object_aliases(raw):
            if alias and alias not in seen:
                deduped_candidates.append(alias)
                seen.add(alias)
    if not deduped_candidates:
        deduped_candidates = extract_color_object_candidates(prompt_text, normalized["color"])

    normalized["object_candidates"] = deduped_candidates
    object_value = str(normalized.get("object", "")).strip().lower()
    normalized["object"] = object_value or deduped_candidates[0]
    if normalized["object"] not in deduped_candidates:
        normalized["object_candidates"] = [normalized["object"], *deduped_candidates]
    normalized["object_key"] = str(
        normalized.get("object_key") or build_color_object_key(prompt_text, normalized["color"])
    ).strip()
    return normalized


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
        tokens = tokenize_prompt_words(prompt_simple)
        # Find the first scene-related keyword in the prompt
        for token in tokens:
            if token in SCENE_KEYWORDS:
                return {"scene": {"scene": token}}
        # Fallback: use preposition-context extraction ("in a park", "on the beach", etc.)
        loc_match = re.search(
            r"(?:in|on|at|near|through|across|along|beside|under|over)\s+"
            r"(?:a|an|the|)\s*([a-z]+)",
            prompt_simple,
        )
        if loc_match:
            candidate = loc_match.group(1).strip()
            if candidate not in PROMPT_STOPWORDS and candidate not in COLOR_WORDS:
                return {"scene": {"scene": candidate}}
        # Final fallback
        first_clause = re.split(r"[,.;]", prompt_simple, maxsplit=1)[0]
        return {"scene": {"scene": extract_object_token(first_clause, default="outdoor")}}

    if dimension == "object_class":
        first_clause = re.split(r"[,.;]", prompt_simple, maxsplit=1)[0]
        return {"object": extract_primary_subject_token(first_clause, default="person")}

    if dimension == "multiple_objects":
        # Try "X and Y" pattern first
        pattern = re.search(
            r"(?:^|\b)(.+?)\s+and\s+(.+?)(?:$|,|;|\.)",
            prompt_simple,
        )
        if pattern:
            obj_a = extract_primary_subject_token(pattern.group(1), default="person")
            obj_b = extract_primary_subject_token(pattern.group(2), default="object")
        else:
            # Try "X with Y", "X beside Y", "X near Y" patterns
            prep_pattern = re.search(
                r"(?:^|\b)(.+?)\s+(?:with|beside|near|next to|behind|facing)\s+(.+?)(?:$|,|;|\.)",
                prompt_simple,
            )
            if prep_pattern:
                obj_a = extract_primary_subject_token(prep_pattern.group(1), default="person")
                obj_b = extract_primary_subject_token(prep_pattern.group(2), default="object")
            else:
                words = [
                    token
                    for token in tokenize_prompt_words(prompt_simple)
                    if token not in PROMPT_STOPWORDS
                    and token not in COLOR_WORDS
                    and token not in COMMON_ACTION_WORDS
                ]
                obj_a = words[0] if len(words) >= 1 else "person"
                obj_b = words[1] if len(words) >= 2 else "object"
        return {"object": f"{obj_a} and {obj_b}"}

    if dimension == "spatial_relationship":
        for relation in SPATIAL_RELATIONS:
            if relation in prompt_simple:
                left, right = prompt_simple.split(relation, 1)
                obj_a = extract_primary_subject_token(left, default="object")
                obj_b = extract_primary_subject_token(right, default="object")
                return {
                    "spatial_relationship": {
                        "object_a": obj_a,
                        "object_b": obj_b,
                        "relationship": relation,
                    }
                }
        # No explicit spatial relation in prompt — extract two objects and
        # default to "on the left of" (score will depend on GrIT detections)
        words = [
            token
            for token in tokenize_prompt_words(prompt_simple)
            if token not in PROMPT_STOPWORDS
            and token not in COLOR_WORDS
            and token not in COMMON_ACTION_WORDS
        ]
        obj_a = extract_primary_subject_token(prompt_simple, default="object")
        obj_b = words[1] if len(words) >= 2 and words[1] != obj_a else "object"
        return {
            "spatial_relationship": {
                "object_a": obj_a,
                "object_b": obj_b,
                "relationship": "on the left of",
            }
        }

    if dimension == "color":
        for color in COLOR_WORDS:
            if re.search(rf"\b{re.escape(color)}\b", prompt_simple):
                normalized_color = normalize_color_token(color)
                return normalize_color_auxiliary_payload(
                    {
                        "color": normalized_color,
                    },
                    prompt_text,
                )
        return None

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

    with open(full_info_path, encoding="utf-8") as f:
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

    def _finalize_color_payload(payload: dict | None) -> dict | None:
        if dimension != "color":
            return payload
        return normalize_color_auxiliary_payload(payload, prompt_text)

    norm_key = normalize_prompt_text(prompt_text)
    simple_key = simplify_prompt_text(prompt_text)

    by_exact = exact_lookup.get(dimension, {})
    by_simple = simple_lookup.get(dimension, {})
    if norm_key in by_exact:
        return _finalize_color_payload(by_exact[norm_key]), "exact"
    if simple_key in by_simple:
        return _finalize_color_payload(by_simple[simple_key]), "simplified"

    inferred = infer_auxiliary_from_prompt(dimension, prompt_text)
    if inferred is not None:
        return _finalize_color_payload(inferred), "heuristic"
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
        prompt_list=None,
        special_str="",
        verbose=False,
        mode="vbench_standard",
        **kwargs,
    ):
        if prompt_list is None:
            prompt_list = []
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
