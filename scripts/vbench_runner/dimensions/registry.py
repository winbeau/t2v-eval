from .appearance_style import SPEC as APPEARANCE_STYLE
from .aesthetic_quality import SPEC as AESTHETIC_QUALITY
from .background_consistency import SPEC as BACKGROUND_CONSISTENCY
from .base import DimensionSpec
from .color import SPEC as COLOR
from .dynamic_degree import SPEC as DYNAMIC_DEGREE
from .human_action import SPEC as HUMAN_ACTION
from .imaging_quality import SPEC as IMAGING_QUALITY
from .motion_smoothness import SPEC as MOTION_SMOOTHNESS
from .multiple_objects import SPEC as MULTIPLE_OBJECTS
from .object_class import SPEC as OBJECT_CLASS
from .overall_consistency import SPEC as OVERALL_CONSISTENCY
from .scene import SPEC as SCENE
from .spatial_relationship import SPEC as SPATIAL_RELATIONSHIP
from .subject_consistency import SPEC as SUBJECT_CONSISTENCY
from .temporal_flickering import SPEC as TEMPORAL_FLICKERING
from .temporal_style import SPEC as TEMPORAL_STYLE

LONG_DIMENSIONS_16: list[DimensionSpec] = [
    SUBJECT_CONSISTENCY,
    BACKGROUND_CONSISTENCY,
    TEMPORAL_FLICKERING,
    MOTION_SMOOTHNESS,
    TEMPORAL_STYLE,
    APPEARANCE_STYLE,
    SCENE,
    OBJECT_CLASS,
    MULTIPLE_OBJECTS,
    SPATIAL_RELATIONSHIP,
    HUMAN_ACTION,
    COLOR,
    OVERALL_CONSISTENCY,
    DYNAMIC_DEGREE,
    IMAGING_QUALITY,
    AESTHETIC_QUALITY,
]

LONG_DIMENSIONS_6_RECOMMENDED: list[DimensionSpec] = [
    SUBJECT_CONSISTENCY,
    BACKGROUND_CONSISTENCY,
    MOTION_SMOOTHNESS,
    DYNAMIC_DEGREE,
    IMAGING_QUALITY,
    AESTHETIC_QUALITY,
]

LONG_DIMENSIONS: list[DimensionSpec] = LONG_DIMENSIONS_16
LONG_DIMENSION_KEYS: list[str] = [spec.key for spec in LONG_DIMENSIONS]
LONG_DIMENSION_SET: set[str] = set(LONG_DIMENSION_KEYS)
CLIP_REQUIRED_DIMENSIONS: set[str] = {spec.key for spec in LONG_DIMENSIONS if spec.requires_clip}
PYIQA_REQUIRED_DIMENSIONS: set[str] = {spec.key for spec in LONG_DIMENSIONS if spec.requires_pyiqa}


def normalize_subtasks(subtasks: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for name in subtasks:
        key = str(name).strip()
        if key and key not in seen:
            normalized.append(key)
            seen.add(key)
    return normalized


def default_long_subtasks(profile: str = "long_6") -> list[str]:
    profile_key = str(profile).strip().lower()
    if profile_key in {"long_16", "16", "16d", "full", "full_16"}:
        return [spec.key for spec in LONG_DIMENSIONS_16]
    return [spec.key for spec in LONG_DIMENSIONS_6_RECOMMENDED]


def supported_long_subtasks() -> list[str]:
    return [spec.key for spec in LONG_DIMENSIONS_16]
