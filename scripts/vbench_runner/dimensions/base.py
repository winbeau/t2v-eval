from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionSpec:
    key: str
    description: str
    requires_clip: bool = False
    requires_pyiqa: bool = False
    long_mode_only: bool = True

