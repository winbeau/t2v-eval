from .base import DimensionSpec

SPEC = DimensionSpec(
    key="imaging_quality",
    description="Frame-wise imaging quality",
    requires_pyiqa=True,
)
