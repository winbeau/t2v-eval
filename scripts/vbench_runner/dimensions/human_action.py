from .base import DimensionSpec


SPEC = DimensionSpec(
    key="human_action",
    description="Human action correctness",
    requires_clip=True,
)
