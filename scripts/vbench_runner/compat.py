"""
Compatibility patches for VBench internals.

Fixes known issues in the VBench submodule without modifying its source:
  1. transformers removed apply_chunking_to_forward from modeling_utils
  2. clip.tokenize() defaults to truncate=False, causing errors on long prompts
  3. GrIT model expects torch.device but receives a string
"""

try:
    from .env import logger
except ImportError:
    from vbench_runner.env import logger


def patch_transformers_compat() -> None:
    """
    Re-export apply_chunking_to_forward into transformers.modeling_utils.

    Newer transformers (>=4.40) moved the function to transformers.pytorch_utils.
    VBench's tag2Text/med.py still imports from the old location.
    """
    try:
        from transformers.modeling_utils import apply_chunking_to_forward  # noqa: F401

        return  # already available, nothing to do
    except ImportError:
        pass

    try:
        from transformers.pytorch_utils import (
            apply_chunking_to_forward,
            find_pruneable_heads_and_indices,
            prune_linear_layer,
        )
    except ImportError:
        logger.warning(
            "Cannot find apply_chunking_to_forward in transformers; " "scene dimension may fail."
        )
        return

    import transformers.modeling_utils as _mu

    _mu.apply_chunking_to_forward = apply_chunking_to_forward  # type: ignore[attr-defined]
    if not hasattr(_mu, "find_pruneable_heads_and_indices"):
        _mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices  # type: ignore[attr-defined]
    if not hasattr(_mu, "prune_linear_layer"):
        _mu.prune_linear_layer = prune_linear_layer  # type: ignore[attr-defined]
    logger.debug("Patched transformers.modeling_utils with apply_chunking_to_forward shim.")


def patch_clip_tokenize_truncate() -> None:
    """
    Monkey-patch clip.tokenize to default truncate=True.

    VBench's appearance_style passes raw prompts to clip.tokenize() without
    truncate=True, causing RuntimeError for prompts exceeding 77 tokens.
    """
    try:
        import clip
    except ImportError:
        return

    original_tokenize = clip.tokenize

    def _tokenize_with_truncate(texts, context_length=77, truncate=True):
        return original_tokenize(texts, context_length=context_length, truncate=truncate)

    # Avoid double-patching
    if getattr(clip.tokenize, "_patched_truncate", False):
        return

    clip.tokenize = _tokenize_with_truncate
    clip.tokenize._patched_truncate = True  # type: ignore[attr-defined]
    logger.debug("Patched clip.tokenize to default truncate=True.")


def patch_grit_device_compat() -> None:
    """
    Patch GrIT's get_parser to accept string device values.

    VBench stores device as a string (e.g. "cuda"), but GrIT's get_parser()
    accesses device.type which requires a torch.device object.
    """
    try:
        import torch
        from vbench.third_party.grit_src import image_dense_captions as _idc
    except ImportError:
        return

    original_get_parser = _idc.get_parser

    def _get_parser_compat(device, *args, **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        return original_get_parser(device, *args, **kwargs)

    if getattr(_idc.get_parser, "_patched_device", False):
        return

    _idc.get_parser = _get_parser_compat
    _idc.get_parser._patched_device = True  # type: ignore[attr-defined]
    logger.debug("Patched GrIT get_parser for string device compatibility.")


def apply_vbench_compat_patches() -> None:
    """Apply all VBench compatibility patches."""
    patch_transformers_compat()
    patch_clip_tokenize_truncate()
    patch_grit_device_compat()
