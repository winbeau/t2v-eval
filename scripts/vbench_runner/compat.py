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


def _import_or_build_chunking_helpers():
    """
    Try importing from transformers, fall back to self-contained implementations.

    Returns (apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer).
    """
    # Try the original location first
    for module_name in ("transformers.modeling_utils", "transformers.pytorch_utils"):
        try:
            import importlib

            mod = importlib.import_module(module_name)
            fn = getattr(mod, "apply_chunking_to_forward", None)
            if fn is not None:
                return (
                    fn,
                    getattr(mod, "find_pruneable_heads_and_indices", None),
                    getattr(mod, "prune_linear_layer", None),
                )
        except ImportError:
            continue

    # --- Self-contained fallback implementations ---
    import inspect

    import torch
    from torch import nn

    def apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        if chunk_size > 0:
            num_args = len(inspect.signature(forward_fn).parameters)
            if num_args != len(input_tensors):
                raise ValueError(
                    f"forward_chunk_fn expects {num_args} arguments, "
                    f"but {len(input_tensors)} input tensors given"
                )
            tensor_shape = input_tensors[0].shape[chunk_dim]
            num_chunks = tensor_shape // chunk_size
            input_chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
            output_chunks = tuple(forward_fn(*chunk) for chunk in zip(*input_chunks, strict=False))
            return torch.cat(output_chunks, dim=chunk_dim)
        return forward_fn(*input_tensors)

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        w = layer.weight.index_select(dim, index).detach().clone()
        if layer.bias is not None:
            b = layer.bias.detach().clone() if dim == 1 else layer.bias[index].detach().clone()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
            layer.weight.device
        )
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(w.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer

    logger.info("Using built-in shims for apply_chunking_to_forward (not found in transformers).")
    return apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer


def patch_transformers_compat() -> None:
    """
    Inject apply_chunking_to_forward into transformers.modeling_utils.

    Newer transformers removed this function (it lived in modeling_utils,
    then pytorch_utils, then was dropped entirely).  VBench's tag2Text/med.py
    still imports it from modeling_utils, so we re-inject it.
    """
    try:
        from transformers.modeling_utils import apply_chunking_to_forward  # noqa: F401

        return  # already available, nothing to do
    except ImportError:
        pass

    chunking, pruning_heads, pruning_layer = _import_or_build_chunking_helpers()

    import transformers.modeling_utils as _mu

    _mu.apply_chunking_to_forward = chunking  # type: ignore[attr-defined]
    if pruning_heads is not None and not hasattr(_mu, "find_pruneable_heads_and_indices"):
        _mu.find_pruneable_heads_and_indices = pruning_heads  # type: ignore[attr-defined]
    if pruning_layer is not None and not hasattr(_mu, "prune_linear_layer"):
        _mu.prune_linear_layer = pruning_layer  # type: ignore[attr-defined]
    logger.debug("Patched transformers.modeling_utils with apply_chunking_to_forward.")


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
