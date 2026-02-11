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
    Inject missing functions into transformers.modeling_utils.

    Newer transformers removed apply_chunking_to_forward, find_pruneable_heads_and_indices,
    and prune_linear_layer from modeling_utils. VBench's tag2Text/med.py still imports
    all three from there, so we re-inject whichever are missing.
    """
    import importlib

    import transformers.modeling_utils as _mu

    # The three functions tag2Text/med.py needs from transformers.modeling_utils
    needed = ["apply_chunking_to_forward", "find_pruneable_heads_and_indices", "prune_linear_layer"]
    missing = [name for name in needed if not hasattr(_mu, name)]
    if not missing:
        return

    # Try to find each missing function in alternative transformers modules
    alt_modules = []
    for mod_name in ("transformers.pytorch_utils",):
        try:
            alt_modules.append(importlib.import_module(mod_name))
        except ImportError:
            pass

    still_missing = []
    for name in missing:
        found = False
        for mod in alt_modules:
            fn = getattr(mod, name, None)
            if fn is not None:
                setattr(_mu, name, fn)
                found = True
                break
        if not found:
            still_missing.append(name)

    if not still_missing:
        logger.debug("Patched transformers.modeling_utils from pytorch_utils: %s", missing)
        return

    # Self-contained fallback for anything still missing
    _inject_builtin_shims(_mu, still_missing)
    logger.info(
        "Injected built-in shims into transformers.modeling_utils: %s",
        still_missing,
    )


def _inject_builtin_shims(target_module, names: list[str]) -> None:
    """Inject self-contained implementations of transformers helper functions."""
    import inspect

    import torch
    from torch import nn

    shims = {}

    def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
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

    shims["apply_chunking_to_forward"] = _apply_chunking_to_forward

    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    shims["find_pruneable_heads_and_indices"] = _find_pruneable_heads_and_indices

    def _prune_linear_layer(layer, index, dim=0):
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

    shims["prune_linear_layer"] = _prune_linear_layer

    for name in names:
        if name in shims:
            setattr(target_module, name, shims[name])


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
