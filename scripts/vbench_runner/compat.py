"""
Compatibility patches for VBench internals.

Fixes known issues in the VBench submodule without modifying its source:
  1. transformers removed apply_chunking_to_forward from modeling_utils
  2. transformers removed all_tied_weights_keys from PreTrainedModel
  3. transformers removed additional_special_tokens_ids dynamic resolution
  4. clip.tokenize() defaults to truncate=False, causing errors on long prompts
  5. GrIT model expects torch.device but receives a string
  6. transformers >=4.50 removed GenerationMixin from PreTrainedModel bases
  7. config.pruned_heads may not be a dict in some code paths
  8. human_action extracts labels from filenames; custom videos need prompt matching
  9. GrIT inference speedup: skip unused visualization + FP16 autocast
 10. color check_generate uses exact match; custom prompts are multi-word
 11. GrIT run_on_batch mode switch:
     - safe mode: sequential inference with batch=1 assertion
     - parallel mode: chunked batch inference for acceleration
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
    Patch GrIT's get_parser to accept string device values and bind CUDA index.

    VBench stores device as a string (e.g. "cuda"), but GrIT's get_parser()
    accesses device.type which requires a torch.device object.
    Also force MODEL.DEVICE to explicit cuda:{index} so each torchrun worker
    stays on its assigned GPU.
    """
    try:
        import torch
        from vbench.third_party.grit_src import image_dense_captions as _idc
    except ImportError:
        return

    original_get_parser = _idc.get_parser

    def _normalize_device(value):
        if isinstance(value, str):
            raw = value.strip().lower()
            if raw == "cuda" and torch.cuda.is_available():
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            return torch.device(value)
        if isinstance(value, torch.device) and value.type == "cuda" and value.index is None:
            if torch.cuda.is_available():
                return torch.device(f"cuda:{torch.cuda.current_device()}")
        return value

    def _with_model_device_override(args_dict: dict, device) -> dict:
        if not isinstance(device, torch.device) or device.type != "cuda":
            return args_dict
        index = device.index
        if index is None:
            index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        device_str = f"cuda:{index}"
        opts = list(args_dict.get("opts", []))
        cleaned_opts: list = []
        skip_next = False
        for idx, item in enumerate(opts):
            if skip_next:
                skip_next = False
                continue
            if item == "MODEL.DEVICE":
                skip_next = idx + 1 < len(opts)
                continue
            cleaned_opts.append(item)
        cleaned_opts.extend(["MODEL.DEVICE", device_str])
        args_dict["opts"] = cleaned_opts
        return args_dict

    def _get_parser_compat(device, *args, **kwargs):
        normalized_device = _normalize_device(device)
        parser_args = original_get_parser(normalized_device, *args, **kwargs)
        if isinstance(parser_args, dict):
            parser_args = _with_model_device_override(parser_args, normalized_device)
        return parser_args

    if getattr(_idc.get_parser, "_patched_device", False):
        return

    _idc.get_parser = _get_parser_compat
    _idc.get_parser._patched_device = True  # type: ignore[attr-defined]
    logger.debug("Patched GrIT get_parser for string device compatibility.")


def patch_tokenizer_special_tokens_ids() -> None:
    """
    Patch tag2Text's init_tokenizer to avoid additional_special_tokens_ids.

    Newer transformers removed the dynamic __getattr__ that resolved
    ``additional_special_tokens_ids`` from ``additional_special_tokens``.
    VBench's tag2Text/tag2text.py calls ``tokenizer.additional_special_tokens_ids[0]``
    which fails. We patch init_tokenizer to use convert_tokens_to_ids directly.
    """
    try:
        from transformers import BertTokenizer
        from vbench.third_party.tag2Text import tag2text as _t2t
    except ImportError:
        return

    if getattr(_t2t.init_tokenizer, "_patched", False):
        return

    def _init_tokenizer_compat():
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        return tokenizer

    _t2t.init_tokenizer = _init_tokenizer_compat
    _t2t.init_tokenizer._patched = True  # type: ignore[attr-defined]
    logger.debug("Patched tag2Text.init_tokenizer for tokenizer compatibility.")


def patch_pretrained_model_tied_weights() -> None:
    """
    Add missing all_tied_weights_keys to PreTrainedModel.

    Newer transformers removed the ``all_tied_weights_keys`` property from
    ``PreTrainedModel``. VBench's tag2Text models (inheriting from
    PreTrainedModel) access it during model initialization/checkpoint loading.
    """
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return

    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    # _tied_weights_keys defaults to None on PreTrainedModel; guard against that.
    PreTrainedModel.all_tied_weights_keys = property(
        lambda self: list(getattr(self, "_tied_weights_keys", None) or [])
    )
    logger.debug("Patched PreTrainedModel with all_tied_weights_keys property.")


def patch_config_pruned_heads() -> None:
    """
    Ensure config.pruned_heads is always a dict.

    In some code paths (e.g. from_dict, from_json_file), pruned_heads may end up
    as a list or other non-dict type. The transformers init_weights() and from_dict()
    call ``config.pruned_heads.items()`` which fails on non-dict types.
    """
    try:
        from transformers import PretrainedConfig
    except ImportError:
        return

    if getattr(PretrainedConfig, "_patched_pruned_heads", False):
        return

    original_init = PretrainedConfig.__init__

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Ensure pruned_heads is always a dict
        if not isinstance(getattr(self, "pruned_heads", None), dict):
            self.pruned_heads = {}

    PretrainedConfig.__init__ = _patched_init
    PretrainedConfig._patched_pruned_heads = True  # type: ignore[attr-defined]
    logger.debug("Patched PretrainedConfig to ensure pruned_heads is always a dict.")


def patch_generation_mixin() -> None:
    """
    Add GenerationMixin to tag2Text's BertLMHeadModel.

    In transformers >=4.50, PreTrainedModel no longer inherits from GenerationMixin.
    VBench's tag2Text/med.py BertLMHeadModel needs generate() for scene captioning.
    We add GenerationMixin to the class bases and fix the prepare_inputs_for_generation
    signature to match the new API.
    """
    try:
        from transformers import GenerationConfig, GenerationMixin
        from vbench.third_party.tag2Text import med as _med
    except ImportError:
        return

    lm_head_cls = _med.BertLMHeadModel

    # Already has GenerationMixin in MRO
    if GenerationMixin in lm_head_cls.__mro__:
        return

    # Avoid double-patching
    if getattr(lm_head_cls, "_patched_generation", False):
        return

    # 1. Add GenerationMixin as a base class
    lm_head_cls.__bases__ = (GenerationMixin,) + lm_head_cls.__bases__

    # 2. Fix prepare_inputs_for_generation to use past_key_values (new API)
    #    instead of past (old API from transformers 4.15)
    def _prepare_inputs_for_generation_compat(
        self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs
    ):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        # Cut decoder_input_ids if past KV cache is provided
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    lm_head_cls.prepare_inputs_for_generation = _prepare_inputs_for_generation_compat

    # 3. Wrap generate to ensure generation_config exists.
    #    During __init__, can_generate() returned False (before our patch),
    #    so generation_config was set to None. We lazily create it on first use.
    _orig_generate = GenerationMixin.generate

    def _generate_with_config(self, *args, **kwargs):
        if getattr(self, "generation_config", None) is None:
            self.generation_config = GenerationConfig.from_model_config(self.config)
        return _orig_generate(self, *args, **kwargs)

    lm_head_cls.generate = _generate_with_config

    lm_head_cls._patched_generation = True  # type: ignore[attr-defined]
    logger.debug("Patched BertLMHeadModel with GenerationMixin for generate() support.")


def patch_grit_fast_inference() -> None:
    """
    Patch GrIT inference for ~2-3x speedup.

    Two optimizations applied to VisualizationDemo.run_on_image:
      1. Skip visualization: draw_instance_predictions renders bounding boxes
         and text onto images (CPU-heavy), but none of the 4 GrIT-based
         dimensions (color, object_class, multiple_objects, spatial_relationship)
         use the visualized output.
      2. FP16 autocast: GrIT runs in FP32 by default. Mixed precision halves
         memory bandwidth and improves throughput on modern GPUs.
    """
    try:
        import torch
        from vbench.third_party.grit_src.grit.predictor import VisualizationDemo
    except ImportError:
        return

    if getattr(VisualizationDemo, "_patched_fast", False):
        return

    _orig_run_on_image = VisualizationDemo.run_on_image

    def _run_on_image_fast(self, image):
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                predictions = self.predictor(image)
        else:
            predictions = self.predictor(image)
        # Skip visualization entirely — no caller uses visualized_output
        return predictions, None

    VisualizationDemo.run_on_image = _run_on_image_fast
    VisualizationDemo._patched_fast = True  # type: ignore[attr-defined]
    logger.debug("Patched GrIT VisualizationDemo: skip visualization + FP16 autocast.")


def patch_grit_batch_inference_compat(
    *,
    enable_batch_parallel: bool = False,
    batch_size: int = 1,
) -> None:
    """
    Patch VisualizationDemo.run_on_batch with a runtime mode switch.

    Safe mode (default):
      - uses sequential single-image predictor calls
      - enforces batch_size == 1 via assertion
    Parallel mode:
      - uses chunked model(batch_inputs) for acceleration
      - strict mode: any model-side assertion/error bubbles up (no auto-fallback)
    """
    try:
        import torch
        from vbench.third_party.grit_src.grit.predictor import VisualizationDemo
    except ImportError:
        return

    requested_batch_size = max(1, int(batch_size))
    effective_batch_size = requested_batch_size if enable_batch_parallel else 1
    if not enable_batch_parallel and requested_batch_size != 1:
        logger.warning(
            "grit_batch_parallel_enable=false, forcing grit_batch_size from %d to 1",
            requested_batch_size,
        )

    if not getattr(VisualizationDemo, "_patched_batch_compat", False):
        VisualizationDemo._orig_run_on_batch = VisualizationDemo.run_on_batch  # type: ignore[attr-defined]

        def _run_on_batch_switch(self, image_arrays):
            parallel_enabled = bool(getattr(VisualizationDemo, "_grit_batch_parallel_enable", False))
            configured_bs = int(getattr(VisualizationDemo, "_grit_batch_size", 1))

            def _run_chunk_sequential(chunk_images):
                chunk_preds = []
                for image in chunk_images:
                    if torch.cuda.is_available():
                        with torch.amp.autocast("cuda"):
                            chunk_preds.append(self.predictor(image))
                    else:
                        chunk_preds.append(self.predictor(image))
                return chunk_preds

            if not parallel_enabled:
                assert configured_bs == 1, (
                    "Safe mode requires grit_batch_size == 1 when "
                    "grit_batch_parallel_enable is false"
                )
                return _run_chunk_sequential(image_arrays)

            bs = max(1, configured_bs)
            predictions = []
            for start in range(0, len(image_arrays), bs):
                chunk = image_arrays[start : start + bs]
                with torch.no_grad():
                    batch_inputs = []
                    for image in chunk:
                        height, width = image.shape[:2]
                        self.predictor.aug.get_transform(image)
                        tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                        batch_inputs.append({"image": tensor, "height": height, "width": width})
                    if torch.cuda.is_available():
                        with torch.amp.autocast("cuda"):
                            chunk_preds = self.predictor.model(batch_inputs)
                    else:
                        chunk_preds = self.predictor.model(batch_inputs)
                predictions.extend(chunk_preds)
            return predictions

        VisualizationDemo.run_on_batch = _run_on_batch_switch
        VisualizationDemo._patched_batch_compat = True  # type: ignore[attr-defined]

    VisualizationDemo._grit_batch_parallel_enable = bool(enable_batch_parallel)  # type: ignore[attr-defined]
    VisualizationDemo._grit_batch_size = int(effective_batch_size)  # type: ignore[attr-defined]
    logger.info(
        "Configured GrIT run_on_batch mode: parallel=%s batch_size=%d",
        bool(enable_batch_parallel),
        int(effective_batch_size),
    )


def patch_color_object_matching() -> None:
    """
    Patch color dimension for custom video prompts.

    VBench's color check_generate uses exact match (``object_key == pred[1]``)
    to compare a stripped prompt string against GrIT detection labels.  For
    official VBench prompts like "a red car" this works (stripped to "car"),
    but custom prompts like "a red car driving on a highway" strip to
    "car driving on highway" which never matches the GrIT label "car".

    Also guards against ZeroDivisionError when no objects are detected.
    """
    try:
        from vbench import color as _color
    except ImportError:
        return

    if getattr(_color, "_patched_color_matching", False):
        return

    def _check_generate_lenient(color_key, object_key, predictions):
        cur_object_color, cur_object = 0, 0
        object_key_lower = object_key.lower().strip()
        for frame_pred in predictions:
            object_flag, color_flag = False, False
            for pred in frame_pred:
                pred_type = str(pred[1]).lower().strip() if pred[1] else ""
                # Lenient: GrIT label is a substring of object_key (e.g. "car" in "car driving")
                if pred_type and pred_type in object_key_lower:
                    for color_query in [
                        "white",
                        "red",
                        "pink",
                        "blue",
                        "silver",
                        "purple",
                        "orange",
                        "green",
                        "gray",
                        "yellow",
                        "black",
                        "grey",
                    ]:
                        if color_query in pred[0]:
                            object_flag = True
                    if color_key in pred[0]:
                        color_flag = True
            if color_flag:
                cur_object_color += 1
            if object_flag:
                cur_object += 1
        return cur_object, cur_object_color

    _orig_color_fn = _color.color

    def _color_safe(model, video_dict, device):
        """Wrap color() with the lenient check_generate and zero-division guard."""
        orig_cg = _color.check_generate
        _color.check_generate = _check_generate_lenient
        try:
            result = _orig_color_fn(model, video_dict, device)
        except ZeroDivisionError:
            logger.warning("color dimension: no objects detected in any video, returning score 0.")
            result = (0.0, [])
        finally:
            _color.check_generate = orig_cg
        return result

    _color.color = _color_safe
    _color._patched_color_matching = True  # type: ignore[attr-defined]
    logger.debug("Patched color check_generate for lenient object matching.")


def apply_vbench_compat_patches(
    *,
    grit_batch_parallel_enable: bool = False,
    grit_batch_size: int = 1,
) -> None:
    """Apply all VBench compatibility patches."""
    patch_transformers_compat()
    patch_pretrained_model_tied_weights()
    patch_config_pruned_heads()
    patch_tokenizer_special_tokens_ids()
    patch_clip_tokenize_truncate()
    patch_grit_device_compat()
    patch_generation_mixin()
    patch_grit_fast_inference()
    patch_grit_batch_inference_compat(
        enable_batch_parallel=grit_batch_parallel_enable,
        batch_size=grit_batch_size,
    )
    patch_color_object_matching()
    patch_human_action_prompt_matching()


def patch_human_action_prompt_matching() -> None:
    """
    Patch human_action to use prompt-based Kinetics-400 matching.

    VBench's human_action extracts action labels from video filenames
    (e.g. "person is running-001.mp4" -> "running"), which only works for
    VBench's official benchmark videos. This patch matches prompt text against
    Kinetics-400 categories so custom videos get meaningful scores.
    """
    try:
        from vbench import human_action as _ha
        from vbench.utils import load_dimension_info
    except ImportError:
        return

    if getattr(_ha, "_patched_prompt_matching", False):
        return

    # Build K-400 category list sorted longest-first for greedy matching
    _k400_cats = sorted(_ha.build_dict().values(), key=len, reverse=True)

    def _match_k400(prompt_text):
        """Find the longest K-400 category substring in the prompt."""
        prompt_lower = prompt_text.lower()
        for cat in _k400_cats:
            if cat in prompt_lower:
                return cat
        return None

    _orig_compute = _ha.compute_human_action

    def _patched_compute_human_action(json_dir, device, submodules_list, **kwargs):
        from vbench.distributed import (
            distribute_list_to_rank,
            gather_list_of_dict,
            get_world_size,
        )

        umt_path = submodules_list[0]
        video_list, prompt_dict_ls = load_dimension_info(
            json_dir, dimension="human_action", lang="en"
        )

        # Build video_path -> K-400 label mapping from prompt text
        label_map = {}
        for info in prompt_dict_ls:
            prompt_text = info.get("prompt", "")
            label = _match_k400(prompt_text)
            for vp in info.get("video_list", []):
                label_map[vp] = label

        video_list = distribute_list_to_rank(video_list)
        all_results, video_results = _human_action_with_labels(
            umt_path, video_list, device, label_map
        )
        if get_world_size() > 1:
            video_results = gather_list_of_dict(video_results)
            all_results = sum(d["cor_num_per_video"] for d in video_results) / len(video_results)
        return all_results, video_results

    def _human_action_with_labels(umt_path, video_list, device, label_map):
        """Evaluate human_action using prompt-derived labels instead of filenames."""
        import torch
        from timm.models import create_model
        from tqdm import tqdm
        from vbench.distributed import get_rank
        from vbench.third_party.umt.datasets.video_transforms import (
            CenterCrop,
            Compose,
            Normalize,
            Resize,
        )
        from vbench.third_party.umt.datasets.volume_transforms import ClipToTensor
        from vbench.third_party.umt.models.modeling_finetune import (  # noqa: F401
            vit_large_patch16_224,
        )
        from vbench.utils import load_video

        state_dict = torch.load(umt_path, map_location="cpu")
        model = create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=400,
            all_frames=16,
            tubelet_size=1,
            use_learnable_pos_emb=False,
            fc_drop_rate=0.0,
            drop_rate=0.0,
            drop_path_rate=0.2,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_checkpoint=False,
            checkpoint_num=16,
            use_mean_pooling=True,
            init_scale=0.001,
        )
        data_transform = Compose(
            [
                Resize(256, interpolation="bilinear"),
                CenterCrop(size=(224, 224)),
                ClipToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        model = model.to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        cat_dict = _ha.build_dict()
        cnt = 0
        cor_num = 0
        video_results = []

        for video_path in tqdm(video_list, disable=get_rank() > 0):
            # Prompt-based label with filename fallback
            video_label = label_map.get(video_path)
            if video_label is None:
                video_label = (
                    video_path.split("/")[-1]
                    .lower()
                    .split("-")[0]
                    .split("person is ")[-1]
                    .split("_")[0]
                )

            cnt += 1
            images = load_video(video_path, data_transform, num_frames=16)
            images = images.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = torch.sigmoid(model(images))
                top_results, indices = torch.topk(logits, 5, dim=1)

            indices = indices.squeeze().tolist()
            scores = [round(f, 4) for f in top_results.squeeze().tolist()]

            cat_ls = [cat_dict[str(indices[i])] for i in range(5) if scores[i] >= 0.85]

            cor_num_per_video = 0
            flag = False
            for cat in cat_ls:
                if cat == video_label:
                    cor_num += 1
                    cor_num_per_video = 1
                    flag = True
                    break

            video_results.append(
                {
                    "video_path": video_path,
                    "video_results": flag,
                    "cor_num_per_video": cor_num_per_video,
                }
            )

        acc = cor_num / cnt if cnt > 0 else 0
        return acc, video_results

    _ha.compute_human_action = _patched_compute_human_action
    _ha._patched_prompt_matching = True  # type: ignore[attr-defined]
    logger.debug("Patched human_action for prompt-based K-400 category matching.")
