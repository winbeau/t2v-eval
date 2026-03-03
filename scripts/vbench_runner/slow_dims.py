"""
Fused execution for VBench slow dimensions.

This path keeps scoring helpers unchanged while sharing:
  - clip decode
  - GrIT ObjectDet inference (object_class / multiple_objects / spatial_relationship)
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import transforms

try:
    from .env import logger
    from .results import extract_subtask_scores, resolve_video_id
except ImportError:
    from vbench_runner.env import logger
    from vbench_runner.results import extract_subtask_scores, resolve_video_id

SLOW_DIMS: tuple[str, ...] = (
    "object_class",
    "multiple_objects",
    "spatial_relationship",
    "color",
)
SLOW_DIM_SET: set[str] = set(SLOW_DIMS)


def split_video_ids_for_rank(valid_video_ids: set[str], rank: int, world_size: int) -> set[str]:
    """Deterministic shard of base video ids for current rank."""
    if world_size <= 1:
        return set(valid_video_ids)
    ordered = sorted(valid_video_ids)
    return {video_id for idx, video_id in enumerate(ordered) if idx % world_size == rank}


def _resize_tensor_if_needed(video_tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    _, _, h, w = video_tensor.size()
    if min(h, w) <= 768:
        return video_tensor, 0.0
    t0 = time.perf_counter()
    scale = 720.0 / min(h, w)
    resized = transforms.Resize(size=(int(scale * h), int(scale * w)))(video_tensor)
    return resized, time.perf_counter() - t0


def _resize_color_arrays_if_needed(video_arrays: np.ndarray) -> tuple[np.ndarray, float]:
    _, h, w, _ = video_arrays.shape
    if min(h, w) <= 768:
        return video_arrays, 0.0
    t0 = time.perf_counter()
    scale = 720.0 / min(h, w)
    new_h = int(scale * h)
    new_w = int(scale * w)
    resized = np.zeros((video_arrays.shape[0], new_h, new_w, 3), dtype=video_arrays.dtype)
    for i in range(video_arrays.shape[0]):
        resized[i] = cv2.resize(video_arrays[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, time.perf_counter() - t0


def _object_preds_from_captions(captions_list: list) -> list[set[str]]:
    preds: list[set[str]] = []
    for caption in captions_list:
        try:
            preds.append(set(caption[0][2]))
        except Exception:
            preds.append(set())
    return preds


def _multiple_preds_from_captions(captions_list: list) -> list[set[str]]:
    preds: list[set[str]] = []
    for caption in captions_list:
        if len(caption) > 0:
            preds.append(set(caption[0][2]))
        else:
            preds.append(set())
    return preds


def _spatial_preds_from_captions(captions_list: list) -> list[list[list]]:
    preds: list[list[list]] = []
    for caption in captions_list:
        cur: list[list] = []
        if len(caption) > 0:
            for info in caption:
                cur.append([info[0], info[1]])
        preds.append(cur)
    return preds


def _color_preds_from_captions(captions_list: list) -> list[list[list[str]]]:
    preds: list[list[list[str]]] = []
    for caption in captions_list:
        cur: list[list[str]] = []
        if len(caption) < 1:
            cur.append(["", ""])
        else:
            for cap_det in caption:
                cur.append([cap_det[0], cap_det[2][0]])
        preds.append(cur)
    return preds


def _collect_clip_meta(full_info_path: Path, dimension: str) -> dict[str, dict]:
    from vbench.utils import load_dimension_info

    _, prompt_dict_ls = load_dimension_info(str(full_info_path), dimension=dimension, lang="en")
    clip_meta: dict[str, dict] = {}
    for info in prompt_dict_ls:
        if "auxiliary_info" not in info:
            raise ValueError(
                f"Missing auxiliary_info for dimension `{dimension}` in {full_info_path}."
            )
        prompt = str(info.get("prompt", ""))
        auxiliary_info = info["auxiliary_info"]
        for video_path in info.get("video_list", []):
            clip_meta[str(video_path)] = {
                "prompt": prompt,
                "auxiliary_info": auxiliary_info,
            }
    return clip_meta


def run_fused_slow_dimensions(
    *,
    full_info_path: Path,
    subtasks: list[str],
    valid_video_ids: set[str],
    rank: int,
    world_size: int,
    device: str,
    local: bool = True,
    read_frame: bool = False,
    progress_callback: Any = None,
    decode_workers: int = 1,
    decode_prefetch: int = 2,
) -> tuple[list[dict], dict[str, float | int]]:
    """
    Run slow dimensions with shared decode/inference and return per-video rows.

    Returns:
        (rows, stats)
    """
    active_dims = [dim for dim in subtasks if dim in SLOW_DIM_SET]
    if not active_dims:
        return [], {
            "clips_total": 0,
            "clips_shard": 0,
            "decode_sec": 0.0,
            "resize_sec": 0.0,
            "det_infer_sec": 0.0,
            "color_infer_sec": 0.0,
            "init_det_sec": 0.0,
            "init_color_sec": 0.0,
        }

    import vbench.color as color_mod
    import vbench.multiple_objects as multi_mod
    import vbench.object_class as object_mod
    import vbench.spatial_relationship as spatial_mod
    from vbench.third_party.grit_model import DenseCaptioning
    from vbench.utils import init_submodules, load_video

    submodules = init_submodules(active_dims, local=local, read_frame=read_frame)

    det_model = None
    det_weight = None
    for name in ("object_class", "multiple_objects", "spatial_relationship"):
        if name in active_dims:
            det_weight = submodules[name]["model_weight"]
            break

    cap_model = None
    color_weight = submodules["color"]["model_weight"] if "color" in active_dims else None

    init_det_sec = 0.0
    init_color_sec = 0.0

    if det_weight is not None:
        det_model = DenseCaptioning(device)
        t0 = time.perf_counter()
        det_model.initialize_model_det(model_weight=det_weight)
        init_det_sec = time.perf_counter() - t0

    if color_weight is not None:
        cap_model = DenseCaptioning(device)
        t0 = time.perf_counter()
        cap_model.initialize_model(model_weight=color_weight)
        init_color_sec = time.perf_counter() - t0

    dim_meta: dict[str, dict[str, dict]] = {}
    all_clip_paths: set[str] = set()
    for dim in active_dims:
        clip_meta = _collect_clip_meta(full_info_path, dim)
        dim_meta[dim] = clip_meta
        all_clip_paths.update(clip_meta.keys())

    assigned_video_ids = split_video_ids_for_rank(valid_video_ids, rank=rank, world_size=world_size)
    clip_paths = [
        path
        for path in sorted(all_clip_paths)
        if resolve_video_id(path, valid_video_ids) in assigned_video_ids
    ]

    clip_results: dict[str, list[dict]] = {dim: [] for dim in active_dims}

    decode_sec = 0.0
    resize_sec = 0.0
    det_infer_sec = 0.0
    color_infer_sec = 0.0
    decode_workers = max(1, int(decode_workers))
    decode_prefetch = max(1, int(decode_prefetch))

    def _decode_clip(path: str) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        arrays = load_video(path, num_frames=16, return_tensor=False)
        return arrays, time.perf_counter() - t0

    def _process_clip(video_path: str, video_arrays: np.ndarray) -> None:
        nonlocal resize_sec
        nonlocal det_infer_sec
        nonlocal color_infer_sec

        need_det = (
            det_model is not None
            and any(
                dim in dim_meta and video_path in dim_meta[dim]
                for dim in ("object_class", "multiple_objects", "spatial_relationship")
            )
        )
        if need_det:
            video_tensor = torch.Tensor(video_arrays).permute(0, 3, 1, 2)
            video_tensor, resize_dt = _resize_tensor_if_needed(video_tensor)
            resize_sec += resize_dt

            det_inputs = video_tensor.permute(0, 2, 3, 1).numpy()
            t0 = time.perf_counter()
            captions_det = det_model.run_caption_tensor_batch(det_inputs)
            det_infer_sec += time.perf_counter() - t0

            if "object_class" in dim_meta and video_path in dim_meta["object_class"]:
                object_meta = dim_meta["object_class"][video_path]
                object_info = object_meta["auxiliary_info"]["object"]
                object_preds = _object_preds_from_captions(captions_det)
                success = object_mod.check_generate(object_info, object_preds)
                score = success / len(object_preds)
                clip_results["object_class"].append(
                    {"video_path": video_path, "video_results": score}
                )

            if "multiple_objects" in dim_meta and video_path in dim_meta["multiple_objects"]:
                multi_meta = dim_meta["multiple_objects"][video_path]
                multi_info = multi_meta["auxiliary_info"]["object"]
                multi_preds = _multiple_preds_from_captions(captions_det)
                success = multi_mod.check_generate(multi_info, multi_preds)
                score = success / len(multi_preds)
                clip_results["multiple_objects"].append(
                    {"video_path": video_path, "video_results": score}
                )

            if "spatial_relationship" in dim_meta and video_path in dim_meta["spatial_relationship"]:
                spatial_meta = dim_meta["spatial_relationship"][video_path]
                spatial_info = spatial_meta["auxiliary_info"]["spatial_relationship"]
                spatial_preds = _spatial_preds_from_captions(captions_det)
                frame_scores = spatial_mod.check_generate(spatial_info, spatial_preds)
                score = float(np.mean(frame_scores))
                clip_results["spatial_relationship"].append(
                    {"video_path": video_path, "video_results": score}
                )

        if "color" in dim_meta and video_path in dim_meta["color"]:
            if cap_model is None:
                raise RuntimeError("color model is not initialized")

            color_arrays, resize_dt = _resize_color_arrays_if_needed(video_arrays)
            resize_sec += resize_dt

            t0 = time.perf_counter()
            captions_color = cap_model.run_caption_tensor_batch(color_arrays)
            color_infer_sec += time.perf_counter() - t0

            color_preds = _color_preds_from_captions(captions_color)
            color_meta = dim_meta["color"][video_path]
            color_info = color_meta["auxiliary_info"]["color"]
            prompt_text = color_meta["prompt"]
            object_key = prompt_text.replace("a ", "").replace("an ", "").replace(
                color_info, ""
            ).strip()
            cur_object, cur_object_color = color_mod.check_generate(
                color_info, object_key, color_preds
            )
            if cur_object > 0:
                score = cur_object_color / cur_object
                clip_results["color"].append(
                    {"video_path": video_path, "video_results": score}
                )

    start_ts = time.perf_counter()
    total_clips = len(clip_paths)
    processed = 0

    if clip_paths:
        if progress_callback is not None:
            progress_callback(
                {
                    "percent": 0,
                    "status_text": f"fused_clips 0/{total_clips}",
                    "elapsed_sec": 0,
                }
            )
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            inflight: deque[tuple[str, Any]] = deque()
            next_idx = 0

            def _fill_inflight() -> None:
                nonlocal next_idx
                while next_idx < len(clip_paths) and len(inflight) < decode_prefetch:
                    path = clip_paths[next_idx]
                    next_idx += 1
                    inflight.append((path, pool.submit(_decode_clip, path)))

            _fill_inflight()
            while inflight:
                current_path, future = inflight.popleft()
                video_arrays, decode_dt = future.result()
                decode_sec += decode_dt
                _process_clip(current_path, video_arrays)
                processed += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "percent": int(99 * processed / max(total_clips, 1)),
                            "status_text": f"fused_clips {processed}/{total_clips}",
                            "elapsed_sec": int(time.perf_counter() - start_ts),
                        }
                    )
                _fill_inflight()
    elif progress_callback is not None:
        progress_callback(
            {
                "percent": 99,
                "status_text": "fused_clips 0/0",
                "elapsed_sec": 0,
            }
        )

    rows: list[dict] = []
    for dim in active_dims:
        rows.extend(
            extract_subtask_scores(
                dimension_data=[None, clip_results[dim]],
                subtask=dim,
                valid_video_ids=valid_video_ids,
                long_mode=True,
            )
        )

    stats: dict[str, float | int] = {
        "clips_total": len(all_clip_paths),
        "clips_shard": len(clip_paths),
        "decode_sec": round(decode_sec, 4),
        "resize_sec": round(resize_sec, 4),
        "det_infer_sec": round(det_infer_sec, 4),
        "color_infer_sec": round(color_infer_sec, 4),
        "init_det_sec": round(init_det_sec, 4),
        "init_color_sec": round(init_color_sec, 4),
    }

    logger.info(
        "[rank %d] slow-dims fused stats: clips %d/%d decode=%.2fs resize=%.2fs det=%.2fs color=%.2fs decode_workers=%d prefetch=%d",
        rank,
        stats["clips_shard"],
        stats["clips_total"],
        stats["decode_sec"],
        stats["resize_sec"],
        stats["det_infer_sec"],
        stats["color_infer_sec"],
        decode_workers,
        decode_prefetch,
    )
    return rows, stats
