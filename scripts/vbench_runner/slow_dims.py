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


def _parse_cuda_index(device: str) -> int:
    raw = str(device or "")
    if ":" in raw:
        try:
            return max(0, int(raw.split(":", 1)[1]))
        except (TypeError, ValueError):
            return 0
    return 0


def _select_clip_paths_for_rank(
    *,
    all_clip_paths: set[str],
    valid_video_ids: set[str],
    rank: int,
    world_size: int,
    shard_mode: str,
) -> list[str]:
    ordered = sorted(all_clip_paths)
    if world_size <= 1:
        return ordered
    if str(shard_mode).strip().lower() == "video":
        assigned_video_ids = split_video_ids_for_rank(valid_video_ids, rank=rank, world_size=world_size)
        return [
            path
            for path in ordered
            if resolve_video_id(path, valid_video_ids) in assigned_video_ids
        ]
    # default: clip-level sharding for better load balance.
    return [path for idx, path in enumerate(ordered) if idx % world_size == rank]


def _decode_clip_decord(
    path: str,
    *,
    num_frames: int,
    device: str,
    decode_backend: str,
):
    from decord import VideoReader, cpu, gpu
    from vbench.utils import get_frame_indices

    backend = str(decode_backend).strip().lower()
    if backend == "decord_gpu":
        ctx = gpu(_parse_cuda_index(device))
    else:
        ctx = cpu(0)
    reader = VideoReader(path, ctx=ctx, num_threads=1)
    frame_indices = get_frame_indices(num_frames, len(reader), sample="middle")
    batch = reader.get_batch(frame_indices)
    return batch.asnumpy().astype(np.uint8)


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


def _resolve_det_exec_plan(
    *,
    det_dims_present: list[str],
    det_single_image_dims_set: set[str],
    det_adaptive_state: dict[str, dict[str, float | int | bool]],
    det_adaptive_probe_clips: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Resolve detection execution mode for current clip.

    Returns:
        (det_dims_batch, det_dims_single, probe_dims)
    """
    forced_single_dims = set(det_single_image_dims_set)
    forced_single_dims.update(
        dim
        for dim, state in det_adaptive_state.items()
        if bool(state.get("triggered", False))
    )
    probe_dims = [
        dim
        for dim in det_dims_present
        if dim in det_adaptive_state
        and dim not in det_single_image_dims_set
        and not bool(det_adaptive_state[dim].get("triggered", False))
        and int(det_adaptive_state[dim].get("probed", 0)) < det_adaptive_probe_clips
    ]

    det_dims_single = [dim for dim in det_dims_present if dim in forced_single_dims or dim in probe_dims]
    det_dims_batch = [dim for dim in det_dims_present if dim not in forced_single_dims]
    return det_dims_batch, det_dims_single, probe_dims


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
    decode_backend: str = "legacy",
    shard_mode: str = "clip",
    profile_window_clips: int = 100,
    stage_profile: bool = True,
    det_single_image_dims: list[str] | None = None,
    det_adaptive_enabled: bool = False,
    det_adaptive_dims: list[str] | None = None,
    det_adaptive_probe_clips: int = 8,
    det_adaptive_score_threshold: float = 0.02,
) -> tuple[list[dict], dict[str, float | int]]:
    """
    Run slow dimensions with shared decode/inference and return per-video rows.

    Returns:
        (rows, stats)
    """
    active_dims = [dim for dim in subtasks if dim in SLOW_DIM_SET]
    det_single_image_dims_set = set(det_single_image_dims or []) & {
        "object_class",
        "multiple_objects",
        "spatial_relationship",
    }
    if det_single_image_dims_set:
        logger.info(
            "[rank %d] Slow-dims low-drift single-image det dims: %s",
            rank,
            sorted(det_single_image_dims_set),
        )
    det_adaptive_probe_clips = max(0, int(det_adaptive_probe_clips))
    det_adaptive_score_threshold = max(0.0, float(det_adaptive_score_threshold))
    det_adaptive_candidates = set(det_adaptive_dims or []) & {
        "object_class",
        "multiple_objects",
        "spatial_relationship",
    }
    if det_single_image_dims_set:
        det_adaptive_candidates -= det_single_image_dims_set
    if not det_adaptive_enabled:
        det_adaptive_candidates = set()
    det_adaptive_state: dict[str, dict[str, float | int | bool]] = {
        dim: {"probed": 0, "triggered": False, "max_abs_diff": 0.0}
        for dim in sorted(det_adaptive_candidates)
    }
    if det_adaptive_state:
        logger.info(
            "[rank %d] Slow-dims adaptive fallback enabled: dims=%s probe_clips=%d score_threshold=%.6f",
            rank,
            sorted(det_adaptive_state.keys()),
            det_adaptive_probe_clips,
            det_adaptive_score_threshold,
        )
    if not active_dims:
        return [], {
            "clips_total": 0,
            "clips_shard": 0,
            "decode_sec": 0.0,
            "tensorize_sec": 0.0,
            "resize_sec": 0.0,
            "det_infer_sec": 0.0,
            "color_infer_sec": 0.0,
            "score_sec": 0.0,
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

    clip_paths = _select_clip_paths_for_rank(
        all_clip_paths=all_clip_paths,
        valid_video_ids=valid_video_ids,
        rank=rank,
        world_size=world_size,
        shard_mode=shard_mode,
    )

    clip_results: dict[str, list[dict]] = {dim: [] for dim in active_dims}

    decode_sec = 0.0
    tensorize_sec = 0.0
    resize_sec = 0.0
    det_infer_sec = 0.0
    color_infer_sec = 0.0
    score_sec = 0.0
    decode_workers = max(1, int(decode_workers))
    decode_prefetch = max(1, int(decode_prefetch))
    profile_window_clips = max(1, int(profile_window_clips))
    decode_backend = str(decode_backend).strip().lower() or "legacy"
    if decode_backend not in {"legacy", "decord", "decord_gpu"}:
        logger.warning("Unknown decode_backend=%s, fallback to legacy", decode_backend)
        decode_backend = "legacy"
    active_decode_backend = decode_backend
    decode_backend_warned = False

    def _decode_clip(path: str) -> tuple[np.ndarray, float]:
        nonlocal active_decode_backend
        nonlocal decode_backend_warned
        t0 = time.perf_counter()
        if active_decode_backend == "legacy":
            arrays = load_video(path, num_frames=16, return_tensor=False)
        else:
            try:
                arrays = _decode_clip_decord(
                    path,
                    num_frames=16,
                    device=device,
                    decode_backend=active_decode_backend,
                )
            except Exception as exc:
                # Keep robustness across heterogeneous server builds; downgrade once.
                if not decode_backend_warned:
                    logger.warning(
                        "[rank %d] decode backend `%s` failed (%s), fallback to legacy",
                        rank,
                        active_decode_backend,
                        exc,
                    )
                    decode_backend_warned = True
                active_decode_backend = "legacy"
                arrays = load_video(path, num_frames=16, return_tensor=False)
        return arrays, time.perf_counter() - t0

    def _process_clip(video_path: str, video_arrays: np.ndarray) -> dict[str, float]:
        nonlocal resize_sec
        nonlocal det_infer_sec
        nonlocal color_infer_sec
        nonlocal tensorize_sec
        nonlocal score_sec

        local_times = {
            "tensorize": 0.0,
            "resize": 0.0,
            "det": 0.0,
            "color": 0.0,
            "score": 0.0,
        }

        need_det = (
            det_model is not None
            and any(
                dim in dim_meta and video_path in dim_meta[dim]
                for dim in ("object_class", "multiple_objects", "spatial_relationship")
            )
        )
        if need_det:
            t0 = time.perf_counter()
            video_tensor = torch.from_numpy(video_arrays).permute(0, 3, 1, 2)
            local_times["tensorize"] += time.perf_counter() - t0
            video_tensor, resize_dt = _resize_tensor_if_needed(video_tensor)
            resize_sec += resize_dt
            local_times["resize"] += resize_dt

            det_inputs = video_tensor.permute(0, 2, 3, 1).numpy()
            det_dims_present = [
                dim
                for dim in ("object_class", "multiple_objects", "spatial_relationship")
                if dim in dim_meta and video_path in dim_meta[dim]
            ]
            det_dims_batch, det_dims_single, probe_dims = _resolve_det_exec_plan(
                det_dims_present=det_dims_present,
                det_single_image_dims_set=det_single_image_dims_set,
                det_adaptive_state=det_adaptive_state,
                det_adaptive_probe_clips=det_adaptive_probe_clips,
            )

            captions_det_batch = None
            captions_det_single = None
            if det_dims_batch:
                t0 = time.perf_counter()
                captions_det_batch = det_model.run_caption_tensor_batch(det_inputs)
                det_dt = time.perf_counter() - t0
                det_infer_sec += det_dt
                local_times["det"] += det_dt
            if det_dims_single:
                t0 = time.perf_counter()
                captions_det_single = []
                for frame in det_inputs:
                    cap, _ = det_model.run_caption_tensor(frame)
                    captions_det_single.append(cap)
                det_dt = time.perf_counter() - t0
                det_infer_sec += det_dt
                local_times["det"] += det_dt

            def _captions_for_dim(dim_name: str, use_single: bool):
                selected = captions_det_single if use_single else captions_det_batch
                if selected is None:
                    raise RuntimeError(
                        f"Missing detection captions for dim={dim_name}, "
                        f"use_single={use_single}, video={video_path}, "
                        f"batch_dims={det_dims_batch}, single_dims={det_dims_single}, probe_dims={probe_dims}"
                    )
                return selected

            def _score_object_class(captions_list) -> float:
                nonlocal score_sec
                object_meta = dim_meta["object_class"][video_path]
                object_info = object_meta["auxiliary_info"]["object"]
                object_preds = _object_preds_from_captions(captions_list)
                t0 = time.perf_counter()
                success = object_mod.check_generate(object_info, object_preds)
                score_dt = time.perf_counter() - t0
                score_sec += score_dt
                local_times["score"] += score_dt
                return success / len(object_preds)

            def _score_multiple_objects(captions_list) -> float:
                nonlocal score_sec
                multi_meta = dim_meta["multiple_objects"][video_path]
                multi_info = multi_meta["auxiliary_info"]["object"]
                multi_preds = _multiple_preds_from_captions(captions_list)
                t0 = time.perf_counter()
                success = multi_mod.check_generate(multi_info, multi_preds)
                score_dt = time.perf_counter() - t0
                score_sec += score_dt
                local_times["score"] += score_dt
                return success / len(multi_preds)

            def _score_spatial_relationship(captions_list) -> float:
                nonlocal score_sec
                spatial_meta = dim_meta["spatial_relationship"][video_path]
                spatial_info = spatial_meta["auxiliary_info"]["spatial_relationship"]
                spatial_preds = _spatial_preds_from_captions(captions_list)
                t0 = time.perf_counter()
                frame_scores = spatial_mod.check_generate(spatial_info, spatial_preds)
                score_dt = time.perf_counter() - t0
                score_sec += score_dt
                local_times["score"] += score_dt
                return float(np.mean(frame_scores))

            def _pick_dim_score(dim_name: str, score_fn) -> float:
                state = det_adaptive_state.get(dim_name)
                probing_now = dim_name in probe_dims and state is not None
                if probing_now:
                    score_batch = score_fn(_captions_for_dim(dim_name, use_single=False))
                    score_single = score_fn(_captions_for_dim(dim_name, use_single=True))
                    abs_diff = abs(float(score_batch) - float(score_single))
                    state["probed"] = int(state.get("probed", 0)) + 1
                    state["max_abs_diff"] = max(float(state.get("max_abs_diff", 0.0)), abs_diff)
                    if abs_diff > det_adaptive_score_threshold:
                        state["triggered"] = True
                        logger.warning(
                            "[rank %d] Slow-dims adaptive fallback triggered for %s: "
                            "abs(score_batch-score_single)=%.6f > %.6f (video=%s)",
                            rank,
                            dim_name,
                            abs_diff,
                            det_adaptive_score_threshold,
                            video_path,
                        )
                        return float(score_single)
                    return float(score_batch)

                use_single = dim_name in det_single_image_dims_set or (
                    state is not None and bool(state.get("triggered", False))
                )
                return float(score_fn(_captions_for_dim(dim_name, use_single=use_single)))

            if "object_class" in dim_meta and video_path in dim_meta["object_class"]:
                score = _pick_dim_score("object_class", _score_object_class)
                clip_results["object_class"].append(
                    {"video_path": video_path, "video_results": score}
                )

            if "multiple_objects" in dim_meta and video_path in dim_meta["multiple_objects"]:
                score = _pick_dim_score("multiple_objects", _score_multiple_objects)
                clip_results["multiple_objects"].append(
                    {"video_path": video_path, "video_results": score}
                )

            if "spatial_relationship" in dim_meta and video_path in dim_meta["spatial_relationship"]:
                score = _pick_dim_score("spatial_relationship", _score_spatial_relationship)
                clip_results["spatial_relationship"].append(
                    {"video_path": video_path, "video_results": score}
                )

        if "color" in dim_meta and video_path in dim_meta["color"]:
            if cap_model is None:
                raise RuntimeError("color model is not initialized")

            color_arrays, resize_dt = _resize_color_arrays_if_needed(video_arrays)
            resize_sec += resize_dt
            local_times["resize"] += resize_dt

            t0 = time.perf_counter()
            captions_color = cap_model.run_caption_tensor_batch(color_arrays)
            color_dt = time.perf_counter() - t0
            color_infer_sec += color_dt
            local_times["color"] += color_dt

            color_preds = _color_preds_from_captions(captions_color)
            color_meta = dim_meta["color"][video_path]
            color_info = color_meta["auxiliary_info"]["color"]
            prompt_text = color_meta["prompt"]
            object_key = prompt_text.replace("a ", "").replace("an ", "").replace(
                color_info, ""
            ).strip()
            t0 = time.perf_counter()
            cur_object, cur_object_color = color_mod.check_generate(
                color_info, object_key, color_preds
            )
            score_dt = time.perf_counter() - t0
            score_sec += score_dt
            local_times["score"] += score_dt
            if cur_object > 0:
                score = cur_object_color / cur_object
                clip_results["color"].append(
                    {"video_path": video_path, "video_results": score}
                )
        tensorize_sec += local_times["tensorize"]
        return local_times

    start_ts = time.perf_counter()
    total_clips = len(clip_paths)
    processed = 0
    window_start_ts = start_ts
    window_stage_totals = {
        "decode": 0.0,
        "tensorize": 0.0,
        "resize": 0.0,
        "det": 0.0,
        "color": 0.0,
        "score": 0.0,
    }
    window_processed = 0

    def _maybe_log_window(final: bool = False) -> None:
        nonlocal window_start_ts
        nonlocal window_processed
        if not stage_profile:
            return
        if window_processed <= 0:
            return
        if not final and window_processed < profile_window_clips:
            return
        elapsed = max(1e-6, time.perf_counter() - window_start_ts)
        c = float(window_processed)
        logger.info(
            "[rank %d] slow-dims window: clips=%d clip/s=%.3f decode=%.3fms tensorize=%.3fms resize=%.3fms det=%.3fms color=%.3fms score=%.3fms",
            rank,
            window_processed,
            c / elapsed,
            1e3 * window_stage_totals["decode"] / c,
            1e3 * window_stage_totals["tensorize"] / c,
            1e3 * window_stage_totals["resize"] / c,
            1e3 * window_stage_totals["det"] / c,
            1e3 * window_stage_totals["color"] / c,
            1e3 * window_stage_totals["score"] / c,
        )
        for key in window_stage_totals:
            window_stage_totals[key] = 0.0
        window_processed = 0
        window_start_ts = time.perf_counter()

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
                stage_times = _process_clip(current_path, video_arrays)
                processed += 1
                window_processed += 1
                window_stage_totals["decode"] += decode_dt
                window_stage_totals["tensorize"] += stage_times["tensorize"]
                window_stage_totals["resize"] += stage_times["resize"]
                window_stage_totals["det"] += stage_times["det"]
                window_stage_totals["color"] += stage_times["color"]
                window_stage_totals["score"] += stage_times["score"]
                _maybe_log_window(final=False)
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
    _maybe_log_window(final=True)

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
        "tensorize_sec": round(tensorize_sec, 4),
        "resize_sec": round(resize_sec, 4),
        "det_infer_sec": round(det_infer_sec, 4),
        "color_infer_sec": round(color_infer_sec, 4),
        "score_sec": round(score_sec, 4),
        "init_det_sec": round(init_det_sec, 4),
        "init_color_sec": round(init_color_sec, 4),
    }
    if det_adaptive_state:
        triggered_dims = [
            dim
            for dim, state in det_adaptive_state.items()
            if bool(state.get("triggered", False))
        ]
        stats["adaptive_triggered_dims"] = len(triggered_dims)
        stats["adaptive_probe_clips"] = det_adaptive_probe_clips
        stats["adaptive_score_threshold"] = round(det_adaptive_score_threshold, 6)
        for dim, state in det_adaptive_state.items():
            logger.info(
                "[rank %d] Slow-dims adaptive summary [%s]: probed=%d triggered=%s max_abs_diff=%.6f",
                rank,
                dim,
                int(state.get("probed", 0)),
                bool(state.get("triggered", False)),
                float(state.get("max_abs_diff", 0.0)),
            )

    logger.info(
        "[rank %d] slow-dims fused stats: clips %d/%d decode=%.2fs tensorize=%.2fs resize=%.2fs det=%.2fs color=%.2fs score=%.2fs decode_workers=%d prefetch=%d backend=%s shard_mode=%s",
        rank,
        stats["clips_shard"],
        stats["clips_total"],
        stats["decode_sec"],
        stats["tensorize_sec"],
        stats["resize_sec"],
        stats["det_infer_sec"],
        stats["color_infer_sec"],
        stats["score_sec"],
        decode_workers,
        decode_prefetch,
        active_decode_backend,
        shard_mode,
    )
    return rows, stats
