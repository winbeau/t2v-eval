#!/usr/bin/env python3
"""
Profile VBench slow dimensions on one video.

This script measures stage-wise latency for:
  - object_class
  - multiple_objects
  - color
  - spatial_relationship

Stages per dimension:
  decode -> resize -> infer -> score

The scoring logic uses upstream VBench helpers directly, so timing reflects
real execution paths while keeping scoring behavior unchanged.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
VBENCH_ROOT = REPO_ROOT / "third_party" / "VBench"
if str(VBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(VBENCH_ROOT))

DenseCaptioning = None
init_submodules = None
load_video = None
color_mod = None
multi_mod = None
object_mod = None
spatial_mod = None


def _ensure_vbench_imports() -> None:
    global DenseCaptioning
    global init_submodules
    global load_video
    global color_mod
    global multi_mod
    global object_mod
    global spatial_mod
    if DenseCaptioning is not None:
        return
    from vbench.third_party.grit_model import DenseCaptioning as _DenseCaptioning
    from vbench.utils import init_submodules as _init_submodules, load_video as _load_video
    import vbench.color as _color_mod
    import vbench.multiple_objects as _multi_mod
    import vbench.object_class as _object_mod
    import vbench.spatial_relationship as _spatial_mod

    DenseCaptioning = _DenseCaptioning
    init_submodules = _init_submodules
    load_video = _load_video
    color_mod = _color_mod
    multi_mod = _multi_mod
    object_mod = _object_mod
    spatial_mod = _spatial_mod


def _sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_block(device: str, fn) -> tuple[Any, float]:
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    out = fn()
    _sync_if_cuda(device)
    return out, time.perf_counter() - t0


def _resize_tensor_if_needed(video_tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    _, _, h, w = video_tensor.size()
    if min(h, w) <= 768:
        return video_tensor, 0.0
    t0 = time.perf_counter()
    scale = 720.0 / min(h, w)
    resized = transforms.Resize(size=(int(scale * h), int(scale * w)))(video_tensor)
    return resized, time.perf_counter() - t0


def _resize_numpy_if_needed(video_arrays: np.ndarray) -> tuple[np.ndarray, float]:
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


def _profile_object_class(video_path: str, device: str, det_model: Any, key_info: str) -> dict:
    result: dict[str, float] = {}
    video_tensor, t_decode = _time_block(
        device, lambda: load_video(video_path, num_frames=16)
    )
    result["decode"] = t_decode

    video_tensor, t_resize = _resize_tensor_if_needed(video_tensor)
    result["resize"] = t_resize

    preds, t_infer = _time_block(
        device, lambda: object_mod.get_dect_from_grit(det_model, video_tensor.permute(0, 2, 3, 1))
    )
    result["infer"] = t_infer

    _, t_score = _time_block(device, lambda: object_mod.check_generate(key_info, preds))
    result["score"] = t_score
    result["total"] = sum(result.values())
    return result


def _profile_multiple_objects(
    video_path: str, device: str, det_model: Any, key_info: str
) -> dict:
    result: dict[str, float] = {}
    video_tensor, t_decode = _time_block(
        device, lambda: load_video(video_path, num_frames=16)
    )
    result["decode"] = t_decode

    video_tensor, t_resize = _resize_tensor_if_needed(video_tensor)
    result["resize"] = t_resize

    preds, t_infer = _time_block(
        device, lambda: multi_mod.get_dect_from_grit(det_model, video_tensor.permute(0, 2, 3, 1))
    )
    result["infer"] = t_infer

    _, t_score = _time_block(device, lambda: multi_mod.check_generate(key_info, preds))
    result["score"] = t_score
    result["total"] = sum(result.values())
    return result


def _profile_color(
    video_path: str,
    device: str,
    cap_model: Any,
    color_key: str,
    object_key: str,
) -> dict:
    result: dict[str, float] = {}
    video_arrays, t_decode = _time_block(
        device, lambda: load_video(video_path, num_frames=16, return_tensor=False)
    )
    result["decode"] = t_decode

    video_arrays, t_resize = _resize_numpy_if_needed(video_arrays)
    result["resize"] = t_resize

    preds, t_infer = _time_block(device, lambda: color_mod.get_dect_from_grit(cap_model, video_arrays))
    result["infer"] = t_infer

    _, t_score = _time_block(device, lambda: color_mod.check_generate(color_key, object_key, preds))
    result["score"] = t_score
    result["total"] = sum(result.values())
    return result


def _profile_spatial_relationship(
    video_path: str,
    device: str,
    det_model: Any,
    key_info: dict[str, str],
) -> dict:
    result: dict[str, float] = {}
    video_tensor, t_decode = _time_block(
        device, lambda: load_video(video_path, num_frames=16)
    )
    result["decode"] = t_decode

    video_tensor, t_resize = _resize_tensor_if_needed(video_tensor)
    result["resize"] = t_resize

    preds, t_infer = _time_block(
        device,
        lambda: spatial_mod.get_dect_from_grit(det_model, video_tensor.permute(0, 2, 3, 1)),
    )
    result["infer"] = t_infer

    _, t_score = _time_block(device, lambda: spatial_mod.check_generate(key_info, preds))
    result["score"] = t_score
    result["total"] = sum(result.values())
    return result


def _choose_video(video: str | None, video_dir: str | None) -> Path:
    if video:
        p = Path(video).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        return p
    if not video_dir:
        raise ValueError("Either --video or --video-dir must be provided.")
    d = Path(video_dir).expanduser().resolve()
    if not d.exists():
        raise FileNotFoundError(f"Directory not found: {d}")
    mp4s = sorted(d.glob("*.mp4"))
    if not mp4s:
        mp4s = sorted(d.rglob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 found under: {d}")
    return mp4s[0].resolve()


def _mean_times(entries: list[dict[str, float]]) -> dict[str, float]:
    keys = entries[0].keys()
    return {k: statistics.mean(e[k] for e in entries) for k in keys}


def _print_report(video_path: Path, init_det: float, init_cap: float, dim_means: dict[str, dict[str, float]]) -> None:
    print("\n=== VBench Slow Dimensions Profile (one video) ===")
    print(f"video: {video_path}")
    print(f"init_det_model: {init_det:.3f}s")
    print(f"init_densecap_model: {init_cap:.3f}s")
    print("\nPer-dimension stage latency (seconds):")
    print("dimension               decode   resize   infer    score    total    infer%")
    print("-" * 78)
    for dim_name in ["object_class", "multiple_objects", "color", "spatial_relationship"]:
        t = dim_means[dim_name]
        infer_pct = (100.0 * t["infer"] / t["total"]) if t["total"] > 0 else 0.0
        print(
            f"{dim_name:22s} {t['decode']:7.3f} {t['resize']:7.3f} {t['infer']:7.3f} "
            f"{t['score']:8.3f} {t['total']:8.3f} {infer_pct:7.2f}%"
        )

    agg = {"decode": 0.0, "resize": 0.0, "infer": 0.0, "score": 0.0, "total": 0.0}
    for v in dim_means.values():
        for k in agg:
            agg[k] += v[k]
    print("\nAggregate over 4 dims:")
    print(
        f"decode={agg['decode']:.3f}s resize={agg['resize']:.3f}s "
        f"infer={agg['infer']:.3f}s score={agg['score']:.3f}s total={agg['total']:.3f}s"
    )
    for k in ["decode", "resize", "infer", "score"]:
        pct = (100.0 * agg[k] / agg["total"]) if agg["total"] > 0 else 0.0
        print(f"  {k:6s}: {pct:6.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile four slow VBench dimensions on one video.")
    parser.add_argument("--video", type=str, default=None, help="Path to a single video file.")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory to auto-pick the first .mp4 (sorted).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (not reported).")
    parser.add_argument("--repeat", type=int, default=2, help="Measured repeats.")
    parser.add_argument(
        "--object-key",
        type=str,
        default="person",
        help="Key for object_class.check_generate",
    )
    parser.add_argument(
        "--multi-key",
        type=str,
        default="person and car",
        help="Key for multiple_objects.check_generate",
    )
    parser.add_argument("--color-key", type=str, default="red", help="Key for color.check_generate")
    parser.add_argument(
        "--color-object-key",
        type=str,
        default="car",
        help="object_key for color.check_generate",
    )
    parser.add_argument(
        "--spatial-object-a", type=str, default="person", help="object_a for spatial check_generate"
    )
    parser.add_argument(
        "--spatial-object-b", type=str, default="car", help="object_b for spatial check_generate"
    )
    parser.add_argument(
        "--spatial-relation",
        type=str,
        default="on the right of",
        help="relationship for spatial check_generate",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save raw timing JSON.",
    )
    args = parser.parse_args()
    _ensure_vbench_imports()

    video_path = _choose_video(args.video, args.video_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)

    dims = ["object_class", "multiple_objects", "color", "spatial_relationship"]
    submods = init_submodules(dims, local=True, read_frame=False)
    model_weight = submods["object_class"]["model_weight"]

    det_model = DenseCaptioning(device)
    _, init_det = _time_block(
        device, lambda: det_model.initialize_model_det(model_weight=model_weight)
    )

    cap_model = DenseCaptioning(device)
    _, init_cap = _time_block(
        device, lambda: cap_model.initialize_model(model_weight=model_weight)
    )

    spatial_key = {
        "object_a": args.spatial_object_a,
        "object_b": args.spatial_object_b,
        "relationship": args.spatial_relation,
    }

    def _run_once() -> dict[str, dict[str, float]]:
        return {
            "object_class": _profile_object_class(
                str(video_path), device, det_model, args.object_key
            ),
            "multiple_objects": _profile_multiple_objects(
                str(video_path), device, det_model, args.multi_key
            ),
            "color": _profile_color(
                str(video_path), device, cap_model, args.color_key, args.color_object_key
            ),
            "spatial_relationship": _profile_spatial_relationship(
                str(video_path), device, det_model, spatial_key
            ),
        }

    for _ in range(max(0, args.warmup)):
        _run_once()

    repeats = []
    for _ in range(max(1, args.repeat)):
        repeats.append(_run_once())

    dim_means: dict[str, dict[str, float]] = {}
    for dim_name in ["object_class", "multiple_objects", "color", "spatial_relationship"]:
        dim_means[dim_name] = _mean_times([r[dim_name] for r in repeats])

    _print_report(video_path, init_det, init_cap, dim_means)

    if args.output_json:
        out = {
            "video_path": str(video_path),
            "device": args.device,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "init_seconds": {
                "det_model": init_det,
                "densecap_model": init_cap,
            },
            "mean_seconds": dim_means,
            "raw_repeats": repeats,
        }
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out_path}")


if __name__ == "__main__":
    main()
