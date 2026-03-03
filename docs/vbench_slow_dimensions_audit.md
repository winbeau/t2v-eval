# VBench Slow Dimensions Audit

## Scope
This document audits four slow VBench dimensions in `third_party/VBench`:
- `object_class`
- `multiple_objects`
- `color`
- `spatial_relationship`

The goal is to describe exact data flow, scoring logic, parameter semantics, and safe parallelization boundaries without changing output semantics for identical inputs.

## Source Map
Core entry points:
- `third_party/VBench/vbench/__init__.py:171`
- `third_party/VBench/vbench/utils.py:202`
- `third_party/VBench/vbench/utils.py:108`
- `third_party/VBench/vbench/utils.py:236`

Slow dimensions:
- `third_party/VBench/vbench/object_class.py:45`
- `third_party/VBench/vbench/multiple_objects.py:48`
- `third_party/VBench/vbench/color.py:58`
- `third_party/VBench/vbench/spatial_relationship.py:117`

Shared GrIT stack:
- `third_party/VBench/vbench/third_party/grit_model.py:7`
- `third_party/VBench/vbench/third_party/grit_src/image_dense_captions.py:57`
- `third_party/VBench/vbench/third_party/grit_src/grit/predictor.py:97`
- `third_party/VBench/vbench/third_party/grit_src/grit/modeling/meta_arch/grit.py:23`
- `third_party/VBench/vbench/third_party/grit_src/grit/modeling/roi_heads/grit_roi_heads.py:236`

Long-mode wrappers:
- `third_party/VBench/vbench2_beta_long/object_class.py:6`
- `third_party/VBench/vbench2_beta_long/multiple_objects.py:5`
- `third_party/VBench/vbench2_beta_long/color.py:6`
- `third_party/VBench/vbench2_beta_long/spatial_relationship.py:6`
- `third_party/VBench/vbench2_beta_long/utils.py:196`

## End-to-End Data Flow
1. `VBench.evaluate(...)` builds `dimension_list`, initializes model assets via `init_submodules(...)`, builds `*_full_info.json`, then dispatches `compute_<dimension>(...)` for each dimension.
   - `third_party/VBench/vbench/__init__.py:171-188`
2. `compute_<dimension>` creates `DenseCaptioning`, initializes GRiT weights, loads prompt/video metadata via `load_dimension_info(...)`, then optionally shards by rank.
   - `third_party/VBench/vbench/object_class.py:73-79`
   - `third_party/VBench/vbench/multiple_objects.py:76-82`
   - `third_party/VBench/vbench/color.py:93-99`
   - `third_party/VBench/vbench/spatial_relationship.py:140-146`
3. Each dimension loops prompts -> videos. Video decoding is done by `load_video(...)` with `num_frames=16` (middle-uniform sampling).
   - `third_party/VBench/vbench/utils.py:108-177`
   - `third_party/VBench/vbench/utils.py:68-106`
4. Frames are passed to GrIT through `DenseCaptioning.run_caption_tensor_batch(...)`.
   - `third_party/VBench/vbench/third_party/grit_model.py:39-54`
5. GRiT returns Detectron2 instances; `dense_pred_to_caption_tuple(...)` converts to tuples consumed by each metric.
   - `third_party/VBench/vbench/third_party/grit_src/image_dense_captions.py:57-65`
6. Per-video and global scores are aggregated, then multi-rank results are gathered and recomputed.
   - `third_party/VBench/vbench/distributed.py:117-125`

## GRiT Output Contract Used by These Metrics
`dense_pred_to_caption_tuple(predictions)` produces per-detection tuples:
- element 0: dense caption text (`pred_object_descriptions[i]`)
- element 1: bounding box `[x0, y0, x1, y1]`
- element 2: detector object labels (`det_obj` from `results_det`)

Relevant lines:
- `third_party/VBench/vbench/third_party/grit_src/image_dense_captions.py:57-65`
- `third_party/VBench/vbench/third_party/grit_src/grit/modeling/meta_arch/grit.py:36-40`

Important runtime constraint:
- GRiT ROIHeads inference path asserts single-image handling: `assert len(boxes) == 1`.
- `third_party/VBench/vbench/third_party/grit_src/grit/modeling/roi_heads/grit_roi_heads.py:236`

## Dimension-by-Dimension Logic

### 1) object_class
Code:
- `third_party/VBench/vbench/object_class.py`

Inputs per prompt item:
- `info['auxiliary_info']['object']` as target class token.
- `info['video_list']` as video paths.

Processing:
1. Decode 16 frames: `load_video(video_path, num_frames=16)`.
2. If `min(h, w) > 768`, resize shortest side to 720 using torchvision resize.
3. Run GrIT on frames.
4. For each frame, convert detection to `set(caption[0][2])` (class set), count frame hit if `key_info in pred_set`.

Scoring:
- Per-video: `cur_success_frame_count / len(cur_video_pred)`.
- Global: `success_frame_count / frame_count`.
- Multi-rank recompute uses gathered `success_frame_count` and `frame_count`.

Notes:
- Frame weighting is explicit and deterministic.

### 2) multiple_objects
Code:
- `third_party/VBench/vbench/multiple_objects.py`

Inputs:
- `info['auxiliary_info']['object']` expected format: `"objA and objB"`.

Processing:
1. Same decode/resize path as `object_class`.
2. GrIT output converted to per-frame class sets.
3. Frame hit if both parsed tokens exist in same frame set.

Scoring:
- Per-video: hit frames / total frames.
- Global: sum hits / sum frames.
- Multi-rank: same count-based recomputation.

Notes:
- Parser assumes exact delimiter `' and '`, no robust fallback.

### 3) color
Code:
- `third_party/VBench/vbench/color.py`

Inputs:
- `color_info = info['auxiliary_info']['color']`.
- `object_info` derived from prompt string after removing `a/an` and the color token.

Processing:
1. Decode 16 frames with numpy output: `load_video(..., return_tensor=False)`.
2. If min side > 768, resize frame-by-frame using `cv2.resize`.
3. GrIT output per frame is converted to `[caption_text, object_type]` pairs.
4. For each frame:
   - object exists if `object_key == pred[1]` and caption contains any color word in hardcoded palette.
   - color hit if object exists and `color_key in pred[0]`.

Scoring:
- Per-video (only when object appears at least once): `cur_object_color / cur_object`.
- Global: mean over videos with `cur_object > 0`.
- Multi-rank: mean of gathered per-video `cur_success_frame_rate`.

Notes:
- Videos with zero object matches are dropped from denominator.
- If all videos are dropped, `success_rate = success_frame_count_all / video_count` risks division by zero.

### 4) spatial_relationship
Code:
- `third_party/VBench/vbench/spatial_relationship.py`

Inputs:
- `info['auxiliary_info']['spatial_relationship']` with:
  - `object_a`
  - `object_b`
  - `relationship`

Processing:
1. Same decode/resize path as `object_class`.
2. GrIT output converted to per-frame `[(object_desc, bbox), ...]`.
3. Frame-level scoring:
   - keep boxes for detections whose label matches object_a or object_b.
   - score every pair with `get_position_score(...)`.
   - frame score is max pair score.

`get_position_score(...)` details:
- computes centers, axis deltas, IoU.
- `iou_threshold` default `0.1`.
- returns in `[0,1]` depending on relation axis dominance and IoU penalty.

Scoring:
- Per-video: mean(frame_scores).
- Single-rank global: mean over all frame scores.
- Multi-rank global: mean of gathered per-video means.

Notes:
- Relation checks use `if locality in 'on the right of'` style substring test, not strict equality.

## Parameter Semantics

### Top-level VBench.evaluate params
From `third_party/VBench/vbench/__init__.py:171`:
- `videos_path`: root path to videos.
- `name`: output prefix, used for `<name>_full_info.json` and `<name>_eval_results.json`.
- `prompt_list`: optional prompt override mapping/list for custom mode.
- `dimension_list`: evaluated dimensions; default full 16.
- `local`: whether to use local cached model artifacts in `init_submodules`.
- `read_frame`: passed into `init_submodules`; not used by these four dimensions directly.
- `mode`: `vbench_standard`, `custom_input`, `vbench_category`, etc., controls metadata construction.

### Shared video sampling params
From `third_party/VBench/vbench/utils.py` and per-dimension calls:
- `num_frames=16`: fixed frame count sampled by `get_frame_indices(..., sample='middle')` for mp4.
- `return_tensor=True/False`: tensor path (`object_class`, `multiple_objects`, `spatial_relationship`) vs numpy path (`color`).
- resize guard `min(h,w)>768` then shortest side scaled to 720.

### GrIT model params
From `third_party/VBench/vbench/third_party/grit_src/image_dense_captions.py`:
- config file: `GRiT_B_DenseCap_ObjectDet.yaml`.
- `confidence_threshold=0.5` (`cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST`).
- `test_task`: `DenseCap` (color) or `ObjectDet` (object/spatial dims via `initialize_model_det`).
- `BEAM_SIZE=1`.
- `SOFT_NMS_ENABLED=False`.

### Submodule asset params for these dimensions
From `third_party/VBench/vbench/utils.py:338-344`:
- all four dimensions use the same weight:
  - `model_weight = ~/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth`
- downloaded from HF if missing.

### Distributed params
From `third_party/VBench/vbench/distributed.py`:
- work partitioning: `distribute_list_to_rank(data_list)` uses stride slicing `data_list[rank::world_size]`.
- result gather: `gather_list_of_dict` uses `all_gather` + flatten.

## Long-Mode Specific Data Flow (VBenchLong)
1. `VBenchLong.evaluate(...)` preprocesses videos into clips.
   - `third_party/VBench/vbench2_beta_long/__init__.py:82-93`
2. In `long_custom_input`, clip duration is forced to `2` seconds.
   - `third_party/VBench/vbench2_beta_long/__init__.py:64-67`
3. `compute_long_<dim>` delegates to base `compute_<dim>` on clip-level json.
4. `reorganize_clips_results(...)` maps clip paths back to long-video prompt path and averages clip scores.
   - `third_party/VBench/vbench2_beta_long/utils.py:196-227`

For the four slow dimensions, long mode reuses the same frame-level inference/scoring code; only pre-splitting and clip-to-long aggregation differ.

## Performance Hotspots
1. Video decode + frame sampling (decord / cv2 path), repeated per dimension.
2. GRiT inference over 16 frames per video.
3. Python-side post-processing and set/list conversions.

## Safe Parallelization Boundaries Without Output Change
Allowed if deterministic ordering and reductions are preserved:
1. Keep scoring formulas unchanged; parallelize only scheduling.
2. Parallelize at video-item granularity (`prompt_dict_ls` shards) with same per-item function.
3. Keep per-video frame order and per-rank reduction formulas unchanged.
4. Keep precision and thresholds unchanged (`confidence_threshold`, no mixed-precision changes).

Not safe for strict equivalence:
1. Changing score thresholds, NMS, beam size, prompt parsing, or relation conditions.
2. Switching to unsupported true multi-image ROIHeads path where model assumptions differ.
3. Changing aggregation denominator policy (for example color's video filtering behavior).

## Known Logic Risks (Current Upstream Behavior)
1. `color` can divide by zero when no video has detectable target object.
2. `spatial_relationship` relation check uses substring-membership style (`in` on literal string), not strict relation enum match.
3. `multiple_objects` assumes exact `' and '` delimiter.

These are behavior notes only; they are not proposed changes in this audit.
