# VBench Slow Dimensions Equivalence Checklist

This checklist is for validating acceleration/parallelization changes while preserving identical outputs for the same inputs.

## Dimensions Covered
- `object_class`
- `multiple_objects`
- `color`
- `spatial_relationship`

## Golden Baseline Setup
1. Freeze environment and model artifacts.
   - same CUDA, torch, detectron2, decord versions
   - same `~/.cache/vbench/grit_model/grit_b_densecap_objectdet.pth`
2. Freeze input set.
   - identical `*_full_info.json`
   - identical video files and order
3. Disable stochasticity where possible.
   - fixed process count / rank mapping
   - no random frame sampling changes (must remain `sample='middle'`)

## Required Invariance Rules
1. Frame sampling invariance
   - same selected 16 frame indices per video.
2. Preprocessing invariance
   - same resize trigger and target size rules.
3. Detection invariance
   - same GRiT config, threshold, task mode, and precision behavior.
4. Scoring invariance
   - same per-frame predicate logic and same denominators.
5. Reduction invariance
   - same per-video and global aggregation formulas.

## Per-Dimension Assertions

### object_class
1. Per-video equality:
   - `success_frame_count`
   - `frame_count`
   - `video_results`
2. Global equality:
   - `sum(success_frame_count) / sum(frame_count)`

### multiple_objects
1. Per-video equality:
   - `success_frame_count`
   - `frame_count`
   - `video_results`
2. Global equality:
   - `sum(success_frame_count) / sum(frame_count)`

### color
1. Per-video equality (for included videos):
   - `cur_success_frame_rate`
2. Inclusion-set equality:
   - same set of videos with `cur_object > 0`
3. Global equality:
   - mean over included video rates

### spatial_relationship
1. Per-video equality:
   - `frame_results` list
   - `video_results` mean
2. Global equality:
   - same reduction policy as baseline mode

## Regression Execution Matrix
1. Single-process baseline vs candidate
2. Multi-rank baseline vs candidate
3. Long-mode clip input vs standard input (if used)

Each cell must compare:
- exact JSON outputs for per-video metrics
- final aggregate scalar per dimension

## Comparison Procedure
1. Run baseline and candidate with identical config/input.
2. Extract per-dimension outputs to normalized CSV/JSON.
3. Diff fields in this order:
   - per-frame fields (where present)
   - per-video fields
   - global scalar
4. Fail on first mismatch.

## Suggested Automation Hooks
1. Add a script that loads two result JSON files and checks:
   - strict equality for integers/lists/sets encoded as arrays
   - `abs(a-b) == 0.0` for floats in strict mode
2. Emit mismatch report with:
   - dimension
   - video_path
   - field name
   - baseline value
   - candidate value

## Acceptance Criteria
1. All four dimensions pass per-video and global checks.
2. No missing or extra videos in gathered outputs.
3. Runtime improves while outputs remain identical.

## Stop Conditions
Stop rollout immediately if any of the following changes unintentionally:
1. number of processed videos
2. frame_count denominators
3. inclusion policy in `color`
4. relation predicate behavior in `spatial_relationship`
