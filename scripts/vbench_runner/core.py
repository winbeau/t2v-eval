#!/usr/bin/env python3
"""
VBench evaluation orchestrator.

Wraps the official VBench repository (https://github.com/Vchitect/VBench)
to evaluate temporal quality of generated videos.

Usage:
    python scripts/run_vbench.py --config configs/eval.yaml
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

try:
    from .assets import ensure_clip_assets_for_subtasks, required_clip_asset_keys
    from .auxiliary import (
        build_auxiliary_prompt_lookup,
        patch_long_custom_full_info_builder,
    )
    from .compat import apply_vbench_compat_patches
    from .distributed import (
        _parse_visible_devices,
        init_distributed_if_needed,
        make_file_barrier,
        maybe_auto_launch_multi_gpu,
        merge_rank_partial_results,
        split_subtasks_for_rank,
    )
    from .env import (
        PROJECT_ROOT,
        VBENCH_ROOT,
        check_vbench_installation,
        configure_third_party_loggers,
        configure_warning_filters,
        ensure_clip_dependency,
        ensure_extended_dimension_dependencies,
        ensure_moviepy_editor_compat,
        ensure_pyav_dependency,
        ensure_pyiqa_dependency,
        get_vbench_subtasks,
        load_config,
        logger,
        resolve_path,
        setup_log_file_handler,
        setup_vbench_path,
        use_vbench_long,
    )
    from .preprocess_long import parallel_split_long_clips
    from .progress import (
        MultiGpuProgressBoard,
        RankProgressReporter,
        run_callable_with_progress,
        run_evaluate_with_progress,
        summarize_vbench_stdout,
    )
    from .results import extract_subtask_scores
    from .video_records import (
        are_split_clips_ready,
        copy_outputs_to_frontend,
        ensure_unique_video_ids,
        get_input_video_files,
        load_video_records_for_vbench,
    )
except ImportError:
    from vbench_runner.assets import ensure_clip_assets_for_subtasks, required_clip_asset_keys
    from vbench_runner.auxiliary import (
        build_auxiliary_prompt_lookup,
        patch_long_custom_full_info_builder,
    )
    from vbench_runner.compat import apply_vbench_compat_patches
    from vbench_runner.distributed import (
        _parse_visible_devices,
        init_distributed_if_needed,
        make_file_barrier,
        maybe_auto_launch_multi_gpu,
        merge_rank_partial_results,
        split_subtasks_for_rank,
    )
    from vbench_runner.env import (
        PROJECT_ROOT,
        VBENCH_ROOT,
        check_vbench_installation,
        configure_third_party_loggers,
        configure_warning_filters,
        ensure_clip_dependency,
        ensure_extended_dimension_dependencies,
        ensure_moviepy_editor_compat,
        ensure_pyav_dependency,
        ensure_pyiqa_dependency,
        get_vbench_subtasks,
        load_config,
        logger,
        resolve_path,
        setup_log_file_handler,
        setup_vbench_path,
        use_vbench_long,
    )
    from vbench_runner.preprocess_long import parallel_split_long_clips
    from vbench_runner.progress import (
        MultiGpuProgressBoard,
        RankProgressReporter,
        run_callable_with_progress,
        run_evaluate_with_progress,
        summarize_vbench_stdout,
    )
    from vbench_runner.results import extract_subtask_scores
    from vbench_runner.video_records import (
        are_split_clips_ready,
        copy_outputs_to_frontend,
        ensure_unique_video_ids,
        get_input_video_files,
        load_video_records_for_vbench,
    )


# ANSI color codes for terminal output
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _log_subtask_summary(subtask_status: dict[str, tuple[bool, str]], rank: int) -> None:
    """Print per-subtask success/fail summary with colors."""
    ok_count = sum(1 for ok, _ in subtask_status.values() if ok)
    fail_count = len(subtask_status) - ok_count
    logger.info(
        f"[rank {rank}] Subtask summary: "
        f"{_GREEN}{ok_count} passed{_RESET}, "
        f"{_RED}{fail_count} failed{_RESET}"
    )
    for name, (ok, detail) in subtask_status.items():
        if ok:
            logger.info(f"  {_GREEN}PASS{_RESET}  {name}")
        else:
            logger.error(f"  {_RED}FAIL{_RESET}  {name}: {detail}")


def _resolve_preprocess_workers(
    args: argparse.Namespace,
    vbench_config: dict,
) -> tuple[int, str]:
    """
    Resolve VBench preprocess worker count.

    Priority: CLI > config > default(1).
    """
    if args.preprocess_workers is not None:
        return int(args.preprocess_workers), "cli"

    cfg_value = vbench_config.get("preprocess_workers")
    if cfg_value is None:
        return 1, "default"

    try:
        workers = int(cfg_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid metrics.vbench.preprocess_workers=%r, fallback to 1",
            cfg_value,
        )
        return 1, "default"

    if workers < 1:
        logger.warning(
            "metrics.vbench.preprocess_workers=%r must be >= 1, fallback to 1",
            cfg_value,
        )
        return 1, "default"

    return workers, "config"


def _resolve_float_option(vbench_config: dict, key: str, default: float) -> float:
    """Parse float option from metrics.vbench with fallback and warning."""
    raw_value = vbench_config.get(key, default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid metrics.vbench.%s=%r, fallback to %s", key, raw_value, default)
        return default
    if value <= 0:
        logger.warning(
            "metrics.vbench.%s=%r must be > 0, fallback to %s",
            key,
            raw_value,
            default,
        )
        return default
    return value


def _wait_for_rank_partial_files(
    partial_dir: Path,
    world_size: int,
    timeout_sec: float,
    poll_sec: float,
) -> list[int]:
    """
    Wait for rank_{i}.csv files to appear.

    Returns a sorted list of missing ranks after timeout (empty means complete).
    """
    deadline = time.time() + timeout_sec

    def _missing_ranks() -> list[int]:
        return [
            worker_rank
            for worker_rank in range(world_size)
            if not (partial_dir / f"rank_{worker_rank}.csv").exists()
        ]

    missing = _missing_ranks()
    while missing and time.time() < deadline:
        time.sleep(max(poll_sec, 0.1))
        missing = _missing_ranks()
    return missing


# =============================================================================
# VBench evaluation (Python API)
# =============================================================================
def run_vbench_evaluation(
    video_records: list,
    output_dir: Path,
    config: dict,
    device: str = "cuda",
    preprocess_workers: int = 1,
    rank: int = 0,
    world_size: int = 1,
    barrier_fn: Callable[[], None] | None = None,
    subtasks_override: list[str] | None = None,
    progress_total_subtasks: int | None = None,
    progress_reporter: RankProgressReporter | None = None,
) -> pd.DataFrame:
    """
    Run VBench evaluation on videos using official implementation.

    Returns:
        DataFrame with per-video VBench scores
    """
    setup_vbench_path()
    if barrier_fn is None:

        def barrier_fn() -> None:
            return None

    long_mode = use_vbench_long(config)

    # VBench configuration
    vbench_config = config.get("metrics", {}).get("vbench", {})
    subtasks = (
        list(subtasks_override) if subtasks_override is not None else get_vbench_subtasks(config)
    )
    eval_mode = vbench_config.get("mode", "long_custom_input" if long_mode else "custom_input")
    total_subtasks = len(subtasks)
    display_total_subtasks = progress_total_subtasks or total_subtasks

    ensure_pyav_dependency(long_mode=long_mode)
    ensure_clip_dependency(subtasks=subtasks)
    ensure_pyiqa_dependency(subtasks=subtasks)
    ensure_extended_dimension_dependencies(subtasks=subtasks)

    try:
        if long_mode:
            ensure_moviepy_editor_compat()
            from vbench2_beta_long import VBenchLong as VBenchRunner
        else:
            from vbench import VBench as VBenchRunner
    except ImportError as e:
        logger.error(f"Failed to import VBench backend: {e}")
        logger.error("Please ensure VBench dependencies are installed:")
        logger.error("  pip install -r third_party/VBench/requirements.txt")
        raise

    apply_vbench_compat_patches()

    long_kwargs = {
        "use_semantic_splitting": bool(vbench_config.get("use_semantic_splitting", False)),
        "clip_length_config": vbench_config.get("clip_length_config", "clip_length_mix.yaml"),
        "threshold": float(vbench_config.get("scene_threshold", 35.0)),
        "static_filter_flag": bool(vbench_config.get("static_filter_flag", False)),
        "sb_clip2clip_feat_extractor": vbench_config.get("sb_clip2clip_feat_extractor", "dino"),
        "bg_clip2clip_feat_extractor": vbench_config.get("bg_clip2clip_feat_extractor", "clip"),
        "w_inclip": float(vbench_config.get("w_inclip", 1.0)),
        "w_clip2clip": float(vbench_config.get("w_clip2clip", 0.0)),
        "imaging_quality_preprocessing_mode": vbench_config.get(
            "imaging_quality_preprocessing_mode", "longer"
        ),
        "dev_flag": bool(vbench_config.get("dev_flag", False)),
        "num_of_samples_per_prompt": int(vbench_config.get("num_of_samples_per_prompt", 5)),
        "slow_fast_eval_config": str(
            vbench_config.get(
                "slow_fast_eval_config",
                VBENCH_ROOT / "vbench2_beta_long" / "configs" / "slow_fast_params.yaml",
            )
        ),
        "sb_mapping_file_path": str(
            vbench_config.get(
                "sb_mapping_file_path",
                VBENCH_ROOT / "vbench2_beta_long" / "configs" / "subject_mapping_table.yaml",
            )
        ),
        "bg_mapping_file_path": str(
            vbench_config.get(
                "bg_mapping_file_path",
                VBENCH_ROOT / "vbench2_beta_long" / "configs" / "background_mapping_table.yaml",
            )
        ),
    }

    # Results directory for VBench output (must be absolute path)
    results_dir = (output_dir / "vbench_results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # VBench requires full_info_dir to be a JSON file path (not directory)
    if long_mode:
        vbench_full_info = VBENCH_ROOT / "vbench2_beta_long" / "VBench_full_info.json"
    else:
        vbench_full_info = VBENCH_ROOT / "vbench" / "VBench_full_info.json"

    if not vbench_full_info.exists():
        logger.warning(f"VBench full info not found: {vbench_full_info}")
        logger.warning("Will use custom_input mode without full_info")
        vbench_full_info_str = str(results_dir / "dummy_info.json")
        with open(vbench_full_info_str, "w") as f:
            f.write("[]")
    else:
        vbench_full_info_str = str(vbench_full_info.resolve())
    vbench_full_info_path = Path(vbench_full_info_str)

    # Prepare video paths for VBench
    video_map = {r["video_id"]: r["video_path"] for r in video_records}
    valid_video_ids = set(video_map.keys())

    # Create a directory with symlinks to all videos for VBench
    video_dir = results_dir / "input_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for video_id, vp in video_map.items():
        src = Path(vp)
        suffix = src.suffix if src.suffix else ".mp4"
        dst = video_dir / f"{video_id}{suffix}"
        if not dst.exists() and src.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                import shutil

                shutil.copy2(src, dst)
    video_dir_str = str(video_dir.resolve())

    results = []

    # Initialize VBench
    try:
        vbench = VBenchRunner(
            device=device,
            full_info_dir=vbench_full_info_str,
            output_path=str(results_dir),
        )
    except Exception as e:
        logger.error(f"Failed to initialize VBench: {e}")
        logger.error("This may be due to missing model weights.")
        logger.error("Please check VBench documentation for weight download instructions.")
        raise

    if long_mode and str(eval_mode).strip().lower() == "long_custom_input":
        exact_lookup, simple_lookup = build_auxiliary_prompt_lookup(vbench_full_info_path)
        patch_long_custom_full_info_builder(
            vbench_runner=vbench,
            video_records=video_records,
            exact_lookup=exact_lookup,
            simple_lookup=simple_lookup,
        )

    if long_mode:
        input_videos = get_input_video_files(video_dir)
        split_ready = are_split_clips_ready(video_dir, input_videos)
        semantic_splitting = bool(long_kwargs.get("use_semantic_splitting", False))
        eval_mode_name = str(eval_mode).strip().lower()
        use_parallel_preprocess = (
            preprocess_workers > 1
            and eval_mode_name == "long_custom_input"
            and not semantic_splitting
        )

        if split_ready:
            if rank == 0:
                logger.info(
                    "Detected existing split clips for all videos; "
                    "skip preprocessing and reuse cache."
                )
        else:
            if rank == 0:
                if use_parallel_preprocess:
                    logger.info(
                        "Preprocessing long clips once before subtasks "
                        "(parallel, workers=%d)...",
                        preprocess_workers,
                    )
                    if progress_reporter is not None:
                        progress_reporter.start_task("preprocess_clips")
                    try:
                        summary = parallel_split_long_clips(
                            video_dir=video_dir,
                            input_videos=input_videos,
                            duration=2,
                            fps=8,
                            workers=preprocess_workers,
                            show_progress=(world_size <= 1 and progress_reporter is None),
                            progress_callback=(
                                (
                                    lambda payload: progress_reporter.update_live(
                                        percent=payload.get("percent"),
                                        status_text=str(payload.get("status_text", "running")),
                                        elapsed_sec=int(payload.get("elapsed_sec", 0)),
                                    )
                                )
                                if progress_reporter is not None
                                else None
                            ),
                        )
                        if summary["failed"] > 0:
                            raise RuntimeError(
                                "Parallel preprocess failed for "
                                f"{summary['failed']} videos"
                            )
                        if progress_reporter is not None:
                            progress_reporter.finish_task(success=True, count_completion=False)
                    except Exception as exc:
                        if progress_reporter is not None:
                            progress_reporter.finish_task(
                                success=False,
                                error=str(exc),
                                count_completion=False,
                            )
                        raise
                    logger.info(
                        "[preprocess] Parallel split done: videos=%d, clips=%d, skipped=%d, "
                        "elapsed=%ds",
                        summary["total_videos"],
                        summary["clips"],
                        summary["skipped"],
                        summary["elapsed_sec"],
                    )
                else:
                    if preprocess_workers > 1:
                        if semantic_splitting:
                            logger.info(
                                "preprocess_workers=%d requested but "
                                "use_semantic_splitting=true; fallback to native preprocess.",
                                preprocess_workers,
                            )
                        elif eval_mode_name != "long_custom_input":
                            logger.info(
                                "preprocess_workers=%d requested but mode=%s; "
                                "fallback to native preprocess.",
                                preprocess_workers,
                                eval_mode,
                            )
                    logger.info("Preprocessing videos into long clips once before subtasks...")
                    preprocess_kwargs = {
                        "videos_path": video_dir_str,
                        "mode": eval_mode,
                        **long_kwargs,
                    }
                    if progress_reporter is not None:
                        progress_reporter.start_task("preprocess_clips")
                    try:
                        preprocess_stdout, saved_count = run_callable_with_progress(
                            task_fn=lambda: vbench.preprocess(**preprocess_kwargs),
                            title="preprocess_clips",
                            prefix=f"[0/{display_total_subtasks}] ",
                            refresh_sec=1.0,
                            status_mode="clips",
                            enable_live=(world_size <= 1),
                            expected_units=len(video_records),
                            progress_callback=(
                                (
                                    lambda payload: progress_reporter.update_live(
                                        percent=payload.get("percent"),
                                        status_text=str(payload.get("status_text", "running")),
                                        elapsed_sec=int(payload.get("elapsed_sec", 0)),
                                    )
                                )
                                if progress_reporter is not None
                                else None
                            ),
                        )
                        if progress_reporter is not None:
                            progress_reporter.finish_task(success=True, count_completion=False)
                    except Exception as exc:
                        if progress_reporter is not None:
                            progress_reporter.finish_task(
                                success=False,
                                error=str(exc),
                                count_completion=False,
                            )
                        raise
                    if saved_count > 0:
                        logger.info(f"[preprocess] Saved split clips: {saved_count}")
                    summarize_vbench_stdout(preprocess_stdout, "preprocess")
            else:
                if progress_reporter is not None:
                    progress_reporter.start_task(
                        "preprocess_clips",
                        status_text="waiting_rank0_preprocess",
                    )
            if world_size > 1:
                barrier_fn()
                if rank != 0 and progress_reporter is not None:
                    progress_reporter.finish_task(success=True, count_completion=False)

        # VBench-Long evaluate() always calls preprocess(). We already did one-time
        # preprocessing above, so disable repeated preprocessing for each subtask.
        def _skip_preprocess(*args, **kwargs):
            return None

        vbench.preprocess = _skip_preprocess

    # Run evaluation for each subtask
    backend_name = "VBench-Long" if long_mode else "VBench"
    if rank == 0:
        logger.info(f"Running {backend_name} evaluation with subtasks: {subtasks}")

    subtask_status: dict[str, tuple[bool, str]] = {}  # subtask -> (success, detail)

    for idx, subtask in enumerate(subtasks, start=1):
        if rank == 0:
            logger.info(f"Evaluating subtask: {subtask}")
        try:
            if progress_reporter is not None:
                progress_reporter.start_task(subtask)
                progress_reporter.log_event(
                    f"Starting {subtask} ({idx}/{display_total_subtasks})"
                )
            eval_kwargs = {
                "videos_path": video_dir_str,
                "name": subtask,
                "dimension_list": [subtask],
                "local": True,
                "read_frame": bool(vbench_config.get("read_frame", False)),
                "mode": eval_mode,
            }
            if long_mode:
                eval_kwargs.update(long_kwargs)
            stdout_text, saved_count = run_evaluate_with_progress(
                vbench=vbench,
                eval_kwargs=eval_kwargs,
                subtask=subtask,
                subtask_index=idx,
                subtask_total=display_total_subtasks,
                refresh_sec=1.0,
                enable_live=(rank == 0 and world_size <= 1),
                progress_callback=(
                    (
                        lambda payload: progress_reporter.update_live(
                            percent=payload.get("percent"),
                            status_text=str(payload.get("status_text", "running")),
                            elapsed_sec=int(payload.get("elapsed_sec", 0)),
                        )
                    )
                    if progress_reporter is not None
                    else None
                ),
            )
            if rank == 0 and saved_count > 0:
                logger.info(f"[{subtask}] Saved split clips: {saved_count}")
            if rank == 0:
                summarize_vbench_stdout(stdout_text, subtask)

            # VBench saves results to {output_path}/{name}_eval_results.json
            result_file = results_dir / f"{subtask}_eval_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    subtask_data = json.load(f)
                if subtask in subtask_data:
                    count_before = len(results)
                    dimension_data = subtask_data[subtask]
                    extracted = extract_subtask_scores(
                        dimension_data=dimension_data,
                        subtask=subtask,
                        valid_video_ids=valid_video_ids,
                        long_mode=long_mode,
                    )
                    results.extend(extracted)

                    if rank == 0:
                        logger.info(f"Parsed {len(results) - count_before} results for {subtask}")
                    if progress_reporter is not None:
                        progress_reporter.log_event(
                            f"Parsed {len(results) - count_before} results for {subtask}"
                        )
            else:
                logger.warning(f"[rank {rank}] Result file not found: {result_file}")
                if progress_reporter is not None:
                    progress_reporter.log_event(
                        f"Result file not found for {subtask}", level="WARN"
                    )
            if progress_reporter is not None:
                progress_reporter.finish_task(success=True)
            subtask_status[subtask] = (True, "OK")
        except Exception as e:
            if progress_reporter is not None:
                progress_reporter.finish_task(success=False, error=str(e))
            subtask_status[subtask] = (False, str(e))
            logger.warning(f"[rank {rank}] Failed to run subtask {subtask}: {e}")
            logger.warning("Subtask traceback:", exc_info=True)
            if progress_reporter is not None:
                progress_reporter.log_event(
                    f"{subtask} failed: {e}", level="ERROR"
                )
            continue

    # Print per-subtask result summary (colored)
    if subtask_status:
        _log_subtask_summary(subtask_status, rank)

    # Convert to DataFrame and pivot
    if results:
        df_results = pd.DataFrame(results)
        df_pivot = df_results.pivot(
            index="video_id", columns="subtask", values="score"
        ).reset_index()

        # Compute aggregate temporal score (mean of all subtasks)
        subtask_cols = [c for c in df_pivot.columns if c != "video_id"]
        if subtask_cols:
            df_pivot["vbench_temporal_score"] = df_pivot[subtask_cols].mean(axis=1)

        return df_pivot
    else:
        logger.warning(f"[rank {rank}] No VBench results obtained")
        return pd.DataFrame()


# =============================================================================
# CLI fallback
# =============================================================================
def run_vbench_cli_fallback(
    video_records: list,
    output_dir: Path,
    config: dict,
) -> pd.DataFrame:
    """
    Fallback: Run VBench via CLI if Python API fails.

    This uses VBench's command-line interface which is more stable.
    """
    import subprocess

    vbench_config = config.get("metrics", {}).get("vbench", {})
    subtasks = vbench_config.get(
        "subtasks",
        [
            "temporal_flickering",
            "motion_smoothness",
        ],
    )

    # Create video directory structure expected by VBench CLI
    videos_dir = output_dir / "vbench_input"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Symlink videos to expected location
    for record in video_records:
        src = Path(record["video_path"])
        dst = videos_dir / src.name
        if not dst.exists() and src.exists():
            dst.symlink_to(src.resolve())

    results_dir = output_dir / "vbench_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run VBench CLI
    vbench_script = VBENCH_ROOT / "evaluate.py"

    for subtask in subtasks:
        cmd = [
            sys.executable,
            str(vbench_script),
            "--videos_path",
            str(videos_dir.resolve()),
            "--dimension",
            subtask,
            "--output_path",
            str(results_dir.resolve()),
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode != 0:
                logger.warning(f"VBench CLI failed for {subtask}: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"VBench CLI timeout for {subtask}")
        except Exception as e:
            logger.warning(f"VBench CLI error for {subtask}: {e}")

    # Parse results from output directory
    results = []
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            video_name = item.get("video_name", item.get("video_path", ""))
                            dimension = item.get("dimension", item.get("name", ""))
                            score = item.get("score", item.get("value", None))
                            if video_name and score is not None:
                                results.append(
                                    {
                                        "video_id": Path(video_name).stem,
                                        "subtask": dimension,
                                        "score": score,
                                    }
                                )
                elif isinstance(data, dict):
                    for video_name, scores in data.items():
                        if isinstance(scores, dict):
                            for dim, score in scores.items():
                                results.append(
                                    {
                                        "video_id": Path(video_name).stem,
                                        "subtask": dim,
                                        "score": score,
                                    }
                                )
                        elif isinstance(scores, (int, float)):
                            dim = result_file.stem
                            results.append(
                                {
                                    "video_id": Path(video_name).stem,
                                    "subtask": dim,
                                    "score": scores,
                                }
                            )
        except Exception as e:
            logger.warning(f"Failed to parse {result_file}: {e}")

    if results:
        df_results = pd.DataFrame(results)
        df_pivot = df_results.pivot(
            index="video_id", columns="subtask", values="score"
        ).reset_index()

        subtask_cols = [c for c in df_pivot.columns if c != "video_id"]
        if subtask_cols:
            df_pivot["vbench_temporal_score"] = df_pivot[subtask_cols].mean(axis=1)

        return df_pivot

    return pd.DataFrame()


def _cleanup_intermediates(output_dir: Path) -> None:
    """Remove intermediate files after a successful VBench run."""
    for dirname in ("vbench_partials", "vbench_sync", "vbench_progress"):
        d = output_dir / dirname
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            logger.info(f"Cleaned up: {d}")
    workers_log = output_dir / "vbench_workers.log"
    if workers_log.exists():
        workers_log.unlink(missing_ok=True)
        logger.info(f"Cleaned up: {workers_log}")


# =============================================================================
# CLI entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run VBench evaluation using official implementation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Skip VBench if errors occur (don't fail pipeline)",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated list of dimensions to skip (e.g. --skip color,object_class)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--no-auto-multi-gpu",
        action="store_true",
        help="Disable auto-launch to multi-GPU torchrun",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=None,
        help=(
            "Worker processes for one-time VBench-Long clip preprocessing "
            "(CLI overrides config)"
        ),
    )
    parser.add_argument(
        "--no-prefetch-assets",
        action="store_true",
        help="Disable preflight download/validation for shared CLIP checkpoints",
    )
    parser.add_argument(
        "--no-verify-asset-sha256",
        action="store_true",
        help="Disable SHA256 validation for prefetched CLIP checkpoints",
    )
    parser.add_argument(
        "--no-repair-corrupted-assets",
        action="store_true",
        help="Disable auto-repair when prefetched CLIP checkpoints are corrupted",
    )
    args = parser.parse_args()
    if args.preprocess_workers is not None and args.preprocess_workers < 1:
        parser.error("--preprocess-workers must be >= 1")
    configure_warning_filters()
    configure_third_party_loggers()

    # Check VBench installation
    if not check_vbench_installation():
        if args.skip_on_error:
            logger.warning("Skipping VBench evaluation due to missing submodule")
            sys.exit(0)
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    paths_config = config["paths"]
    runtime_config = config.get("runtime", {})
    rank, world_size, barrier_fn = init_distributed_if_needed()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    configured_log = os.environ.get("VBENCH_LOG_FILE") or paths_config.get("log_file")
    log_file_path: Path | None = None
    if configured_log:
        log_file_path = resolve_path(str(configured_log))
    else:
        default_output = (
            resolve_path(str(paths_config.get("output_dir", "./outputs")))
            or PROJECT_ROOT / "outputs"
        )
        log_file_path = default_output / "run_vbench.log"

    actual_log_path = setup_log_file_handler(
        log_path=log_file_path,
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        if world_size > 1:
            logger.info(
                "VBench logging enabled: rank0 -> %s (other ranks -> .rankN suffix)",
                actual_log_path,
            )
        else:
            logger.info("VBench logging enabled: %s", actual_log_path)

    if rank == 0 and world_size > 1:
        logger.info(
            "Multi-process dimension-parallel mode enabled: world_size=%d "
            "(no torch.distributed init)",
            world_size,
        )

    output_dir = Path(paths_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        sync_run_id = os.environ.get("TORCHELASTIC_RUN_ID") or str(int(time.time()))
        barrier_fn = make_file_barrier(
            sync_dir=output_dir / "vbench_sync" / sync_run_id,
            rank=rank,
            world_size=world_size,
        )

    config_stem = Path(args.config).stem
    vbench_output = output_dir / f"vbench_{config_stem}.csv"
    vbench_config = config.get("metrics", {}).get("vbench", {})
    preprocess_workers, workers_source = _resolve_preprocess_workers(args, vbench_config)

    # Check if already exists
    if rank == 0:
        logger.info(f"VBench output target: {vbench_output}")
        logger.info(
            "VBench long preprocess workers: %d (source=%s)",
            preprocess_workers,
            workers_source,
        )
    if vbench_output.exists() and not args.force:
        if rank == 0:
            logger.info(f"VBench results already exist: {vbench_output}")
            logger.info("Use --force to recompute")
        return
    elif vbench_output.exists() and args.force:
        if rank == 0:
            logger.info(f"Force recomputing: {vbench_output}")

    if maybe_auto_launch_multi_gpu(
        args=args,
        config=config,
        rank=rank,
        world_size=world_size,
        output_dir=output_dir,
    ):
        return

    # Load video metadata / build from local dataset
    try:
        video_records = load_video_records_for_vbench(
            config=config,
            output_dir=output_dir,
            paths_config=paths_config,
        )
        video_records = ensure_unique_video_ids(video_records, config)
    except Exception as e:
        logger.error(f"Failed to prepare video records: {e}")
        logger.error(
            "Need one of: processed_metadata.csv / metadata.csv / "
            "dataset.local_video_dir(with optional prompt_file/prompt_files_by_group)"
        )
        if args.skip_on_error:
            sys.exit(0)
        sys.exit(1)

    if rank == 0:
        logger.info(f"Loaded {len(video_records)} videos for VBench evaluation")

    all_subtasks = get_vbench_subtasks(config)

    # Filter out skipped dimensions
    skip_dims = set()
    skip_arg = getattr(args, "skip", "") or os.environ.get("VBENCH_SKIP_DIMS", "")
    if skip_arg:
        skip_dims = {d.strip() for d in skip_arg.split(",") if d.strip()}
    if skip_dims:
        skipped = sorted(skip_dims & set(all_subtasks))
        if skipped and rank == 0:
            logger.info("Skipping dimensions (--skip): %s", skipped)
        all_subtasks = [s for s in all_subtasks if s not in skip_dims]

    prefetch_assets = bool(vbench_config.get("prefetch_assets", True))
    verify_asset_sha256 = bool(vbench_config.get("verify_asset_sha256", True))
    repair_corrupted_assets = bool(vbench_config.get("repair_corrupted_assets", True))
    if args.no_prefetch_assets:
        prefetch_assets = False
    if args.no_verify_asset_sha256:
        verify_asset_sha256 = False
    if args.no_repair_corrupted_assets:
        repair_corrupted_assets = False

    asset_lock_timeout_sec = _resolve_float_option(
        vbench_config=vbench_config,
        key="asset_lock_timeout_sec",
        default=1800.0,
    )
    asset_download_timeout_sec = _resolve_float_option(
        vbench_config=vbench_config,
        key="asset_download_timeout_sec",
        default=600.0,
    )
    clip_asset_keys = required_clip_asset_keys(all_subtasks)
    if rank == 0:
        logger.info(
            "VBench asset prefetch: enabled=%s verify_sha256=%s repair_corrupted=%s required=%s",
            prefetch_assets,
            verify_asset_sha256,
            repair_corrupted_assets,
            clip_asset_keys,
        )

    prefetch_error: Exception | None = None
    if prefetch_assets and clip_asset_keys:
        if rank == 0:
            try:
                prefetch_summary = ensure_clip_assets_for_subtasks(
                    subtasks=all_subtasks,
                    verify_sha256=verify_asset_sha256,
                    repair_corrupted=repair_corrupted_assets,
                    lock_timeout_sec=asset_lock_timeout_sec,
                    download_timeout_sec=asset_download_timeout_sec,
                )
                logger.info(
                    "VBench asset prefetch summary: required=%d reused=%d downloaded=%d repaired=%d",
                    prefetch_summary["required"],
                    prefetch_summary["reused"],
                    prefetch_summary["downloaded"],
                    prefetch_summary["repaired"],
                )
            except Exception as exc:
                prefetch_error = exc
                logger.error("VBench asset prefetch failed: %s", exc)
        if world_size > 1:
            barrier_fn()
        if prefetch_error is not None:
            if args.skip_on_error:
                logger.warning(
                    "Continue without strict asset prefetch because --skip-on-error is set."
                )
            else:
                raise prefetch_error

    assigned_subtasks = split_subtasks_for_rank(all_subtasks, rank=rank, world_size=world_size)
    visible_devices = _parse_visible_devices()
    if rank == 0:
        logger.info(
            "Subtask distribution: total=%d, world_size=%d",
            len(all_subtasks),
            world_size,
        )
        if world_size > 1:
            for worker_rank in range(world_size):
                worker_subtasks = split_subtasks_for_rank(
                    all_subtasks, rank=worker_rank, world_size=world_size
                )
                logger.info("  rank %d -> %s", worker_rank, worker_subtasks)
    if not assigned_subtasks:
        logger.warning("Rank %d has no assigned subtasks; producing empty partial.", rank)

    progress_reporter: RankProgressReporter | None = None
    progress_board: MultiGpuProgressBoard | None = None
    parent_managed = os.environ.get("VBENCH_PROGRESS_PARENT_MANAGED") == "1"
    if world_size > 1:
        progress_dir = output_dir / "vbench_progress"
        if rank == 0:
            progress_dir.mkdir(parents=True, exist_ok=True)
            for stale_file in progress_dir.glob("rank_*.json"):
                try:
                    stale_file.unlink()
                except OSError:
                    pass
            # Clear events buffer for fresh run
            (progress_dir / "events.log").write_text("", encoding="utf-8")
        barrier_fn()
        progress_reporter = RankProgressReporter(
            progress_dir=progress_dir,
            rank=rank,
            local_rank=local_rank,
            assigned_subtasks=assigned_subtasks,
            visible_devices=visible_devices,
        )
        if rank == 0 and not parent_managed:
            assignment_map = {
                worker_rank: split_subtasks_for_rank(
                    all_subtasks, rank=worker_rank, world_size=world_size
                )
                for worker_rank in range(world_size)
            }
            gpu_map = {
                worker_rank: (
                    visible_devices[worker_rank]
                    if worker_rank < len(visible_devices)
                    else str(worker_rank)
                )
                for worker_rank in range(world_size)
            }
            progress_board = MultiGpuProgressBoard(
                progress_dir=progress_dir,
                assignment_map=assignment_map,
                gpu_map=gpu_map,
                refresh_sec=1.0,
            )
            progress_board.start()

    try:
        device = runtime_config.get("device", "cuda")
        if world_size > 1 and isinstance(device, str) and device.startswith("cuda"):
            device = f"cuda:{local_rank}"

        # Try Python API first, fall back to CLI
        long_mode = use_vbench_long(config)

        try:
            if rank == 0:
                logger.info("Attempting VBench evaluation via Python API...")
            df_results = run_vbench_evaluation(
                video_records=video_records,
                output_dir=output_dir,
                config=config,
                device=device,
                preprocess_workers=preprocess_workers,
                rank=rank,
                world_size=world_size,
                barrier_fn=barrier_fn,
                subtasks_override=assigned_subtasks,
                progress_total_subtasks=len(all_subtasks),
                progress_reporter=progress_reporter,
            )
        except Exception as e:
            if rank == 0:
                logger.warning(f"Python API failed: {e}")
            if long_mode:
                if rank == 0:
                    logger.error(
                        "VBench-Long currently supports Python API path only in this pipeline."
                    )
                if args.skip_on_error:
                    if rank == 0:
                        logger.warning("Skipping VBench evaluation")
                        pd.DataFrame(columns=["video_id", "vbench_temporal_score"]).to_csv(
                            vbench_output, index=False
                        )
                    sys.exit(0)
                raise

            if rank == 0:
                logger.info("Falling back to CLI-based evaluation...")
            try:
                if rank == 0:
                    df_results = run_vbench_cli_fallback(
                        video_records=video_records,
                        output_dir=output_dir,
                        config=config,
                    )
                else:
                    df_results = pd.DataFrame()
            except Exception as e2:
                if rank == 0:
                    logger.error(f"CLI fallback also failed: {e2}")
                if args.skip_on_error:
                    if rank == 0:
                        logger.warning("Skipping VBench evaluation")
                        pd.DataFrame(columns=["video_id", "vbench_temporal_score"]).to_csv(
                            vbench_output, index=False
                        )
                    sys.exit(0)
                raise

        if world_size > 1:
            partial_dir = output_dir / "vbench_partials"
            partial_dir.mkdir(parents=True, exist_ok=True)
            partial_file = partial_dir / f"rank_{rank}.csv"
            df_results.to_csv(partial_file, index=False)
            logger.info("Rank %d wrote partial results: %s", rank, partial_file)
            if rank != 0:
                logger.info(f"Rank {rank} finished distributed VBench worker.")
                return

            partial_collect_timeout_sec = _resolve_float_option(
                vbench_config=vbench_config,
                key="partial_collect_timeout_sec",
                default=43200.0,
            )
            partial_collect_poll_sec = _resolve_float_option(
                vbench_config=vbench_config,
                key="partial_collect_poll_sec",
                default=2.0,
            )
            missing_ranks = _wait_for_rank_partial_files(
                partial_dir=partial_dir,
                world_size=world_size,
                timeout_sec=partial_collect_timeout_sec,
                poll_sec=partial_collect_poll_sec,
            )
            if missing_ranks:
                logger.warning(
                    "Timed out waiting for partial results from ranks=%s "
                    "(timeout=%.1fs, poll=%.1fs). Will merge available partials.",
                    missing_ranks,
                    partial_collect_timeout_sec,
                    partial_collect_poll_sec,
                )

            partial_frames: list[pd.DataFrame] = []
            for worker_rank in range(world_size):
                rank_file = partial_dir / f"rank_{worker_rank}.csv"
                if rank_file.exists():
                    try:
                        partial_frames.append(pd.read_csv(rank_file))
                    except Exception as exc:
                        logger.warning("Failed to read %s: %s", rank_file, exc)
                else:
                    logger.warning(
                        "Missing partial results for rank %d: %s", worker_rank, rank_file
                    )
            df_results = merge_rank_partial_results(partial_frames)
            logger.info("Merged %d partial result files on CPU.", len(partial_frames))

        # Merge with metadata to get group info
        df_meta = pd.DataFrame(video_records)[["video_id", "group"]].drop_duplicates(
            subset=["video_id"]
        )
        if not df_results.empty:
            df_results = df_results.merge(df_meta, on="video_id", how="left")
        expected_count = len(video_records)
        coverage_rows: list[tuple[str, int, int]] = []
        for subtask in all_subtasks:
            if subtask not in df_results.columns:
                covered = 0
                logger.warning("Missing subtask column in merged output: %s", subtask)
            else:
                covered = int(df_results[subtask].notna().sum())
            coverage_rows.append((subtask, covered, expected_count))
            if covered != expected_count:
                logger.warning(
                    "Subtask coverage mismatch for %s: %d/%d",
                    subtask,
                    covered,
                    expected_count,
                )

        missing_coverage = [
            (name, covered, total) for name, covered, total in coverage_rows if covered != total
        ]

        # Save results
        df_results.to_csv(vbench_output, index=False)
        logger.info(f"VBench results saved to: {vbench_output}")

        # Print summary
        if "vbench_temporal_score" in df_results.columns:
            logger.info("\nVBench Temporal Score Summary by Group:")
            summary = df_results.groupby("group")["vbench_temporal_score"].agg(["mean", "std"])
            logger.info(f"\n{summary}")

        # Copy outputs to frontend/public/data
        logger.info("\nCopying VBench outputs to frontend/public/data ...")
        copy_outputs_to_frontend(
            output_dir=output_dir,
            paths_config=paths_config,
            vbench_output=vbench_output,
        )

        # Clean up intermediate files (partials, sync, progress, worker logs)
        _cleanup_intermediates(output_dir)

        if coverage_rows:
            name_width = max(len("subtask"), max(len(name) for name, _, _ in coverage_rows))
            logger.info("\nVBench Coverage Summary:")
            logger.info(f"  {'subtask':<{name_width}} | coverage | status")
            logger.info(f"  {'-' * name_width}-+----------+-------")
            for name, covered, total in coverage_rows:
                if covered == total:
                    status = f"{_GREEN}OK{_RESET}"
                elif covered == 0:
                    status = f"{_RED}FAIL{_RESET}"
                else:
                    status = f"{_YELLOW}PARTIAL{_RESET}"
                logger.info(f"  {name:<{name_width}} | {covered:>4d}/{total:<4d} | {status}")

        profile = str(vbench_config.get("dimension_profile", "")).strip().lower()
        strict_full_coverage = bool(vbench_config.get("require_full_coverage", False))
        if not strict_full_coverage and profile in {"long_16", "16", "16d", "full", "full_16"}:
            strict_full_coverage = True

        if strict_full_coverage and missing_coverage:
            short_msg = ", ".join(
                f"{name}:{covered}/{total}" for name, covered, total in missing_coverage
            )
            if args.skip_on_error:
                logger.warning(
                    f"{_YELLOW}Coverage incomplete (non-fatal with --skip-on-error): "
                    f"{short_msg}{_RESET}"
                )
            else:
                raise RuntimeError(
                    "VBench coverage check failed under strict mode: "
                    f"{short_msg}. "
                    "Use --skip-on-error to save partial results, or "
                    "--skip <dim> to exclude known-failing dimensions."
                )
    finally:
        if progress_reporter is not None:
            progress_reporter.mark_done()
        if progress_board is not None:
            progress_board.stop()


if __name__ == "__main__":
    main()
