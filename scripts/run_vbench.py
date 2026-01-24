#!/usr/bin/env python3
"""
run_vbench.py
Run VBench evaluation using the official VBench implementation.

This script wraps the official VBench repository (https://github.com/Vchitect/VBench)
to evaluate temporal quality of generated videos.

Usage:
    python scripts/run_vbench.py --config configs/eval.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# =============================================================================
# Setup paths to use official VBench from submodule
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
VBENCH_ROOT = PROJECT_ROOT / "third_party" / "VBench"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_vbench_installation() -> bool:
    """Check if VBench submodule is properly initialized."""
    vbench_init_file = VBENCH_ROOT / "vbench" / "__init__.py"
    if not vbench_init_file.exists():
        logger.error("=" * 70)
        logger.error("VBench submodule not found or not initialized!")
        logger.error("")
        logger.error("Please run the following commands from project root:")
        logger.error("  git submodule update --init --recursive")
        logger.error("")
        logger.error("Or if cloning fresh:")
        logger.error("  git clone --recurse-submodules <repo_url>")
        logger.error("=" * 70)
        return False
    return True


def setup_vbench_path():
    """Add VBench to Python path."""
    vbench_path = str(VBENCH_ROOT)
    if vbench_path not in sys.path:
        sys.path.insert(0, vbench_path)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_video_list(metadata_path: Path) -> list:
    """Get list of video paths from metadata."""
    df = pd.read_csv(metadata_path)
    return df.to_dict("records")


def run_vbench_evaluation(
    video_records: list,
    output_dir: Path,
    config: dict,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Run VBench evaluation on videos using official implementation.

    Args:
        video_records: List of dicts with video_id, group, video_path, prompt
        output_dir: Directory to save results
        config: VBench configuration from eval.yaml
        device: Device to run on

    Returns:
        DataFrame with per-video VBench scores
    """
    setup_vbench_path()

    try:
        from vbench import VBench
    except ImportError as e:
        logger.error(f"Failed to import VBench: {e}")
        logger.error("Please ensure VBench dependencies are installed:")
        logger.error("  pip install -r third_party/VBench/requirements.txt")
        raise

    # VBench configuration
    vbench_config = config.get("metrics", {}).get("vbench", {})
    temporal_only = vbench_config.get("temporal_only", True)
    # Default to safe subtasks (subject_consistency often fails with ZeroDivisionError)
    subtasks = vbench_config.get("subtasks", [
        "temporal_flickering",
        "motion_smoothness",
    ])

    # Results directory for VBench output (must be absolute path)
    results_dir = (output_dir / "vbench_results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # VBench requires full_info_dir to be a JSON file path (not directory)
    # Use the bundled VBench_full_info.json from the VBench submodule
    vbench_full_info = VBENCH_ROOT / "vbench" / "VBench_full_info.json"
    if not vbench_full_info.exists():
        logger.warning(f"VBench full info not found: {vbench_full_info}")
        logger.warning("Will use custom_input mode without full_info")
        vbench_full_info_str = str(results_dir / "dummy_info.json")
        # Create a dummy empty JSON for custom_input mode
        with open(vbench_full_info_str, "w") as f:
            f.write("[]")
    else:
        vbench_full_info_str = str(vbench_full_info.resolve())

    # Prepare video paths for VBench
    # VBench expects a specific format: list of video paths or a directory
    video_paths = [r["video_path"] for r in video_records]

    # Create a directory with symlinks to all videos for VBench
    # VBench custom_input mode expects a directory of videos
    video_dir = results_dir / "input_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for vp in video_paths:
        src = Path(vp)
        dst = video_dir / src.name
        if not dst.exists() and src.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                # If symlink fails, try copying
                import shutil
                shutil.copy2(src, dst)
    video_dir_str = str(video_dir.resolve())

    results = []

    # Initialize VBench
    try:
        # VBench requires full_info_dir (JSON file path) and output_path (directory)
        vbench = VBench(
            device=device,
            full_info_dir=vbench_full_info_str,
            output_path=str(results_dir),
        )
    except Exception as e:
        logger.error(f"Failed to initialize VBench: {e}")
        logger.error("This may be due to missing model weights.")
        logger.error("Please check VBench documentation for weight download instructions.")
        raise

    # Run evaluation for each subtask
    logger.info(f"Running VBench evaluation with subtasks: {subtasks}")

    for subtask in subtasks:
        logger.info(f"Evaluating subtask: {subtask}")
        try:
            # VBench evaluate method
            # Use mode='custom_input' since we're providing custom videos
            # (not following VBench's standard naming convention)
            vbench.evaluate(
                videos_path=video_dir_str,  # VBench expects directory path for custom_input
                name=subtask,
                dimension_list=[subtask],
                local=True,
                read_frame=False,
                mode='custom_input',
            )

            # VBench saves results to {output_path}/{name}_eval_results.json
            result_file = results_dir / f"{subtask}_eval_results.json"
            if result_file.exists():
                with open(result_file) as f:
                    subtask_data = json.load(f)
                # VBench output format: {dimension: [avg_score, [{"video_path": ..., "video_results": ...}, ...]]}
                if subtask in subtask_data:
                    count_before = len(results)
                    dimension_data = subtask_data[subtask]

                    # Handle format: [avg_score, [per_video_results]]
                    if isinstance(dimension_data, list) and len(dimension_data) >= 2:
                        per_video_list = dimension_data[1]  # Second element is the list of per-video results
                        if isinstance(per_video_list, list):
                            for item in per_video_list:
                                if isinstance(item, dict):
                                    vp = item.get("video_path", item.get("video_name", ""))
                                    score = item.get("video_results", item.get("score", None))
                                    if vp and score is not None:
                                        video_id = Path(vp).stem
                                        results.append({
                                            "video_id": video_id,
                                            "subtask": subtask,
                                            "score": float(score),
                                        })
                    # Fallback: handle other formats
                    elif isinstance(dimension_data, list):
                        for item in dimension_data:
                            if isinstance(item, dict):
                                vp = item.get("video_path", item.get("video_name", ""))
                                score = item.get("video_results", item.get("score", None))
                                if vp and score is not None:
                                    video_id = Path(vp).stem
                                    results.append({
                                        "video_id": video_id,
                                        "subtask": subtask,
                                        "score": float(score),
                                    })

                    logger.info(f"Parsed {len(results) - count_before} results for {subtask}")
            else:
                logger.warning(f"Result file not found: {result_file}")
        except Exception as e:
            logger.warning(f"Failed to run subtask {subtask}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            continue

    # Convert to DataFrame and pivot
    if results:
        df_results = pd.DataFrame(results)
        # Pivot to get one row per video with subtask scores as columns
        df_pivot = df_results.pivot(
            index="video_id",
            columns="subtask",
            values="score"
        ).reset_index()

        # Compute aggregate temporal score (mean of all subtasks)
        subtask_cols = [c for c in df_pivot.columns if c != "video_id"]
        if subtask_cols:
            df_pivot["vbench_temporal_score"] = df_pivot[subtask_cols].mean(axis=1)

        return df_pivot
    else:
        logger.warning("No VBench results obtained")
        return pd.DataFrame()


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
    import tempfile

    vbench_config = config.get("metrics", {}).get("vbench", {})
    subtasks = vbench_config.get("subtasks", [
        "temporal_flickering",
        "motion_smoothness",
    ])

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
            "--videos_path", str(videos_dir.resolve()),
            "--dimension", subtask,
            "--output_path", str(results_dir.resolve()),
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
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

                # VBench outputs different formats depending on version/mode
                # Handle list format: [{"video_name": "...", "dimension": "...", "score": ...}, ...]
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            video_name = item.get("video_name", item.get("video_path", ""))
                            dimension = item.get("dimension", item.get("name", ""))
                            score = item.get("score", item.get("value", None))
                            if video_name and score is not None:
                                results.append({
                                    "video_id": Path(video_name).stem,
                                    "subtask": dimension,
                                    "score": score,
                                })
                # Handle dict format: {"video_name": {"dimension": score, ...}, ...}
                elif isinstance(data, dict):
                    for video_name, scores in data.items():
                        if isinstance(scores, dict):
                            for dim, score in scores.items():
                                results.append({
                                    "video_id": Path(video_name).stem,
                                    "subtask": dim,
                                    "score": score,
                                })
                        elif isinstance(scores, (int, float)):
                            # Single score format
                            dim = result_file.stem  # Use filename as dimension
                            results.append({
                                "video_id": Path(video_name).stem,
                                "subtask": dim,
                                "score": scores,
                            })
        except Exception as e:
            logger.warning(f"Failed to parse {result_file}: {e}")

    if results:
        df_results = pd.DataFrame(results)
        df_pivot = df_results.pivot(
            index="video_id",
            columns="subtask",
            values="score"
        ).reset_index()

        subtask_cols = [c for c in df_pivot.columns if c != "video_id"]
        if subtask_cols:
            df_pivot["vbench_temporal_score"] = df_pivot[subtask_cols].mean(axis=1)

        return df_pivot

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Run VBench evaluation using official implementation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--skip-on-error", action="store_true",
        help="Skip VBench if errors occur (don't fail pipeline)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing results"
    )
    args = parser.parse_args()

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

    output_dir = Path(paths_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_metadata = output_dir / paths_config["processed_metadata"]
    vbench_output = output_dir / "vbench_per_video.csv"

    # Check if already exists
    if vbench_output.exists() and not args.force:
        logger.info(f"VBench results already exist: {vbench_output}")
        logger.info("Use --force to recompute")
        return
    elif vbench_output.exists() and args.force:
        logger.info(f"Force recomputing: {vbench_output}")

    # Load video metadata
    if not processed_metadata.exists():
        logger.error(f"Processed metadata not found: {processed_metadata}")
        logger.error("Run preprocess_videos.py first")
        if args.skip_on_error:
            sys.exit(0)
        sys.exit(1)

    video_records = get_video_list(processed_metadata)
    logger.info(f"Loaded {len(video_records)} videos for VBench evaluation")

    device = runtime_config.get("device", "cuda")

    # Try Python API first, fall back to CLI
    try:
        logger.info("Attempting VBench evaluation via Python API...")
        df_results = run_vbench_evaluation(
            video_records=video_records,
            output_dir=output_dir,
            config=config,
            device=device,
        )
    except Exception as e:
        logger.warning(f"Python API failed: {e}")
        logger.info("Falling back to CLI-based evaluation...")
        try:
            df_results = run_vbench_cli_fallback(
                video_records=video_records,
                output_dir=output_dir,
                config=config,
            )
        except Exception as e2:
            logger.error(f"CLI fallback also failed: {e2}")
            if args.skip_on_error:
                logger.warning("Skipping VBench evaluation")
                # Create empty result file
                pd.DataFrame(columns=["video_id", "vbench_temporal_score"]).to_csv(
                    vbench_output, index=False
                )
                sys.exit(0)
            raise

    # Merge with metadata to get group info
    df_meta = pd.read_csv(processed_metadata)[["video_id", "group"]]
    if not df_results.empty:
        df_results = df_results.merge(df_meta, on="video_id", how="left")

    # Save results
    df_results.to_csv(vbench_output, index=False)
    logger.info(f"VBench results saved to: {vbench_output}")

    # Print summary
    if "vbench_temporal_score" in df_results.columns:
        logger.info("\nVBench Temporal Score Summary by Group:")
        summary = df_results.groupby("group")["vbench_temporal_score"].agg(["mean", "std"])
        logger.info(f"\n{summary}")


if __name__ == "__main__":
    main()
