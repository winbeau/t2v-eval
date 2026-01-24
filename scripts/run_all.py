#!/usr/bin/env python3
"""
run_all.py
One-click entry point for the complete T2V evaluation pipeline.

This script:
1. Checks git submodules are initialized
2. Runs all evaluation steps in sequence
3. Provides clear progress and error reporting

Usage:
    python scripts/run_all.py --config configs/eval.yaml
    python scripts/run_all.py --config configs/eval.yaml --skip-vbench
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _submodule_key_file(name: str, path: Path) -> Path:
    """Return a key file path used to validate a submodule."""
    if name == "VBench":
        return path / "vbench" / "__init__.py"
    key_file = path / "t2v_metrics" / "__init__.py"
    return key_file if key_file.exists() else (path / "t2v_metrics.py")


def _submodule_ready(name: str, path: Path) -> bool:
    """Check if a submodule directory looks initialized and usable."""
    if not path.exists():
        return False
    git_file = path / ".git"
    if not git_file.exists():
        return False
    key_file = _submodule_key_file(name, path)
    return key_file.exists()


def check_submodules() -> bool:
    """
    Check if git submodules are properly initialized.

    Returns:
        True if all submodules are ready, False otherwise.
    """
    submodules = [
        ("VBench", PROJECT_ROOT / "third_party" / "VBench"),
        ("t2v_metrics", PROJECT_ROOT / "third_party" / "t2v_metrics"),
    ]

    all_ready = True

    for name, path in submodules:
        if not path.exists():
            logger.error(f"Submodule {name} directory not found: {path}")
            all_ready = False
            continue

        git_file = path / ".git"
        if not git_file.exists():
            logger.error(f"Submodule {name} not initialized (no .git): {path}")
            all_ready = False
            continue

        key_file = _submodule_key_file(name, path)
        if not key_file.exists():
            logger.error(f"Submodule {name} incomplete: {key_file} not found")
            all_ready = False
            continue

        logger.info(f"✓ Submodule {name}: OK")

    return all_ready


def init_submodules() -> bool:
    """
    Initialize git submodules only when missing.

    Returns:
        True if successful, False otherwise.
    """
    submodules = [
        ("VBench", PROJECT_ROOT / "third_party" / "VBench"),
        ("t2v_metrics", PROJECT_ROOT / "third_party" / "t2v_metrics"),
    ]

    to_init = [(name, path) for name, path in submodules if not _submodule_ready(name, path)]
    if not to_init:
        logger.info("All submodules already initialized; skipping.")
        return True

    logger.info("Initializing missing submodules...")

    try:
        for name, path in to_init:
            if path.exists() and (path / ".git").exists():
                logger.info(f"Repairing submodule {name} working tree...")
                subprocess.run(
                    ["git", "-C", str(path), "reset", "--hard", "HEAD"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                subprocess.run(
                    ["git", "-C", str(path), "clean", "-fd"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

            if _submodule_ready(name, path):
                logger.info(f"✓ Submodule {name}: OK")
                continue

            logger.info(f"Fetching submodule {name}...")
            result = subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive", "--force", str(path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.error(f"Submodule init failed for {name}: {result.stderr}")
                return False

            if not _submodule_ready(name, path):
                logger.error(f"Submodule {name} still incomplete after init.")
                return False

        logger.info("Submodules initialized successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Submodule initialization timed out")
        return False
    except Exception as e:
        logger.error(f"Submodule initialization error: {e}")
        return False


def run_script(
    script_name: str,
    config_path: str,
    extra_args: list = None,
    skip_on_error: bool = False,
) -> bool:
    """
    Run a Python script from the scripts directory.

    Returns:
        True if successful, False otherwise.
    """
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path), "--config", config_path]

    if extra_args:
        cmd.extend(extra_args)

    if skip_on_error:
        cmd.append("--skip-on-error")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running: {script_name}")
    logger.info(f"{'=' * 60}")

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            timeout=7200,  # 2 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"{script_name} failed with code {result.returncode}")
            return False

        logger.info(f"✓ {script_name} completed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"{script_name} timed out")
        return False
    except Exception as e:
        logger.error(f"{script_name} error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete T2V evaluation pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--auto-init-submodules", action="store_true",
        help="Automatically initialize submodules if missing"
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Skip HuggingFace export step"
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip video preprocessing step"
    )
    parser.add_argument(
        "--skip-vbench", action="store_true",
        help="Skip VBench evaluation (if weights unavailable)"
    )
    parser.add_argument(
        "--skip-clipvqa", action="store_true",
        help="Skip CLIP/VQA evaluation"
    )
    parser.add_argument(
        "--skip-flicker", action="store_true",
        help="Skip Flicker evaluation"
    )
    parser.add_argument(
        "--skip-niqe", action="store_true",
        help="Skip NIQE evaluation"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force recomputation of all metrics"
    )
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())

    logger.info("=" * 60)
    logger.info("T2V-Eval: Text-to-Video Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Config: {config_path}")

    # Step 0: Check submodules
    logger.info("\n[Step 0/7] Checking git submodules...")

    if not check_submodules():
        if args.auto_init_submodules:
            logger.info("Attempting to initialize submodules...")
            if not init_submodules():
                logger.error("Failed to initialize submodules")
                logger.error("Please run manually:")
                logger.error("  git submodule update --init --recursive")
                sys.exit(1)
        else:
            logger.error("")
            logger.error("Git submodules are not initialized!")
            logger.error("")
            logger.error("Please run the following command from project root:")
            logger.error("  git submodule update --init --recursive")
            logger.error("")
            logger.error("Or use --auto-init-submodules flag to auto-initialize")
            sys.exit(1)

    # Build extra args
    extra_args = []
    if args.force:
        extra_args.append("--force")
        logger.info("🔄 Force mode enabled: All metrics will be recomputed")

    # Step 1: Export from HuggingFace
    if not args.skip_export:
        logger.info("\n[Step 1/7] Exporting dataset from HuggingFace...")
        if not run_script("export_from_hf.py", config_path, extra_args):
            logger.error("Export failed. Please check your HF dataset configuration.")
            sys.exit(1)
    else:
        logger.info("\n[Step 1/7] Skipping HuggingFace export")

    # Step 2: Preprocess videos
    if not args.skip_preprocess:
        logger.info("\n[Step 2/7] Preprocessing videos...")
        if not run_script("preprocess_videos.py", config_path, extra_args):
            logger.error("Preprocessing failed.")
            sys.exit(1)
    else:
        logger.info("\n[Step 2/7] Skipping video preprocessing")

    # Step 3: CLIP/VQA evaluation
    if not args.skip_clipvqa:
        logger.info("\n[Step 3/7] Running CLIP/VQA evaluation...")
        run_script("run_clip_or_vqa.py", config_path, extra_args, skip_on_error=True)
    else:
        logger.info("\n[Step 3/7] Skipping CLIP/VQA evaluation")

    # Step 4: VBench evaluation
    if not args.skip_vbench:
        logger.info("\n[Step 4/7] Running VBench evaluation...")
        run_script("run_vbench.py", config_path, extra_args, skip_on_error=True)
    else:
        logger.info("\n[Step 4/7] Skipping VBench evaluation")

    # Step 5: Flicker evaluation
    if not args.skip_flicker:
        logger.info("\n[Step 5/7] Running Flicker evaluation...")
        run_script("run_flicker.py", config_path, extra_args)
    else:
        logger.info("\n[Step 5/7] Skipping Flicker evaluation")

    # Step 6: NIQE evaluation
    if not args.skip_niqe:
        logger.info("\n[Step 6/7] Running NIQE evaluation...")
        run_script("run_niqe.py", config_path, extra_args)
    else:
        logger.info("\n[Step 6/7] Skipping NIQE evaluation")

    # Step 7: Summarize
    logger.info("\n[Step 7/7] Generating summary...")
    run_script("summarize.py", config_path, extra_args)

    # Done
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed!")
    logger.info("=" * 60)
    logger.info(f"\nOutput files:")
    logger.info(f"  - outputs/per_video_metrics.csv")
    logger.info(f"  - outputs/group_summary.csv")


if __name__ == "__main__":
    main()
