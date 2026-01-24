#!/usr/bin/env python3
"""
run_clip_or_vqa.py
Run CLIPScore or VQAScore evaluation using the official t2v_metrics implementation.

This script wraps the official t2v_metrics repository (https://github.com/linzhiqiu/t2v_metrics)
to evaluate text-video alignment.

Usage:
    python scripts/run_clip_or_vqa.py --config configs/eval.yaml
    python scripts/run_clip_or_vqa.py --config configs/eval.yaml --mode vqa
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# =============================================================================
# Setup paths to use official t2v_metrics from submodule
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
T2V_METRICS_ROOT = PROJECT_ROOT / "third_party" / "t2v_metrics"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_t2v_metrics_installation() -> bool:
    """Check if t2v_metrics submodule is properly initialized."""
    # Check for key files that indicate proper initialization
    init_file = T2V_METRICS_ROOT / "t2v_metrics" / "__init__.py"
    if not init_file.exists():
        # Try alternative structure
        alt_file = T2V_METRICS_ROOT / "t2v_metrics.py"
        if not alt_file.exists():
            logger.error("=" * 70)
            logger.error("t2v_metrics submodule not found or not initialized!")
            logger.error("")
            logger.error("Please run the following commands from project root:")
            logger.error("  git submodule update --init --recursive")
            logger.error("")
            logger.error("Or if cloning fresh:")
            logger.error("  git clone --recurse-submodules <repo_url>")
            logger.error("=" * 70)
            return False
    return True


def setup_t2v_metrics_path():
    """Add t2v_metrics to Python path."""
    t2v_path = str(T2V_METRICS_ROOT)
    if t2v_path not in sys.path:
        sys.path.insert(0, t2v_path)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def read_video_frames(
    video_path: str,
    num_frames: int = 8,
    size: Optional[int] = None,
) -> np.ndarray:
    """
    Read frames from video file.

    Returns:
        numpy array of shape (N, H, W, C) in RGB format
    """
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C)

        return frames
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()

        return np.array(frames)


def compute_clip_score_official(
    video_path: str,
    prompt: str,
    num_frames: int,
    model,
    preprocess,
    device: str,
) -> float:
    """
    Compute CLIPScore using official t2v_metrics implementation.
    """
    setup_t2v_metrics_path()

    try:
        # Try to use t2v_metrics official implementation
        from t2v_metrics import compute_clip_score

        score = compute_clip_score(
            video_path=video_path,
            text=prompt,
            num_frames=num_frames,
        )
        return float(score)
    except (ImportError, AttributeError):
        # Fallback to direct implementation using the model
        frames = read_video_frames(video_path, num_frames=num_frames)

        import torch
        from PIL import Image

        # Process frames
        frame_features = []
        for frame in frames:
            img = Image.fromarray(frame)
            img_input = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img_input)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                frame_features.append(feat)

        # Average frame features
        frame_features = torch.cat(frame_features, dim=0)
        video_feat = frame_features.mean(dim=0, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        # Encode text
        import open_clip
        text_input = open_clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(text_input)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        score = (video_feat @ text_feat.T).item()

        # Scale to [0, 100] like original CLIPScore
        score = max(0, score) * 100

        return score


def compute_vqa_score_official(
    video_path: str,
    prompt: str,
    num_frames: int,
    device: str,
) -> float:
    """
    Compute VQAScore using official t2v_metrics implementation.
    """
    setup_t2v_metrics_path()

    try:
        # Use official t2v_metrics VQAScore
        from t2v_metrics import VQAScore

        # Initialize VQAScore model
        vqa_model = VQAScore(device=device)

        score = vqa_model.score(
            video_path=video_path,
            text=prompt,
            num_frames=num_frames,
        )
        return float(score)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Official VQAScore not available: {e}")
        logger.warning("Falling back to CLIP-based score")
        raise


def run_clip_evaluation(
    video_records: list,
    config: dict,
    device: str,
) -> pd.DataFrame:
    """
    Run CLIP-based evaluation on all videos.
    """
    clip_config = config.get("metrics", {}).get("clip_or_vqa", {})
    model_name = clip_config.get("model_name", "ViT-B-32")
    pretrained = clip_config.get("pretrained", "openai")
    num_frames = clip_config.get("num_frames_for_score", 8)
    aggregation = clip_config.get("aggregation", "mean")

    logger.info(f"Loading CLIP model: {model_name} ({pretrained})")

    # Try to use t2v_metrics' recommended model loading
    setup_t2v_metrics_path()

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device).eval()
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        logger.error("Please install open_clip: pip install open_clip_torch")
        raise

    results = []

    for record in tqdm(video_records, desc="Computing CLIPScore"):
        video_id = record["video_id"]
        video_path = record["video_path"]
        prompt = record.get("prompt", "")
        group = record.get("group", "unknown")

        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            continue

        if not prompt:
            logger.warning(f"No prompt for video {video_id}, skipping")
            continue

        try:
            score = compute_clip_score_official(
                video_path=video_path,
                prompt=prompt,
                num_frames=num_frames,
                model=model,
                preprocess=preprocess,
                device=device,
            )

            results.append({
                "video_id": video_id,
                "group": group,
                "clip_score": score,
            })
        except Exception as e:
            logger.warning(f"Failed to compute CLIPScore for {video_id}: {e}")
            continue

    return pd.DataFrame(results)


def run_vqa_evaluation(
    video_records: list,
    config: dict,
    device: str,
) -> pd.DataFrame:
    """
    Run VQAScore evaluation on all videos using official t2v_metrics.
    """
    setup_t2v_metrics_path()

    clip_config = config.get("metrics", {}).get("clip_or_vqa", {})
    num_frames = clip_config.get("num_frames_for_score", 8)

    results = []
    vqa_model = None

    for record in tqdm(video_records, desc="Computing VQAScore"):
        video_id = record["video_id"]
        video_path = record["video_path"]
        prompt = record.get("prompt", "")
        group = record.get("group", "unknown")

        if not os.path.exists(video_path):
            logger.warning(f"Video not found: {video_path}")
            continue

        if not prompt:
            logger.warning(f"No prompt for video {video_id}, skipping")
            continue

        try:
            # Lazy load VQAScore model
            if vqa_model is None:
                try:
                    from t2v_metrics import VQAScore
                    vqa_model = VQAScore(device=device)
                except ImportError:
                    logger.error("VQAScore not available in t2v_metrics")
                    logger.error("Falling back to CLIP-based scoring")
                    return run_clip_evaluation(video_records, config, device)

            score = vqa_model.score(
                video_path=video_path,
                text=prompt,
                num_frames=num_frames,
            )

            results.append({
                "video_id": video_id,
                "group": group,
                "vqa_score": float(score),
            })
        except Exception as e:
            logger.warning(f"Failed to compute VQAScore for {video_id}: {e}")
            continue

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Run CLIPScore/VQAScore using official t2v_metrics"
    )
    parser.add_argument(
        "--config", type=str, default="configs/eval.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["clip", "vqa"],
        help="Override evaluation mode (default: from config)"
    )
    parser.add_argument(
        "--skip-on-error", action="store_true",
        help="Skip if errors occur"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing results"
    )
    args = parser.parse_args()

    # Check t2v_metrics installation
    if not check_t2v_metrics_installation():
        if args.skip_on_error:
            logger.warning("Skipping CLIP/VQA evaluation due to missing submodule")
            sys.exit(0)
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    paths_config = config["paths"]
    runtime_config = config.get("runtime", {})
    clip_config = config.get("metrics", {}).get("clip_or_vqa", {})

    # Determine mode
    mode = args.mode or clip_config.get("mode", "clip")
    logger.info(f"Running evaluation in {mode.upper()} mode")

    output_dir = Path(paths_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_metadata = output_dir / paths_config["processed_metadata"]
    output_file = output_dir / "clipvqa_per_video.csv"

    # Check if already exists
    if output_file.exists() and not args.force:
        logger.info(f"Results already exist: {output_file}")
        logger.info("Use --force to recompute")
        return
    elif output_file.exists() and args.force:
        logger.info(f"Force recomputing: {output_file}")

    # Load video metadata
    if not processed_metadata.exists():
        logger.error(f"Processed metadata not found: {processed_metadata}")
        logger.error("Run preprocess_videos.py first")
        if args.skip_on_error:
            sys.exit(0)
        sys.exit(1)

    df_meta = pd.read_csv(processed_metadata)
    video_records = df_meta.to_dict("records")
    logger.info(f"Loaded {len(video_records)} videos for evaluation")

    device = runtime_config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Run evaluation
    try:
        if mode == "vqa":
            df_results = run_vqa_evaluation(video_records, config, device)
            score_col = "vqa_score"
        else:
            df_results = run_clip_evaluation(video_records, config, device)
            score_col = "clip_score"
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.skip_on_error:
            # Create empty result
            pd.DataFrame(columns=["video_id", "group", "clip_or_vqa_score"]).to_csv(
                output_file, index=False
            )
            sys.exit(0)
        raise

    # Rename score column to unified name
    if not df_results.empty:
        df_results = df_results.rename(columns={score_col: "clip_or_vqa_score"})
        df_results["score_type"] = mode

    # Save results
    df_results.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")

    # Print summary
    if "clip_or_vqa_score" in df_results.columns:
        logger.info(f"\n{mode.upper()} Score Summary by Group:")
        summary = df_results.groupby("group")["clip_or_vqa_score"].agg(["mean", "std"])
        logger.info(f"\n{summary}")


if __name__ == "__main__":
    main()
