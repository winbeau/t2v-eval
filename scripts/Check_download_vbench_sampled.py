#!/usr/bin/env python3
"""
Download a subset of VBench official sampled videos from Hugging Face.

Default: 8 classic T2V model packs into ./hf/VBench_sampled_video
Requires: huggingface_hub (and access to the gated dataset if applicable).
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, List


DEFAULT_MODELS = [
    "lavie",
    "modelscope",
    "cogvideo",
    "videocrafter-09",
    "videocrafter-1",
    "show-1",
    "pika-all-dimension",
    "gen-2-all-dimension",
]

ALIASES = {
    "videocrafter-0.9": "videocrafter-09",
    "videocrafter-0.9.zip": "videocrafter-09",
    "pika": "pika-all-dimension",
    "gen-2": "gen-2-all-dimension",
}

ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz")


def _normalize_models(models: Iterable[str]) -> List[str]:
    normalized = []
    for m in models:
        key = m.strip()
        key = ALIASES.get(key, key)
        if key and key not in normalized:
            normalized.append(key)
    return normalized


def _select_patterns(files: List[str], suite: str, models: List[str]) -> tuple[list[str], list[str]]:
    patterns: list[str] = []
    missing: list[str] = []

    for model in models:
        # 1) directory-style dataset
        dir_prefix = f"{suite}/{model}/"
        if any(f.startswith(dir_prefix) for f in files):
            patterns.append(f"{dir_prefix}**")
            continue

        # 2) archive-style dataset under suite/
        archive_candidates = [
            f for f in files
            if f.startswith(f"{suite}/")
            and model in f
            and f.lower().endswith(ARCHIVE_EXTS)
        ]
        if archive_candidates:
            patterns.extend(archive_candidates)
            continue

        # 3) archive-style dataset with suite name in filename
        archive_candidates = [
            f for f in files
            if model in f
            and suite in f
            and f.lower().endswith(ARCHIVE_EXTS)
        ]
        if archive_candidates:
            patterns.extend(archive_candidates)
            continue

        missing.append(model)

    # de-dup while keeping order
    seen = set()
    patterns = [p for p in patterns if not (p in seen or seen.add(p))]
    return patterns, missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download selected VBench sampled videos from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default="Vchitect/VBench_sampled_video",
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--suite",
        default="per_dimension",
        choices=["per_dimension", "per_category"],
        help="Which official suite to download",
    )
    parser.add_argument(
        "--dest",
        default="./hf/VBench_sampled_video",
        help="Destination directory for downloads",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model packs to download (default: 8 classic models)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matched files/patterns, do not download",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    models = _normalize_models(args.models)
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    try:
        api = HfApi()
        files = api.list_repo_files(args.repo_id, repo_type="dataset")
    except Exception as exc:
        print("Failed to list repo files. Is the dataset gated?", file=sys.stderr)
        print("Try: huggingface-cli login", file=sys.stderr)
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    patterns, missing = _select_patterns(files, args.suite, models)

    if missing:
        print("Warning: some model packs were not found in repo:", ", ".join(missing), file=sys.stderr)

    if not patterns:
        print("No files matched. Check suite name and model names.", file=sys.stderr)
        sys.exit(1)

    print("Matched patterns/files:")
    for p in patterns:
        print("  -", p)

    if args.dry_run:
        return

    print(f"Downloading to: {dest}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )

    print("Done.")


if __name__ == "__main__":
    main()
