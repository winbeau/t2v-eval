#!/usr/bin/env python3
"""
Download classic VBench sampled videos from Google Drive.

This avoids Hugging Face listing/rate-limit issues for gated datasets.
Default: 8 classic T2V model packs (per_dimension).
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List


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
    "pika": "pika-all-dimension",
    "gen-2": "gen-2-all-dimension",
}

# Google Drive file IDs from third_party/VBench/sampled_videos/README.md (per_dimension)
GDRIVE_FILES: Dict[str, Dict[str, str]] = {
    "per_dimension": {
        "lavie": "1hviZzsInIgJA96ppVj4B2DHhTZWeM4nc",
        "modelscope": "1UH2-lALFShjBywyImjDPPHTpE43eoMQE",
        "cogvideo": "1-oAHf6inm4CFeldKktWerXkjwQ_q26Ic",
        "videocrafter-09": "1VoNPAttMFOV_6FIYCGW4fzFE9m18Ry22",
        "videocrafter-1": "1FCRj48-Yv7LM7XGgfDCvIo7Kb9EId5KX",
        "show-1": "1QOInCcCI04LQ38BiY0o4oLehAFQfiVh2",
        "pika-all-dimension": "1G2VVD5ArLxYtKeAVdANnxNNAPlP2bbZO",
        "gen-2-all-dimension": "1tPL_PMmnBM4518UNiu52nhQCbUmF0A8q",
    }
}

# Expected archive extensions from VBench README (per_dimension)
ARCHIVE_NAMES = {
    "lavie": "lavie.zip",
    "modelscope": "modelscope.zip",
    "cogvideo": "cogvideo.zip",
    "videocrafter-09": "videocrafter-09.zip",
    "videocrafter-1": "videocrafter-1.tar.gz",
    "show-1": "show-1.tar.gz",
    "pika-all-dimension": "pika-all-dimension.zip",
    "gen-2-all-dimension": "gen-2-all-dimension.tar.gz",
}


def _normalize_models(models: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for m in models:
        key = ALIASES.get(m.strip(), m.strip())
        if key and key not in normalized:
            normalized.append(key)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VBench sampled videos from Google Drive.")
    parser.add_argument(
        "--suite",
        default="per_dimension",
        choices=["per_dimension"],
        help="Suite to download (per_dimension only for classic 8-model pack).",
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
        help="Print planned downloads without fetching",
    )
    parser.add_argument(
        "--ignore-failures",
        action="store_true",
        help="Continue on download failures and exit 0",
    )
    args = parser.parse_args()

    try:
        import gdown  # noqa: F401
    except ImportError:
        print("gdown not installed. Run: pip install gdown", file=sys.stderr)
        sys.exit(1)

    suite = args.suite
    root = Path(args.dest).resolve()
    root.mkdir(parents=True, exist_ok=True)

    models = _normalize_models(args.models)
    mapping = GDRIVE_FILES.get(suite, {})

    planned = []
    for model in models:
        file_id = mapping.get(model)
        if not file_id:
            print(f"Skip unknown model (no file id): {model}", file=sys.stderr)
            continue
        name = ARCHIVE_NAMES.get(model, f"{model}.zip")
        planned.append((model, file_id, root / name))

    if not planned:
        raise SystemExit("No downloads planned. Check model names.")

    print("Planned downloads:")
    for model, file_id, out_path in planned:
        print(f"  - {model}: {out_path.name} (id={file_id})")

    if args.dry_run:
        return

    import gdown

    failures = []
    for model, file_id, out_path in planned:
        if out_path.exists():
            print(f"Skip existing: {out_path.name}")
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {model} -> {out_path}")
        try:
            gdown.download(url, str(out_path), quiet=False)
        except Exception as exc:
            failures.append((model, file_id, str(exc)))
            print(f"Failed: {model} ({file_id})")
            print(f"  Error: {exc}")
            continue

    if failures:
        print("\nSome downloads failed:")
        for model, file_id, err in failures:
            print(f"  - {model} (id={file_id})")
        print("\nTip: open the Google Drive link in a browser and download manually,")
        print("then place the file in the destination folder with the exact name.")
        if not args.ignore_failures:
            raise SystemExit(2)

    print("Done.")


if __name__ == "__main__":
    main()
