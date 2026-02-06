#!/usr/bin/env python3
"""
Download a specific subdirectory from a HuggingFace dataset repo into a local folder.

Example:
  python scripts/download_hf_subdir.py \
    --repo-id hf/AdaHead \
    --subdir Exp_OscStable_Head_Window \
    --output-dir hf/AdaHead/Exp_OscStable_Head_Window
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def normalize_subdir(subdir: str | None) -> str | None:
    if not subdir:
        return None
    normalized = subdir.strip("/").rstrip("/")
    return normalized or None


def build_allow_patterns(subdir: str | None, includes: list[str] | None) -> list[str] | None:
    if not subdir and not includes:
        return None

    patterns: list[str] = []
    if subdir:
        patterns.append(f"{subdir}/**")
    if includes:
        patterns.extend(includes)
    return patterns


def resolve_remote_subdir(
    repo_id: str,
    subdir: str | None,
    repo_type: str,
    revision: str | None,
    token: str | None,
) -> str | None:
    """
    Resolve user-provided subdir to an actual prefix in the remote repo.

    Supports:
      - Exact path (e.g. videos/Exp_A)
      - Basename-only path (e.g. Exp_A), if unique in repo
    """
    subdir = normalize_subdir(subdir)
    if not subdir:
        return None

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)

    # Exact prefix match first.
    if any(f == subdir or f.startswith(f"{subdir}/") for f in files):
        return subdir

    # Fallback: basename match across file parents.
    target_name = Path(subdir).name
    matches: set[str] = set()
    for file_path in files:
        parts = Path(file_path).parts
        for idx in range(len(parts) - 1):
            prefix = "/".join(parts[: idx + 1])
            if Path(prefix).name == target_name:
                matches.add(prefix)

    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        preview = ", ".join(sorted(matches)[:5])
        raise ValueError(
            f"Subdir '{subdir}' is ambiguous. Matched: {preview}. "
            "Please pass full subdir path."
        )

    raise FileNotFoundError(
        f"Subdir '{subdir}' not found in remote repo '{repo_id}'. "
        "Please check subdir path."
    )


def download_subdir(
    repo_id: str,
    subdir: str | None,
    output_dir: Path,
    repo_type: str,
    revision: str | None,
    token: str | None,
    includes: list[str] | None,
    excludes: list[str] | None,
    strip_prefix: bool,
    cache_dir: Path | None,
) -> None:
    remote_subdir = resolve_remote_subdir(repo_id, subdir, repo_type, revision, token)
    allow_patterns = build_allow_patterns(remote_subdir, includes)

    if strip_prefix and remote_subdir:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                token=token,
                local_dir=str(tmp_path),
                allow_patterns=allow_patterns,
                ignore_patterns=excludes,
                cache_dir=str(cache_dir) if cache_dir else None,
            )

            src = tmp_path / remote_subdir
            if not src.exists():
                raise FileNotFoundError(f"Subdir not found after download: {src}")

            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, output_dir, dirs_exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=str(output_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=excludes,
            cache_dir=str(cache_dir) if cache_dir else None,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a subdirectory from a HuggingFace dataset repo."
    )
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. hf/AdaHead")
    parser.add_argument("--subdir", default=None, help="Subdirectory inside the repo to download")
    parser.add_argument("--output-dir", required=True, help="Local output directory")
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="HF repo type (default: dataset)",
    )
    parser.add_argument("--revision", default=None, help="Git revision or tag")
    parser.add_argument("--token", default=None, help="HF token (or login via huggingface-cli)")
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Extra allow patterns (can be repeated), e.g. --include '*.csv'",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Ignore patterns (can be repeated), e.g. --exclude '*.pt'",
    )
    parser.add_argument(
        "--strip-prefix",
        action="store_true",
        help="Place subdir contents directly into output-dir (default for subdir)",
    )
    parser.add_argument(
        "--no-strip-prefix",
        dest="strip_prefix",
        action="store_false",
        help="Keep subdir path under output-dir",
    )
    parser.set_defaults(strip_prefix=True)
    parser.add_argument("--cache-dir", default=None, help="Optional HF cache dir")

    args = parser.parse_args()

    download_subdir(
        repo_id=args.repo_id,
        subdir=args.subdir,
        output_dir=Path(args.output_dir),
        repo_type=args.repo_type,
        revision=args.revision,
        token=args.token,
        includes=args.include,
        excludes=args.exclude,
        strip_prefix=args.strip_prefix,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )

    print(f"Done. Downloaded to: {args.output_dir}")


if __name__ == "__main__":
    main()
