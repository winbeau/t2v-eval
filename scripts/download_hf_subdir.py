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
import fnmatch
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def normalize_subdir(subdir: str | None) -> str | None:
    if not subdir:
        return None
    normalized = subdir.strip("/").rstrip("/")
    return normalized or None


def matches_patterns(path: str, includes: list[str] | None, excludes: list[str] | None) -> bool:
    include_ok = True
    if includes:
        include_ok = any(fnmatch.fnmatch(path, pat) for pat in includes)
    exclude_hit = False
    if excludes:
        exclude_hit = any(fnmatch.fnmatch(path, pat) for pat in excludes)
    return include_ok and not exclude_hit


def resolve_remote_subdir(
    files: list[str],
    repo_id: str,
    subdir: str | None,
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
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)
    remote_subdir = resolve_remote_subdir(files, repo_id, subdir)

    if remote_subdir:
        prefix = f"{remote_subdir}/"
        scoped_files = [f for f in files if f.startswith(prefix)]
    else:
        scoped_files = list(files)

    if not scoped_files:
        raise FileNotFoundError(
            f"No files found under subdir '{remote_subdir or subdir}' in repo '{repo_id}'."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_files: list[str] = []
    for remote_file in scoped_files:
        relative = remote_file[len(prefix) :] if remote_subdir else remote_file
        # Allow include/exclude matching by either full path or relative path.
        path_for_match = [remote_file, relative]
        if includes or excludes:
            if not any(matches_patterns(p, includes, excludes) for p in path_for_match):
                continue
        selected_files.append(remote_file)

    if not selected_files:
        raise FileNotFoundError(
            f"No files matched include/exclude filters under '{remote_subdir or '/'}'."
        )

    for remote_file in selected_files:
        cached_file = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=token,
            filename=remote_file,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        if strip_prefix and remote_subdir:
            relative = remote_file[len(prefix) :]
            target = output_dir / relative
        else:
            target = output_dir / remote_file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached_file, target)


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
