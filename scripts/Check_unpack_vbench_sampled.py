#!/usr/bin/env python3
"""
Unpack VBench sampled video archives into a usable folder structure.

Looks for .zip/.tar/.tar.gz/.tgz in the dataset folder and extracts them.
Also performs a light normalization so that {root}/{suite}/{group}/... exists.
"""

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path


ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz")


def _is_within_directory(base: Path, target: Path) -> bool:
    base = base.resolve()
    target = target.resolve()
    return str(target).startswith(str(base))


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    for member in zf.namelist():
        target = dest / member
        if not _is_within_directory(dest, target):
            raise RuntimeError(f"Unsafe path in zip: {member}")
    zf.extractall(dest)


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    for member in tf.getmembers():
        target = dest / member.name
        if not _is_within_directory(dest, target):
            raise RuntimeError(f"Unsafe path in tar: {member.name}")
    tf.extractall(dest)


def _extract_archive(path: Path, dest: Path) -> None:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            _safe_extract_zip(zf, dest)
        return
    if path.suffix in {".tar", ".tgz"} or path.name.endswith(".tar.gz"):
        mode = "r:gz" if path.name.endswith(".tar.gz") or path.suffix == ".tgz" else "r"
        with tarfile.open(path, mode) as tf:
            _safe_extract_tar(tf, dest)
        return
    raise ValueError(f"Unsupported archive: {path}")


def _normalize_suite(root: Path, suite: str) -> None:
    suite_dir = root / suite
    suite_dir.mkdir(parents=True, exist_ok=True)

    # Common wrappers found after extraction
    candidates = [
        root / "t2v_sampled_videos" / suite,
        root / "VBench_sampled_video" / suite,
        root / suite / suite,
    ]

    for cand in candidates:
        if cand.exists() and cand.is_dir() and cand != suite_dir:
            for child in cand.iterdir():
                target = suite_dir / child.name
                if target.exists():
                    continue
                shutil.move(str(child), str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack VBench sampled video archives.")
    parser.add_argument(
        "--root",
        default="./hf/VBench_sampled_video",
        help="Root directory containing downloaded archives",
    )
    parser.add_argument(
        "--suite",
        default="per_dimension",
        choices=["per_dimension", "per_category"],
        help="Suite folder to normalize",
    )
    parser.add_argument(
        "--remove-archives",
        action="store_true",
        help="Delete archives after extraction",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    archives = []
    for ext in ARCHIVE_EXTS:
        archives.extend(root.rglob(f"*{ext}"))

    if not archives:
        print("No archives found; skipping extraction.")
        _normalize_suite(root, args.suite)
        return

    for archive in sorted(archives):
        print(f"Extracting: {archive}")
        _extract_archive(archive, root)
        if args.remove_archives:
            archive.unlink()

    _normalize_suite(root, args.suite)
    print("Done.")


if __name__ == "__main__":
    main()
