"""
Shared model asset management for VBench runs.

This module guards CLIP checkpoint downloads with:
  - filesystem lock (cross-process)
  - atomic replace on successful download
  - optional SHA256 verification
  - optional auto-repair for corrupted files
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import urllib.request
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

try:
    from .env import logger
except ImportError:
    from vbench_runner.env import logger


@dataclass(frozen=True)
class ClipAsset:
    key: str
    filename: str
    url: str
    sha256: str


CLIP_ASSETS: dict[str, ClipAsset] = {
    "vit_b_32": ClipAsset(
        key="vit_b_32",
        filename="ViT-B-32.pt",
        url=(
            "https://openaipublic.azureedge.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/"
            "ViT-B-32.pt"
        ),
        sha256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    ),
    "vit_l_14": ClipAsset(
        key="vit_l_14",
        filename="ViT-L-14.pt",
        url=(
            "https://openaipublic.azureedge.net/clip/models/"
            "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/"
            "ViT-L-14.pt"
        ),
        sha256="b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
    ),
}

SUBTASK_TO_CLIP_ASSETS: dict[str, tuple[str, ...]] = {
    "background_consistency": ("vit_b_32",),
    "appearance_style": ("vit_b_32",),
    "aesthetic_quality": ("vit_l_14",),
}


def get_vbench_cache_dir(cache_dir: Path | None = None) -> Path:
    """Resolve VBench cache root (same rule as official VBench)."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env_value = os.environ.get("VBENCH_CACHE_DIR")
    if env_value:
        return Path(env_value).expanduser()
    return Path.home() / ".cache" / "vbench"


def required_clip_asset_keys(subtasks: Iterable[str]) -> list[str]:
    """Return sorted CLIP asset keys required by the given subtasks."""
    required: set[str] = set()
    for subtask in subtasks:
        required.update(SUBTASK_TO_CLIP_ASSETS.get(str(subtask), ()))
    return sorted(required)


def clip_asset_path(asset_key: str, cache_dir: Path | None = None) -> Path:
    """Return on-disk target path for a named CLIP asset."""
    if asset_key not in CLIP_ASSETS:
        raise KeyError(f"Unknown CLIP asset key: {asset_key}")
    return get_vbench_cache_dir(cache_dir) / "clip_model" / CLIP_ASSETS[asset_key].filename


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute file SHA256."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_asset_file(path: Path, expected_sha256: str | None) -> bool:
    """Check whether checkpoint exists, is non-empty, and optionally hash-matches."""
    try:
        if not path.is_file():
            return False
        if path.stat().st_size <= 0:
            return False
        if expected_sha256 is None:
            return True
        return file_sha256(path).lower() == expected_sha256.lower()
    except OSError:
        return False


@contextmanager
def _exclusive_file_lock(
    lock_path: Path,
    timeout_sec: float = 1800.0,
    poll_sec: float = 0.2,
):
    """Acquire an advisory cross-process lock for asset creation/repair."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        if fcntl is None:  # pragma: no cover - non-POSIX fallback
            yield
            return

        deadline = time.time() + max(float(timeout_sec), 0.0)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as err:
                if time.time() > deadline:
                    raise TimeoutError(f"Timed out waiting for lock: {lock_path}") from err
                time.sleep(poll_sec)

        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _download_file(url: str, output_path: Path, timeout_sec: float = 600.0) -> None:
    """Download URL to file atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(
        f"{output_path.suffix}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "t2v-eval-vbench-assets"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            with tmp_path.open("wb") as f:
                shutil.copyfileobj(response, f)
        if tmp_path.stat().st_size <= 0:
            raise RuntimeError(f"Downloaded empty file from: {url}")
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _ensure_clip_asset(
    asset: ClipAsset,
    cache_dir: Path,
    verify_sha256: bool,
    repair_corrupted: bool,
    lock_timeout_sec: float,
    download_timeout_sec: float,
) -> str:
    """
    Ensure a single CLIP asset exists and is valid.

    Returns one of: reused / downloaded / repaired
    """
    target_path = cache_dir / "clip_model" / asset.filename
    lock_path = target_path.with_suffix(target_path.suffix + ".lock")
    expected_sha = asset.sha256 if verify_sha256 else None

    with _exclusive_file_lock(lock_path, timeout_sec=lock_timeout_sec):
        if is_valid_asset_file(target_path, expected_sha):
            return "reused"

        existed_before = target_path.exists()
        if existed_before and not repair_corrupted:
            raise RuntimeError(
                f"Detected invalid checkpoint (repair disabled): {target_path}. "
                "Enable repair_corrupted_assets or delete the file manually."
            )

        if existed_before:
            target_path.unlink(missing_ok=True)

        _download_file(asset.url, target_path, timeout_sec=download_timeout_sec)

        if not is_valid_asset_file(target_path, expected_sha):
            target_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded checkpoint failed validation: {target_path} "
                f"(sha256 expected={asset.sha256})"
            )

        return "repaired" if existed_before else "downloaded"


def ensure_clip_assets_for_subtasks(
    subtasks: Iterable[str],
    *,
    cache_dir: Path | None = None,
    verify_sha256: bool = True,
    repair_corrupted: bool = True,
    lock_timeout_sec: float = 1800.0,
    download_timeout_sec: float = 600.0,
) -> dict[str, int]:
    """
    Ensure CLIP model checkpoints needed by configured subtasks are ready.

    Returns:
        {
            "required": int,
            "reused": int,
            "downloaded": int,
            "repaired": int,
        }
    """
    asset_keys = required_clip_asset_keys(subtasks)
    summary = {"required": len(asset_keys), "reused": 0, "downloaded": 0, "repaired": 0}
    if not asset_keys:
        return summary

    cache_root = get_vbench_cache_dir(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    for asset_key in asset_keys:
        asset = CLIP_ASSETS[asset_key]
        action = _ensure_clip_asset(
            asset=asset,
            cache_dir=cache_root,
            verify_sha256=verify_sha256,
            repair_corrupted=repair_corrupted,
            lock_timeout_sec=lock_timeout_sec,
            download_timeout_sec=download_timeout_sec,
        )
        summary[action] += 1
        if action != "reused":
            logger.info("Prepared CLIP asset: %s -> %s", asset.filename, action)

    return summary
