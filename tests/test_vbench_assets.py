"""
Tests for scripts/vbench_runner/assets.py — shared CLIP asset prefetch/repair.
"""

import hashlib
from pathlib import Path

import pytest

from scripts.vbench_runner import assets


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_required_clip_asset_keys_deduplicates():
    keys = assets.required_clip_asset_keys(
        [
            "appearance_style",
            "background_consistency",
            "aesthetic_quality",
            "appearance_style",
            "motion_smoothness",
        ]
    )
    assert keys == ["vit_b_32", "vit_l_14"]


def test_ensure_clip_assets_reuses_valid_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    valid_bytes = b"valid-weights"
    expected_hash = _sha256(valid_bytes)

    monkeypatch.setattr(
        assets,
        "CLIP_ASSETS",
        {
            "vit_b_32": assets.ClipAsset(
                key="vit_b_32",
                filename="ViT-B-32.pt",
                url="https://example.invalid/ViT-B-32.pt",
                sha256=expected_hash,
            )
        },
    )
    monkeypatch.setattr(
        assets,
        "SUBTASK_TO_CLIP_ASSETS",
        {"appearance_style": ("vit_b_32",)},
    )

    target = tmp_path / "clip_model" / "ViT-B-32.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(valid_bytes)

    calls: list[tuple[str, Path, float]] = []

    def _fake_download(url: str, output_path: Path, timeout_sec: float = 600.0):
        calls.append((url, output_path, timeout_sec))
        output_path.write_bytes(valid_bytes)

    monkeypatch.setattr(assets, "_download_file", _fake_download)

    summary = assets.ensure_clip_assets_for_subtasks(
        ["appearance_style"],
        cache_dir=tmp_path,
    )

    assert summary == {"required": 1, "reused": 1, "downloaded": 0, "repaired": 0}
    assert calls == []


def test_ensure_clip_assets_repairs_corrupted_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    valid_bytes = b"valid-weights"
    expected_hash = _sha256(valid_bytes)

    monkeypatch.setattr(
        assets,
        "CLIP_ASSETS",
        {
            "vit_b_32": assets.ClipAsset(
                key="vit_b_32",
                filename="ViT-B-32.pt",
                url="https://example.invalid/ViT-B-32.pt",
                sha256=expected_hash,
            )
        },
    )
    monkeypatch.setattr(
        assets,
        "SUBTASK_TO_CLIP_ASSETS",
        {"appearance_style": ("vit_b_32",)},
    )

    target = tmp_path / "clip_model" / "ViT-B-32.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"corrupted")

    def _fake_download(url: str, output_path: Path, timeout_sec: float = 600.0):
        output_path.write_bytes(valid_bytes)

    monkeypatch.setattr(assets, "_download_file", _fake_download)

    summary = assets.ensure_clip_assets_for_subtasks(
        ["appearance_style"],
        cache_dir=tmp_path,
        verify_sha256=True,
        repair_corrupted=True,
    )

    assert summary == {"required": 1, "reused": 0, "downloaded": 0, "repaired": 1}
    assert target.read_bytes() == valid_bytes


def test_ensure_clip_assets_raises_when_repair_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    valid_bytes = b"valid-weights"
    expected_hash = _sha256(valid_bytes)

    monkeypatch.setattr(
        assets,
        "CLIP_ASSETS",
        {
            "vit_b_32": assets.ClipAsset(
                key="vit_b_32",
                filename="ViT-B-32.pt",
                url="https://example.invalid/ViT-B-32.pt",
                sha256=expected_hash,
            )
        },
    )
    monkeypatch.setattr(
        assets,
        "SUBTASK_TO_CLIP_ASSETS",
        {"appearance_style": ("vit_b_32",)},
    )

    target = tmp_path / "clip_model" / "ViT-B-32.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"corrupted")

    with pytest.raises(RuntimeError, match="repair disabled"):
        assets.ensure_clip_assets_for_subtasks(
            ["appearance_style"],
            cache_dir=tmp_path,
            verify_sha256=True,
            repair_corrupted=False,
        )


def test_ensure_clip_assets_skip_hash_reuses_nonempty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        assets,
        "CLIP_ASSETS",
        {
            "vit_b_32": assets.ClipAsset(
                key="vit_b_32",
                filename="ViT-B-32.pt",
                url="https://example.invalid/ViT-B-32.pt",
                sha256="deadbeef",
            )
        },
    )
    monkeypatch.setattr(
        assets,
        "SUBTASK_TO_CLIP_ASSETS",
        {"appearance_style": ("vit_b_32",)},
    )

    target = tmp_path / "clip_model" / "ViT-B-32.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not-empty")

    summary = assets.ensure_clip_assets_for_subtasks(
        ["appearance_style"],
        cache_dir=tmp_path,
        verify_sha256=False,
    )

    assert summary == {"required": 1, "reused": 1, "downloaded": 0, "repaired": 0}
