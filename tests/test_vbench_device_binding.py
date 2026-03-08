"""Tests for runtime CUDA device binding in scripts/vbench_runner/core.py."""

import torch

from scripts.vbench_runner.core import _bind_runtime_device


def test_bind_runtime_device_multigpu_uses_local_rank(monkeypatch):
    state = {"idx": 0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: state.__setitem__("idx", idx))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["idx"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    resolved = _bind_runtime_device(
        {"device": "cuda"},
        world_size=4,
        rank=2,
        local_rank=2,
    )

    assert resolved == "cuda:2"
    assert state["idx"] == 2


def test_bind_runtime_device_single_process_keeps_explicit_index(monkeypatch):
    calls: list[int] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: calls.append(int(idx)))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: calls[-1] if calls else 0)

    resolved = _bind_runtime_device(
        {"device": "cuda:1"},
        world_size=1,
        rank=0,
        local_rank=0,
    )

    assert resolved == "cuda:1"
    assert calls == [1]


def test_bind_runtime_device_cpu_skips_cuda_binding(monkeypatch):
    called = {"set_device": False}

    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: called.__setitem__("set_device", True))

    resolved = _bind_runtime_device(
        {"device": "cpu"},
        world_size=4,
        rank=1,
        local_rank=1,
    )

    assert resolved == "cpu"
    assert called["set_device"] is False
