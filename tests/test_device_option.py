"""Tests for explicit CPU/CUDA device option parsing."""

import sys

import pytest
import torch

from options.train_options import TrainOptions


def _parse_train_options(monkeypatch, tmp_path, args, cuda_available):
    set_device_calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "set_device", lambda idx: set_device_calls.append(idx))

    checkpoints_dir = tmp_path / "checkpoints"
    argv = [
        "train.py",
        "--dataroot",
        str(tmp_path),
        "--name",
        "device_option_test",
        "--checkpoints_dir",
        str(checkpoints_dir),
    ] + args
    monkeypatch.setattr(sys, "argv", argv)
    opt = TrainOptions().parse()
    return opt, set_device_calls


def test_device_cpu_forces_cpu(monkeypatch, tmp_path):
    opt, set_device_calls = _parse_train_options(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        args=["--device", "cpu"],
        cuda_available=True,
    )
    assert opt.device == "cpu"
    assert opt.gpu_ids == []
    assert set_device_calls == []


def test_device_auto_falls_back_to_cpu_when_cuda_unavailable(monkeypatch, tmp_path):
    opt, set_device_calls = _parse_train_options(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        args=["--device", "auto"],
        cuda_available=False,
    )
    assert opt.device == "cpu"
    assert opt.gpu_ids == []
    assert set_device_calls == []


def test_device_cuda_raises_when_cuda_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _idx: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--dataroot",
            str(tmp_path),
            "--name",
            "device_option_test_cuda",
            "--checkpoints_dir",
            str(tmp_path / "checkpoints"),
            "--device",
            "cuda",
        ],
    )
    with pytest.raises(RuntimeError, match="CUDA requested"):
        TrainOptions().parse()
