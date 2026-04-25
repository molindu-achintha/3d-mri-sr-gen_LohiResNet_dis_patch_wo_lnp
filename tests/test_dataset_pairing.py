"""Tests for multi-LR to single-HR dataset pairing behavior."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from train import PairedPatchVolumeDataset, PairedVolumeDataset  # noqa: E402


def _write_npy(path, value=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((4, 4, 4), value, dtype=np.float32))


def _write_npy_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array.astype(np.float32))


def _stem(path):
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def test_auto_detects_lr_hr_layout(tmp_path):
    dataroot = tmp_path / "processed"
    _write_npy(dataroot / "HR" / "100307_T2w.npy", 1.0)
    _write_npy(dataroot / "HR" / "101309_T2w.npy", 2.0)
    _write_npy(dataroot / "LR" / "100307_T2w_gap_th3_gap0mm.npy", 1.0)
    _write_npy(dataroot / "LR" / "100307_T2w_inplane_ds1.npy", 1.0)
    _write_npy(dataroot / "LR" / "101309_T2w_gap_th3_gap0mm.npy", 1.0)
    _write_npy(dataroot / "LR" / "101309_T2w_inplane_ds1.npy", 1.0)

    ds = PairedVolumeDataset(dataroot=dataroot, phase="train", no_flip=True)
    assert ds.lr_dir.name == "LR"
    assert ds.hr_dir.name == "HR"
    assert len(ds) == 4


def test_expands_multi_lr_pairs_per_subject(tmp_path):
    dataroot = tmp_path / "processed"
    subjects = ["100307_T2w", "101309_T2w"]
    variants = [
        "gap_th3_gap0mm",
        "gap_th4_gap0mm",
        "gap_th5_gap0mm",
        "gap_th5_gap1mm",
        "inplane_ds1",
        "inplane_ds2",
        "thick_3mm",
        "thick_5mm",
    ]
    for subject in subjects:
        _write_npy(dataroot / "HR" / f"{subject}.npy", 1.0)
        for variant in variants:
            _write_npy(dataroot / "LR" / f"{subject}_{variant}.npy", 1.0)

    ds = PairedVolumeDataset(dataroot=dataroot, phase="train", no_flip=True)
    assert len(ds) == len(subjects) * len(variants)


def test_prefix_pairing_maps_lr_to_correct_hr(tmp_path):
    dataroot = tmp_path / "processed"
    _write_npy(dataroot / "HR" / "100307_T2w.npy", 1.0)
    _write_npy(dataroot / "HR" / "101309_T2w.npy", 2.0)
    _write_npy(dataroot / "LR" / "100307_T2w_inplane_ds1.npy", 1.0)
    _write_npy(dataroot / "LR" / "101309_T2w_thick_3mm.npy", 1.0)

    ds = PairedVolumeDataset(dataroot=dataroot, phase="train", no_flip=True)
    lr_to_hr = {_stem(pair.lr): _stem(pair.hr) for pair in ds.pairs}

    assert lr_to_hr["100307_T2w_inplane_ds1"] == "100307_T2w"
    assert lr_to_hr["101309_T2w_thick_3mm"] == "101309_T2w"


def test_unmatched_lr_strict_mode_raises(tmp_path):
    dataroot = tmp_path / "processed"
    _write_npy(dataroot / "HR" / "100307_T2w.npy", 1.0)
    _write_npy(dataroot / "LR" / "999999_T2w_inplane_ds1.npy", 1.0)

    with pytest.raises(FileNotFoundError, match="Could not find matching HR"):
        PairedVolumeDataset(dataroot=dataroot, phase="train", no_flip=True)


def test_unmatched_lr_relaxed_mode_skips_with_warning(tmp_path):
    dataroot = tmp_path / "processed"
    _write_npy(dataroot / "HR" / "100307_T2w.npy", 1.0)
    _write_npy(dataroot / "LR" / "100307_T2w_inplane_ds1.npy", 1.0)
    _write_npy(dataroot / "LR" / "999999_T2w_inplane_ds1.npy", 1.0)

    with pytest.warns(RuntimeWarning, match="Skipped 1 unmatched LR files"):
        ds = PairedVolumeDataset(
            dataroot=dataroot,
            phase="train",
            allow_unmatched_lr=True,
            no_flip=True,
        )
    assert len(ds) == 1


def test_traina_trainb_layout_still_supported(tmp_path):
    dataroot = tmp_path / "processed"
    _write_npy(dataroot / "trainA" / "subject1.npy", 1.0)
    _write_npy(dataroot / "trainB" / "subject1.npy", 2.0)

    ds = PairedVolumeDataset(dataroot=dataroot, phase="train", no_flip=True)
    assert ds.lr_dir.name == "trainA"
    assert ds.hr_dir.name == "trainB"
    assert len(ds) == 1


def test_patch_dataset_extracts_eight_aligned_64_cubes(tmp_path):
    pytest.importorskip("torchio")
    dataroot = tmp_path / "processed"
    volume = np.arange(128 * 128 * 128, dtype=np.float32).reshape(128, 128, 128)
    _write_npy_array(dataroot / "LR" / "subject1.npy", volume)
    _write_npy_array(dataroot / "HR" / "subject1.npy", volume)

    ds = PairedPatchVolumeDataset(
        dataroot=dataroot,
        phase="train",
        no_flip=True,
        depth_size=128,
        fine_size=128,
        patch_size=64,
        patch_overlap=0,
        input_nc=1,
        output_nc=1,
    )

    assert len(ds) == 8
    sample = ds[0]
    assert sample["A"].shape == (1, 64, 64, 64)
    assert sample["B"].shape == (1, 64, 64, 64)
    assert torch.equal(sample["A"], sample["B"])

    locations = {tuple(int(v) for v in ds[i]["location"]) for i in range(len(ds))}
    expected_locations = {
        (d, h, w, d + 64, h + 64, w + 64)
        for d in (0, 64)
        for h in (0, 64)
        for w in (0, 64)
    }
    assert locations == expected_locations


def test_patch_dataset_keeps_lr_hr_patch_coordinates_matched(tmp_path):
    pytest.importorskip("torchio")
    dataroot = tmp_path / "processed"
    volume = np.arange(128 * 128 * 128, dtype=np.float32).reshape(128, 128, 128)
    _write_npy_array(dataroot / "LR" / "subject1.npy", volume)
    _write_npy_array(dataroot / "HR" / "subject1.npy", volume)

    ds = PairedPatchVolumeDataset(
        dataroot=dataroot,
        phase="train",
        no_flip=True,
        depth_size=128,
        fine_size=128,
        patch_size=64,
        patch_overlap=0,
        input_nc=1,
        output_nc=1,
    )

    for index in range(len(ds)):
        sample = ds[index]
        assert torch.equal(sample["A"], sample["B"])
