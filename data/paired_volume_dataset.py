from __future__ import annotations

import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .volume_io import ALLOWED_EXTS, load_volume, resize_volume, to_tensor


def _import_torchio():
    try:
        import torchio as tio
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "TorchIO is required for patch training. Install it with `pip install torchio` "
            "or pass `--patch_size 0` to train on full volumes."
        ) from exc
    return tio


def _patch_tuple(patch_size: int) -> Tuple[int, int, int]:
    size = int(patch_size)
    if size <= 0:
        raise ValueError("patch_size must be a positive integer when patch training is enabled.")
    return (size, size, size)


@dataclass
class PairedPaths:
    lr: Path
    hr: Path


class PairedVolumeDataset(Dataset):
    """Paired LR/HR 3D volume dataset with optional resizing and flips."""

    def __init__(
        self,
        dataroot: str | Path,
        phase: str = "train",
        lr_subdir: str | None = None,
        hr_subdir: str | None = None,
        scale_factor: int = 1,
        depth_size: int | None = None,
        fine_size: int | None = None,
        resize_or_crop: str = "resize_and_crop",
        max_dataset_size: int = float("inf"),
        no_flip: bool = False,
        allow_unmatched_lr: bool = False,
        input_nc: int = 1,
        output_nc: int = 1,
    ) -> None:
        self.dataroot = Path(dataroot)
        self.phase = phase
        self.lr_subdir = (lr_subdir or "").strip()
        self.hr_subdir = (hr_subdir or "").strip()
        self.allow_unmatched_lr = allow_unmatched_lr
        self.scale_factor = int(scale_factor)
        self.depth_size = depth_size
        self.fine_size = fine_size
        self.resize_or_crop = resize_or_crop
        self.max_dataset_size = max_dataset_size
        self.no_flip = no_flip
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.lr_dir, self.hr_dir = self._resolve_data_dirs()
        self.pairs: List[PairedPaths] = self._build_pairs()

    def _resolve_data_dirs(self) -> Tuple[Path, Path]:
        if self.lr_subdir or self.hr_subdir:
            if not self.lr_subdir or not self.hr_subdir:
                raise ValueError(
                    "When overriding dataset folders, both --lr_subdir and --hr_subdir must be set."
                )
            lr_dir = self.dataroot / self.lr_subdir
            hr_dir = self.dataroot / self.hr_subdir
            if not lr_dir.is_dir() or not hr_dir.is_dir():
                raise FileNotFoundError(
                    f"Configured LR/HR folders not found: lr='{lr_dir}', hr='{hr_dir}'"
                )
            return lr_dir, hr_dir

        phase_lr = self.dataroot / f"{self.phase}A"
        phase_hr = self.dataroot / f"{self.phase}B"
        if phase_lr.is_dir() and phase_hr.is_dir():
            return phase_lr, phase_hr

        fallback_lr = self.dataroot / "LR"
        fallback_hr = self.dataroot / "HR"
        if fallback_lr.is_dir() and fallback_hr.is_dir():
            return fallback_lr, fallback_hr

        raise FileNotFoundError(
            "Could not resolve dataset directories. Expected either "
            f"'{phase_lr.name}/{phase_hr.name}', or 'LR/HR' under {self.dataroot}, "
            "or explicit --lr_subdir/--hr_subdir."
        )

    def _list_vols(self, folder: Path) -> List[Path]:
        files = [
            p
            for p in sorted(folder.iterdir())
            if p.is_file()
            and (p.suffix.lower() in ALLOWED_EXTS or p.name.endswith(".nii.gz"))
        ]
        if not files:
            raise FileNotFoundError(f"No volumes with {ALLOWED_EXTS} under {folder}")
        return files

    @staticmethod
    def _match_lr_to_hr_key(lr_key: str, hr_keys: List[str]) -> List[str]:
        return [hr_key for hr_key in hr_keys if lr_key == hr_key or lr_key.startswith(f"{hr_key}_")]

    def _build_pairs(self) -> List[PairedPaths]:
        lr_files = self._list_vols(self.lr_dir)
        hr_files = self._list_vols(self.hr_dir)

        hr_map = {self._stem(p): p for p in hr_files}
        hr_keys = sorted(hr_map.keys(), key=lambda key: (-len(key), key))

        pair_rows = []
        per_subject_counts = Counter()
        unmatched_lr = []
        for lf in lr_files:
            lr_key = self._stem(lf)
            matches = self._match_lr_to_hr_key(lr_key, hr_keys)

            if len(matches) == 0:
                if self.allow_unmatched_lr:
                    unmatched_lr.append(lf.name)
                    continue
                raise FileNotFoundError(
                    f"Could not find matching HR volume for LR '{lf.name}' in {self.hr_dir}"
                )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Ambiguous HR match for LR '{lf.name}'. Candidates: {matches}"
                )

            hr_key = matches[0]
            pair_rows.append((hr_key, PairedPaths(lr=lf, hr=hr_map[hr_key])))
            per_subject_counts[hr_key] += 1

        pair_rows.sort(key=lambda item: (item[0], item[1].lr.name))
        pairs = [pair for _hr_key, pair in pair_rows]

        if len(pairs) > self.max_dataset_size:
            pairs = pairs[: self.max_dataset_size]

        avg_variants = (sum(per_subject_counts.values()) / max(1, len(per_subject_counts)))
        print(
            "Pairing summary: "
            f"HR={len(hr_files)}, LR={len(lr_files)}, matched_pairs={len(pairs)}, "
            f"subjects_with_pairs={len(per_subject_counts)}, avg_lr_per_hr={avg_variants:.2f}"
        )
        top_subjects = sorted(per_subject_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        if top_subjects:
            top_text = ", ".join([f"{subject}:{count}" for subject, count in top_subjects])
            print(f"Top subject pair counts: {top_text}")
        if unmatched_lr:
            warnings.warn(
                f"Skipped {len(unmatched_lr)} unmatched LR files because --allow_unmatched_lr is set.",
                RuntimeWarning,
            )

        return pairs

    @staticmethod
    def _stem(path: Path) -> str:
        if path.name.endswith(".nii.gz"):
            return path.name[:-7]
        return path.stem

    def _load_pair_tensors(self, index: int, apply_flip: bool = True) -> Dict[str, torch.Tensor]:
        pair = self.pairs[index]

        lr_t = to_tensor(load_volume(pair.lr), self.input_nc)
        hr_t = to_tensor(load_volume(pair.hr), self.output_nc)

        if self.depth_size and self.fine_size:
            lr_t = resize_volume(lr_t, (self.depth_size, self.fine_size, self.fine_size))

        hr_target = (
            (self.depth_size or hr_t.shape[1]) * self.scale_factor,
            (self.fine_size or hr_t.shape[2]) * self.scale_factor,
            (self.fine_size or hr_t.shape[3]) * self.scale_factor,
        )
        hr_t = resize_volume(hr_t, hr_target)

        if apply_flip and not self.no_flip:
            if random.random() > 0.5:
                lr_t = torch.flip(lr_t, dims=[2])
                hr_t = torch.flip(hr_t, dims=[2])
            if random.random() > 0.5:
                lr_t = torch.flip(lr_t, dims=[3])
                hr_t = torch.flip(hr_t, dims=[3])
            if random.random() > 0.5:
                lr_t = torch.flip(lr_t, dims=[1])
                hr_t = torch.flip(hr_t, dims=[1])

        return {"A": lr_t, "B": hr_t}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        return self._load_pair_tensors(index, apply_flip=True)


class PairedPatchVolumeDataset(PairedVolumeDataset):
    """Paired LR/HR dataset that yields aligned TorchIO grid patches."""

    def __init__(
        self,
        *args,
        patch_size: int = 64,
        patch_overlap: int = 0,
        **kwargs,
    ) -> None:
        self.patch_size = int(patch_size)
        self.patch_overlap = int(patch_overlap)
        if self.patch_size <= 0:
            raise ValueError("PairedPatchVolumeDataset requires patch_size > 0.")
        if self.patch_overlap < 0:
            raise ValueError("patch_overlap must be >= 0.")
        if self.patch_overlap >= self.patch_size:
            raise ValueError("patch_overlap must be smaller than patch_size.")

        self._tio = _import_torchio()
        super().__init__(*args, **kwargs)
        self._patch_shape = _patch_tuple(self.patch_size)
        self._patch_index: List[Tuple[int, int]] = self._build_patch_index()

    def _make_subject(self, tensors: Dict[str, torch.Tensor]):
        return self._tio.Subject(
            A=self._tio.ScalarImage(tensor=tensors["A"]),
            B=self._tio.ScalarImage(tensor=tensors["B"]),
        )

    def _make_sampler(self, pair_index: int, apply_flip: bool):
        tensors = self._load_pair_tensors(pair_index, apply_flip=apply_flip)
        if tensors["A"].shape[1:] != tensors["B"].shape[1:]:
            raise ValueError(
                "Patch training requires LR and HR tensors to have the same spatial shape; "
                f"got A={tuple(tensors['A'].shape)} and B={tuple(tensors['B'].shape)}."
            )
        if any(dim < self.patch_size for dim in tensors["A"].shape[1:]):
            raise ValueError(
                f"Patch size {self.patch_size} does not fit volume shape "
                f"{tuple(tensors['A'].shape[1:])}."
            )
        subject = self._make_subject(tensors)
        return self._tio.GridSampler(
            subject,
            patch_size=self._patch_shape,
            patch_overlap=self.patch_overlap,
        )

    def _build_patch_index(self) -> List[Tuple[int, int]]:
        patch_index: List[Tuple[int, int]] = []
        for pair_index in range(len(self.pairs)):
            sampler = self._make_sampler(pair_index, apply_flip=False)
            patch_index.extend((pair_index, patch_index_in_pair) for patch_index_in_pair in range(len(sampler)))
        if not patch_index:
            raise RuntimeError("TorchIO GridSampler produced no patches.")
        print(
            "Patch summary: "
            f"patch_size={self.patch_size}, patch_overlap={self.patch_overlap}, "
            f"patches={len(patch_index)}"
        )
        return patch_index

    def __len__(self) -> int:
        return len(self._patch_index)

    def __getitem__(self, index: int):
        pair_index, patch_index = self._patch_index[index]
        sampler = self._make_sampler(pair_index, apply_flip=True)
        sample = sampler[patch_index]
        result = {
            "A": sample["A"][self._tio.DATA].float(),
            "B": sample["B"][self._tio.DATA].float(),
        }
        if self._tio.LOCATION in sample:
            result["location"] = sample[self._tio.LOCATION]
        return result
