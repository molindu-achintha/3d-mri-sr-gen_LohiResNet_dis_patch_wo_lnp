"""Train script for 3D MRI super-resolution (pix2pix-style).

This script wires together the provided 3D generator/discriminator
implementations with a simple paired-volume dataset loader. It supports
`.npy` as well as NIfTI (`.nii` / `.nii.gz`) volumes and assumes paired
low-resolution and high-resolution volumes live in sibling folders. It
supports both `trainA/trainB` layouts and `LR/HR` layouts.

Typical usage (same-size 128^3, single-channel MRI):

    python train.py \
        --dataroot /path/to/dataset \
        --phase train \
        --input_nc 1 --output_nc 1 \
        --which_model_netG resunet_3d --which_model_netD basic \
        --fineSize 128 --depthSize 128 \
        --batchSize 2 --niter 50 --niter_decay 50

Using `processed/LR` and `processed/HR`:

    python train.py \
        --dataroot "/path/with spaces/data/processed" \
        --phase train \
        --lr_subdir LR --hr_subdir HR \
        --input_nc 1 --output_nc 1 \
        --which_model_netG resunet_3d --which_model_netD basic \
        --batchSize 1 --device cuda

Research-backed balanced profile (recommended for your multi-LR dataset):

    python train.py \
        --dataroot "/home/cse_g3/FYP - explo solutions/data/processed" \
        --phase train \
        --lr_subdir LR --hr_subdir HR \
        --input_nc 1 --output_nc 1 \
        --device cuda \
        --research_profile balanced_mri_sr_v1 \
        --name mri_sr_balanced_v1

During training, checkpoints are saved under `checkpoints/<name>/`.

Perceptual loss examples:
    # SwinUNETR perceptual loss
    python train.py --dataroot /path/to/dataset --use_perceptual_loss \
        --perceptual_backbone swinunetr --perceptual_model_ckpt /path/to/swin.ckpt

    # Local DINOv3 perceptual loss
    python train.py --dataroot /path/to/dataset --use_perceptual_loss \
        --perceptual_backbone dinov3 \
        --perceptual_dinov3_repo /Users/molinduachintha/Documents/Work/University/FYP/dinov3 \
        --perceptual_model_arch dinov3_vitb16 \
        --perceptual_model_ckpt /path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

    # timm fallback for DINO-style perceptual loss
    python train.py --dataroot /path/to/dataset --use_perceptual_loss \
        --perceptual_backbone dinov3 --perceptual_model_arch vit_small_patch14_dinov2.lvd142m

Python API with custom perceptual model:
    from train import train
    custom_model = ...
    train(perceptual_model=custom_model)
"""

from __future__ import annotations

import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.models import create_model
from options.train_options import TrainOptions


ALLOWED_EXTS = {".npy", ".npz", ".nii", ".nii.gz"}


def _load_volume(path: Path) -> np.ndarray:
    """Load a 3D volume from .npy/.npz or NIfTI file."""

    suffix = path.suffix.lower()
    if suffix == ".npz":
        # Expect an array stored under default key 'arr_0'
        return np.load(path)["arr_0"]
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".nii", ".gz"} or path.name.endswith(".nii.gz"):
        try:
            import nibabel as nib  # lazy import; common for MRI
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise RuntimeError(
                "nibabel is required to read NIfTI files. Install via `pip install nibabel`."
            ) from exc
        return np.asarray(nib.load(path).get_fdata())

    raise ValueError(f"Unsupported volume format: {path}")


def _to_tensor(vol: np.ndarray, channels: int) -> torch.Tensor:
    """Convert numpy volume to torch tensor with shape (C, D, H, W)."""

    if vol.ndim == 3:
        vol = np.expand_dims(vol, 0)  # (1, D, H, W)
    elif vol.ndim == 4:
        # Try channel-first first, then channel-last
        if vol.shape[0] == channels:
            pass
        elif vol.shape[-1] == channels:
            vol = np.moveaxis(vol, -1, 0)
        else:
            raise ValueError(
                f"Cannot infer channel dimension from shape {vol.shape}; "
                f"expected {channels} channels either first or last."
            )
    else:
        raise ValueError(f"Volume must have 3 or 4 dims, got {vol.ndim}")

    vol = vol.astype(np.float32)

    # Min-max normalize per volume to [-1, 1]
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    vol = vol * 2.0 - 1.0

    return torch.from_numpy(vol)


def _resize_volume(x: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Resize 3D volume (C, D, H, W) to target shape via trilinear interpolation."""

    if tuple(x.shape[1:]) == tuple(target_shape):
        return x

    x = x.unsqueeze(0)  # (1, C, D, H, W)
    x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
    return x.squeeze(0)


def _ensure_header(path: Path, header: List[str]) -> None:
    if path.exists():
        return
    with path.open("w") as handle:
        handle.write("\t".join(header) + "\n")


def _append_row(path: Path, values: List[str]) -> None:
    with path.open("a") as handle:
        handle.write("\t".join(values) + "\n")


def _read_tabular_log(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        return {}

    with path.open("r") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        return {}

    header = lines[0].split("\t")
    cols: Dict[str, List[float]] = {name: [] for name in header}
    for line in lines[1:]:
        values = line.split("\t")
        if len(values) != len(header):
            continue
        for idx, name in enumerate(header):
            try:
                cols[name].append(float(values[idx]))
            except ValueError:
                cols[name].append(float("nan"))
    return cols


def _save_loss_plots(log_dir: Path, iter_log: Path, epoch_log: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not save plots (matplotlib unavailable): {exc}")
        return

    iter_cols = _read_tabular_log(iter_log)
    if iter_cols:
        x = iter_cols.get("total_steps")
        if x:
            fig, ax = plt.subplots(figsize=(9, 5))
            for key, values in iter_cols.items():
                if key in {"epoch", "iter", "total_steps"}:
                    continue
                ax.plot(x, values, label=key, linewidth=1.2)
            ax.set_xlabel("Total Steps")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss by Iteration")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(log_dir / "loss_by_iteration.png", dpi=160)
            plt.close(fig)

    epoch_cols = _read_tabular_log(epoch_log)
    if epoch_cols:
        x = epoch_cols.get("epoch")
        if x:
            fig, ax = plt.subplots(figsize=(9, 5))
            for key, values in epoch_cols.items():
                if key in {"epoch", "total_steps"}:
                    continue
                ax.plot(x, values, marker="o", label=key, linewidth=1.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")
            ax.set_title("Training Loss by Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(log_dir / "loss_by_epoch.png", dpi=160)
            plt.close(fig)


def _save_training_state(model, state_path: Path, epoch: int, next_epoch: int, total_steps: int) -> None:
    state = {
        "epoch": epoch,
        "next_epoch": next_epoch,
        "total_steps": total_steps,
    }
    if hasattr(model, "old_lr"):
        state["old_lr"] = model.old_lr
    if hasattr(model, "optimizer_G"):
        state["optimizer_G"] = model.optimizer_G.state_dict()
    if hasattr(model, "optimizer_D"):
        state["optimizer_D"] = model.optimizer_D.state_dict()
    torch.save(state, state_path)


def _try_resume_state(model, state_path: Path, default_epoch: int) -> Tuple[int, int]:
    if not state_path.exists():
        return default_epoch, 0

    state = torch.load(state_path, map_location="cpu")
    if hasattr(model, "optimizer_G") and "optimizer_G" in state:
        model.optimizer_G.load_state_dict(state["optimizer_G"])
    if hasattr(model, "optimizer_D") and "optimizer_D" in state:
        model.optimizer_D.load_state_dict(state["optimizer_D"])
    if hasattr(model, "old_lr") and "old_lr" in state:
        model.old_lr = state["old_lr"]

    start_epoch = int(state.get("next_epoch", default_epoch))
    total_steps = int(state.get("total_steps", 0))
    print(f"Resumed training state from {state_path} (start_epoch={start_epoch}, total_steps={total_steps})")
    return start_epoch, total_steps


@dataclass
class PairedPaths:
    lr: Path
    hr: Path


class PairedVolumeDataset(Dataset):
    """Paired LR/HR 3D volume dataset with optional resizing and flips.

    For multi-LR datasets, every LR variant matched to an HR subject key
    becomes its own training pair sample.
    """

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
        # Handle double suffix (.nii.gz)
        if path.name.endswith(".nii.gz"):
            return path.name[:-7]
        return path.stem

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]

        lr_np = _load_volume(pair.lr)
        hr_np = _load_volume(pair.hr)

        lr_t = _to_tensor(lr_np, self.input_nc)
        hr_t = _to_tensor(hr_np, self.output_nc)

        # Resize to expected shapes
        if self.depth_size and self.fine_size:
            lr_t = _resize_volume(lr_t, (self.depth_size, self.fine_size, self.fine_size))

        hr_target = (
            (self.depth_size or hr_t.shape[1]) * self.scale_factor,
            (self.fine_size or hr_t.shape[2]) * self.scale_factor,
            (self.fine_size or hr_t.shape[3]) * self.scale_factor,
        )
        hr_t = _resize_volume(hr_t, hr_target)

        if not self.no_flip:
            if random.random() > 0.5:
                lr_t = torch.flip(lr_t, dims=[2])  # H flip
                hr_t = torch.flip(hr_t, dims=[2])
            if random.random() > 0.5:
                lr_t = torch.flip(lr_t, dims=[3])  # W flip
                hr_t = torch.flip(hr_t, dims=[3])
            if random.random() > 0.5:
                lr_t = torch.flip(lr_t, dims=[1])  # D flip
                hr_t = torch.flip(hr_t, dims=[1])

        return {"A": lr_t, "B": hr_t}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(perceptual_model=None):
    opt = TrainOptions().parse()
    set_seed(42)

    checkpoint_dir = Path(opt.checkpoints_dir) / opt.name
    log_dir = checkpoint_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    iter_log_path = log_dir / "losses_by_iteration.txt"
    epoch_log_path = log_dir / "losses_by_epoch.txt"
    iter_state_path = checkpoint_dir / "training_state_latest.pth"

    loss_keys = ["G_GAN", "G_L1", "G_Perc", "D_Real", "D_Fake"]
    _ensure_header(iter_log_path, ["epoch", "iter", "total_steps"] + loss_keys)
    _ensure_header(epoch_log_path, ["epoch", "total_steps"] + loss_keys)

    dataset = PairedVolumeDataset(
        dataroot=opt.dataroot,
        phase=opt.phase,
        lr_subdir=opt.lr_subdir,
        hr_subdir=opt.hr_subdir,
        scale_factor=opt.scale_factor,
        depth_size=opt.depthSize,
        fine_size=opt.fineSize,
        resize_or_crop=opt.resize_or_crop,
        max_dataset_size=opt.max_dataset_size,
        no_flip=opt.no_flip,
        allow_unmatched_lr=opt.allow_unmatched_lr,
        input_nc=opt.input_nc,
        output_nc=opt.output_nc,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=opt.nThreads,
        drop_last=True,
    )

    model = create_model(opt, perceptual_model=perceptual_model)

    start_epoch = opt.epoch_count
    total_steps = 0
    if opt.continue_train:
        start_epoch, total_steps = _try_resume_state(model, iter_state_path, opt.epoch_count)

    dataset_size = len(dataloader)
    print(f"Dataset size: {dataset_size} batches")

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_loss_sum = {key: 0.0 for key in loss_keys}
        epoch_loss_count = 0

        for i, data in enumerate(dataloader):
            total_steps += opt.batchSize

            model.set_input(data)
            model.optimize_parameters()
            errors = model.get_current_errors()

            for key in loss_keys:
                epoch_loss_sum[key] += float(errors.get(key, 0.0))
            epoch_loss_count += 1

            _append_row(
                iter_log_path,
                [
                    str(epoch),
                    str(i + 1),
                    str(total_steps),
                    f"{float(errors.get('G_GAN', 0.0)):.8f}",
                    f"{float(errors.get('G_L1', 0.0)):.8f}",
                    f"{float(errors.get('G_Perc', 0.0)):.8f}",
                    f"{float(errors.get('D_Real', 0.0)):.8f}",
                    f"{float(errors.get('D_Fake', 0.0)):.8f}",
                ],
            )

            if total_steps % opt.print_freq < opt.batchSize:
                err_str = ", ".join([f"{k}: {v:.4f}" for k, v in errors.items()])
                print(f"[Epoch {epoch}][{i+1}/{dataset_size}] {err_str}")

            if total_steps % opt.save_latest_freq < opt.batchSize:
                print(f"Saving latest model (epoch {epoch}, total_steps {total_steps})")
                model.save("latest")
                _save_training_state(
                    model=model,
                    state_path=iter_state_path,
                    epoch=epoch,
                    next_epoch=epoch,
                    total_steps=total_steps,
                )

        epoch_avg = {key: (epoch_loss_sum[key] / max(1, epoch_loss_count)) for key in loss_keys}
        _append_row(
            epoch_log_path,
            [
                str(epoch),
                str(total_steps),
                f"{epoch_avg['G_GAN']:.8f}",
                f"{epoch_avg['G_L1']:.8f}",
                f"{epoch_avg['G_Perc']:.8f}",
                f"{epoch_avg['D_Real']:.8f}",
                f"{epoch_avg['D_Fake']:.8f}",
            ],
        )
        _save_loss_plots(log_dir, iter_log_path, epoch_log_path)

        # Epoch end: periodic checkpoint
        if epoch % opt.save_epoch_freq == 0:
            print(f"Saving checkpoint for epoch {epoch}")
            model.save(epoch)
            _save_training_state(
                model=model,
                state_path=checkpoint_dir / f"training_state_epoch_{epoch}.pth",
                epoch=epoch,
                next_epoch=epoch + 1,
                total_steps=total_steps,
            )

        _save_training_state(
            model=model,
            state_path=iter_state_path,
            epoch=epoch,
            next_epoch=epoch + 1,
            total_steps=total_steps,
        )

        # Linear LR decay after niter epochs
        if epoch > opt.niter:
            model.update_learning_rate()


if __name__ == "__main__":
    train()
