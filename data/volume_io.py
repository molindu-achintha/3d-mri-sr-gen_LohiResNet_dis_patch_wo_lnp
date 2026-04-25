from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


ALLOWED_EXTS = {".npy", ".npz", ".nii", ".nii.gz"}


def load_volume(path: Path) -> np.ndarray:
    """Load a 3D volume from .npy/.npz or NIfTI file."""

    suffix = path.suffix.lower()
    if suffix == ".npz":
        return np.load(path)["arr_0"]
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".nii", ".gz"} or path.name.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover - handled at runtime
            raise RuntimeError(
                "nibabel is required to read NIfTI files. Install via `pip install nibabel`."
            ) from exc
        return np.asarray(nib.load(path).get_fdata())

    raise ValueError(f"Unsupported volume format: {path}")


def to_tensor(vol: np.ndarray, channels: int) -> torch.Tensor:
    """Convert numpy volume to torch tensor with shape (C, D, H, W)."""

    if vol.ndim == 3:
        vol = np.expand_dims(vol, 0)
    elif vol.ndim == 4:
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

    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    vol = vol * 2.0 - 1.0

    return torch.from_numpy(vol)


def resize_volume(x: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """Resize 3D volume (C, D, H, W) to target shape via trilinear interpolation."""

    if tuple(x.shape[1:]) == tuple(target_shape):
        return x

    x = x.unsqueeze(0)
    x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
    return x.squeeze(0)
