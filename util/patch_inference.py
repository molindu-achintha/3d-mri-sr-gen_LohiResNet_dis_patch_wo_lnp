"""
Patch-wise 3D inference using TorchIO GridSampler/GridAggregator.

This utility cuts a 3D MRI volume into non-overlapping patches, runs them
through a model, and stitches the outputs back into a full volume.

Requirements:
    pip install torchio
"""

from typing import Tuple, Union

import torch
import numpy as np
import torchio as tio


def _to_channel_first_no_batch(volume: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Normalize input to shape (C, X, Y, Z) without batch dim."""
    if isinstance(volume, np.ndarray):
        vol = torch.from_numpy(volume)
    else:
        vol = volume

    if vol.dim() == 3:            # (D, H, W)
        vol = vol.unsqueeze(0)    # (1, D, H, W)
    elif vol.dim() == 4:
        # assume (C, D, H, W) already
        pass
    else:
        raise ValueError(f"Expected volume with 3 or 4 dims, got {vol.shape}")
    return vol.float()


@torch.no_grad()
def run_patched_inference(
    model: torch.nn.Module,
    volume: Union[np.ndarray, torch.Tensor],
    patch_size: Tuple[int, int, int],
    device: torch.device = None,
) -> torch.Tensor:
    """
    Run non-overlapping patch inference and rebuild the full volume.

    Args:
        model:       3D model expecting input shape (B, C, D, H, W).
        volume:      Numpy array or torch tensor with shape (D, H, W) or
                     (C, D, H, W), values in model's expected range.
        patch_size:  (px, py, pz) patch spatial size.
        device:      torch.device or None (auto CUDA if available).

    Returns:
        torch.Tensor with same shape as input (C, D, H, W).
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()

    vol_ch_first = _to_channel_first_no_batch(volume)

    subject = tio.Subject(
        volume=tio.ScalarImage(tensor=vol_ch_first)  # shape (C, X, Y, Z)
    )

    sampler = tio.GridSampler(subject, patch_size=patch_size, patch_overlap=0)
    aggregator = tio.GridAggregator(sampler, overlap_mode="crop")

    loader = torch.utils.data.DataLoader(sampler, batch_size=1)

    for batch in loader:
        patch = batch["volume"][tio.DATA].to(device)  # (1, C, px, py, pz)
        # add batch dimension for model
        pred = model(patch)
        aggregator.add_batch(pred.cpu(), batch[tio.LOCATION])

    output = aggregator.get_output_tensor()  # (C, X, Y, Z)
    return output
