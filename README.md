# 3D MRI ResUNet GAN

This project trains a pix2pix-style 3D MRI translation model with a single active generator: `resunet_3d`.

The current generator is a residual 3D U-Net adapted for TorchIO `64x64x64` patch training and legacy full-volume `128x128x128` training. It decodes back to the input spatial size.

## Current Generator

- Active generator: `ResUNetGenerator3D`
- CLI name: `resunet_3d`
- Input shape: `(B, C, D, H, W)`
- Default training input: `64x64x64` TorchIO grid patches
- Full-volume compatibility: `128x128x128` with `--patch_size 0`
- Output shape: same spatial size as input
- `scale_factor` is forced to `1`
- Legacy generator names such as `mdrn_3d`, `unet_128`, and `unet_256` are ignored and routed to `resunet_3d`
- Legacy options `n_fmdrb` and `skip_compress_ratio` are kept only for old command compatibility

The deepest encoder block disables normalization so batch size `1` works when the bottleneck reaches `1x1x1`; the 64-cube path uses six downsampling stages, and the 128-cube path uses seven.

## Setup

Create and activate your Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA training, install the PyTorch build that matches your CUDA version before installing the rest of the requirements.

## Dataset Layout

The training script supports paired `.npy`, `.npz`, `.nii`, and `.nii.gz` volumes.

Default pix2pix-style layout:

```text
dataset/
  trainA/
    subject_001.npy
  trainB/
    subject_001.npy
```

MRI LR/HR layout:

```text
processed/
  LR/
    subject_001_variant.npy
  HR/
    subject_001.npy
```

For multi-LR datasets, LR filenames can use the HR subject name as a prefix. For example, `100307_T2w_gap_th3.npy` matches `100307_T2w.npy`.

## Training

Default patch-wise single-channel training resizes volumes to `128x128x128`, then trains on non-overlapping `64x64x64` patches:

```bash
python train.py \
  --dataroot /path/to/dataset \
  --phase train \
  --input_nc 1 --output_nc 1 \
  --which_model_netG resunet_3d \
  --which_model_netD basic \
  --fineSize 128 --depthSize 128 \
  --batchSize 1 \
  --device cuda \
  --name mri_resunet_64_patches
```

Using `processed/LR` and `processed/HR`:

```bash
python train.py \
  --dataroot "/path/to/data/processed" \
  --phase train \
  --lr_subdir LR --hr_subdir HR \
  --input_nc 1 --output_nc 1 \
  --which_model_netG resunet_3d \
  --which_model_netD basic \
  --fineSize 128 --depthSize 128 \
  --patch_size 64 --patch_overlap 0 \
  --batchSize 1 \
  --device cuda \
  --name mri_resunet_lr_hr
```

Balanced research profile:

```bash
python train.py \
  --dataroot "/path/to/data/processed" \
  --phase train \
  --lr_subdir LR --hr_subdir HR \
  --input_nc 1 --output_nc 1 \
  --device cuda \
  --research_profile balanced_mri_sr_v1 \
  --fineSize 128 --depthSize 128 \
  --patch_size 64 --patch_overlap 0 \
  --name mri_resunet_balanced_v1
```

Use `--patch_size 0` to disable TorchIO patch extraction and train on full volumes.

Checkpoints and logs are saved under:

```text
checkpoints/<name>/
```

## Perceptual Loss

SwinUNETR perceptual loss:

```bash
python train.py \
  --dataroot /path/to/dataset \
  --use_perceptual_loss \
  --perceptual_backbone swinunetr \
  --perceptual_model_ckpt /path/to/swin.ckpt
```

DINO-style perceptual loss through `timm`:

```bash
python train.py \
  --dataroot /path/to/dataset \
  --use_perceptual_loss \
  --perceptual_backbone dinov3 \
  --perceptual_model_arch vit_small_patch14_dinov2.lvd142m
```

Local DINOv3 repo:

```bash
python train.py \
  --dataroot /path/to/dataset \
  --use_perceptual_loss \
  --perceptual_backbone dinov3 \
  --perceptual_dinov3_repo /path/to/dinov3 \
  --perceptual_model_arch dinov3_vitb16 \
  --perceptual_model_ckpt /path/to/dinov3_vitb16.pth
```

## Tests

Run the full suite:

```bash
pytest -q
```

Focused generator tests:

```bash
pytest tests/test_resunet3d.py -q
```

The generator tests verify same-size `128x128x128` output, multi-channel output, gradient flow, and factory compatibility with legacy generator names.

## Project Structure

```text
models/
  networks3d.py        # ResUNet generator, discriminator, GAN loss
  pix2pix3d_model.py   # Training model wrapper
  perceptual_loss.py   # SwinUNETR/DINO perceptual loss helpers
options/
  base_options.py      # CLI defaults and compatibility handling
  train_options.py     # Training-specific options
tests/
  test_resunet3d.py    # Active generator tests
train.py               # Dataset loader and training loop
```
