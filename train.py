"""Train script for 3D MRI super-resolution (pix2pix-style).

This script wires together the provided 3D generator/discriminator
implementations with a simple paired-volume dataset loader. It supports
`.npy` as well as NIfTI (`.nii` / `.nii.gz`) volumes and assumes paired
low-resolution and high-resolution volumes live in sibling folders. It
supports both `trainA/trainB` layouts and `LR/HR` layouts.

Typical usage (128^3 volumes -> non-overlapping 64^3 training patches):

    python train.py \
        --dataroot /path/to/dataset \
        --phase train \
        --input_nc 1 --output_nc 1 \
        --which_model_netG resunet_3d --which_model_netD basic \
        --fineSize 128 --depthSize 128 \
        --patch_size 64 --patch_overlap 0 \
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

from data.paired_volume_dataset import PairedPatchVolumeDataset, PairedPaths, PairedVolumeDataset
from options.train_options import TrainOptions
from training.runner import run_training
from training.seed import set_seed


def train(perceptual_model=None):
    opt = TrainOptions().parse()
    set_seed(42)
    run_training(opt, perceptual_model=perceptual_model)


if __name__ == "__main__":
    train()
