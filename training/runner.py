from pathlib import Path

from torch.utils.data import DataLoader

from data.paired_volume_dataset import PairedPatchVolumeDataset, PairedVolumeDataset
from models.models import create_model
from .logs import append_row, ensure_header, save_loss_plots
from .state import save_training_state, try_resume_state


LOSS_KEYS = ["G_GAN", "G_L1", "G_Perc", "D_Real", "D_Fake"]


def _build_dataset(opt):
    full_depth_size = opt.depthSize
    full_fine_size = opt.fineSize
    patch_size = int(getattr(opt, "patch_size", 64))
    dataset_kwargs = dict(
        dataroot=opt.dataroot,
        phase=opt.phase,
        lr_subdir=opt.lr_subdir,
        hr_subdir=opt.hr_subdir,
        scale_factor=opt.scale_factor,
        depth_size=full_depth_size,
        fine_size=full_fine_size,
        resize_or_crop=opt.resize_or_crop,
        max_dataset_size=opt.max_dataset_size,
        no_flip=opt.no_flip,
        allow_unmatched_lr=opt.allow_unmatched_lr,
        input_nc=opt.input_nc,
        output_nc=opt.output_nc,
    )

    if patch_size <= 0:
        return PairedVolumeDataset(**dataset_kwargs)

    dataset = PairedPatchVolumeDataset(
        **dataset_kwargs,
        patch_size=patch_size,
        patch_overlap=getattr(opt, "patch_overlap", 0),
    )
    opt.depthSize = patch_size
    opt.fineSize = patch_size
    print(
        "Patch training enabled: "
        f"full_volume=({full_depth_size}, {full_fine_size}, {full_fine_size}), "
        f"effective_model_input=({opt.depthSize}, {opt.fineSize}, {opt.fineSize})"
    )
    return dataset


def _append_iteration_losses(iter_log_path: Path, epoch: int, index: int, total_steps: int, errors) -> None:
    append_row(
        iter_log_path,
        [
            str(epoch),
            str(index + 1),
            str(total_steps),
            f"{float(errors.get('G_GAN', 0.0)):.8f}",
            f"{float(errors.get('G_L1', 0.0)):.8f}",
            f"{float(errors.get('G_Perc', 0.0)):.8f}",
            f"{float(errors.get('D_Real', 0.0)):.8f}",
            f"{float(errors.get('D_Fake', 0.0)):.8f}",
        ],
    )


def _append_epoch_losses(epoch_log_path: Path, epoch: int, total_steps: int, epoch_avg) -> None:
    append_row(
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


def run_training(opt, perceptual_model=None):
    checkpoint_dir = Path(opt.checkpoints_dir) / opt.name
    log_dir = checkpoint_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    iter_log_path = log_dir / "losses_by_iteration.txt"
    epoch_log_path = log_dir / "losses_by_epoch.txt"
    iter_state_path = checkpoint_dir / "training_state_latest.pth"

    ensure_header(iter_log_path, ["epoch", "iter", "total_steps"] + LOSS_KEYS)
    ensure_header(epoch_log_path, ["epoch", "total_steps"] + LOSS_KEYS)

    dataset = _build_dataset(opt)
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
        start_epoch, total_steps = try_resume_state(model, iter_state_path, opt.epoch_count)

    dataset_size = len(dataloader)
    print(f"Dataset size: {dataset_size} batches")

    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_loss_sum = {key: 0.0 for key in LOSS_KEYS}
        epoch_loss_count = 0

        for i, data in enumerate(dataloader):
            total_steps += opt.batchSize

            model.set_input(data)
            model.optimize_parameters()
            errors = model.get_current_errors()

            for key in LOSS_KEYS:
                epoch_loss_sum[key] += float(errors.get(key, 0.0))
            epoch_loss_count += 1

            _append_iteration_losses(iter_log_path, epoch, i, total_steps, errors)

            if total_steps % opt.print_freq < opt.batchSize:
                err_str = ", ".join([f"{k}: {v:.4f}" for k, v in errors.items()])
                print(f"[Epoch {epoch}][{i+1}/{dataset_size}] {err_str}")

            if total_steps % opt.save_latest_freq < opt.batchSize:
                print(f"Saving latest model (epoch {epoch}, total_steps {total_steps})")
                model.save("latest")
                save_training_state(
                    model=model,
                    state_path=iter_state_path,
                    epoch=epoch,
                    next_epoch=epoch,
                    total_steps=total_steps,
                )

        epoch_avg = {key: (epoch_loss_sum[key] / max(1, epoch_loss_count)) for key in LOSS_KEYS}
        _append_epoch_losses(epoch_log_path, epoch, total_steps, epoch_avg)
        save_loss_plots(log_dir, iter_log_path, epoch_log_path)

        if epoch % opt.save_epoch_freq == 0:
            print(f"Saving checkpoint for epoch {epoch}")
            model.save(epoch)
            save_training_state(
                model=model,
                state_path=checkpoint_dir / f"training_state_epoch_{epoch}.pth",
                epoch=epoch,
                next_epoch=epoch + 1,
                total_steps=total_steps,
            )

        save_training_state(
            model=model,
            state_path=iter_state_path,
            epoch=epoch,
            next_epoch=epoch + 1,
            total_steps=total_steps,
        )

        if epoch > opt.niter:
            model.update_learning_rate()
