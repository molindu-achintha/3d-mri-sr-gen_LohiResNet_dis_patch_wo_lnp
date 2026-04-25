"""Tests for research profile defaults and overrides."""

import sys

from options.train_options import TrainOptions


def _parse_train_options(monkeypatch, tmp_path, args):
    argv = [
        "train.py",
        "--dataroot",
        str(tmp_path),
        "--name",
        "research_profile_test",
        "--checkpoints_dir",
        str(tmp_path / "checkpoints"),
        "--device",
        "cpu",
    ] + args
    monkeypatch.setattr(sys, "argv", argv)
    return TrainOptions().parse()


def test_balanced_research_profile_sets_locked_defaults(monkeypatch, tmp_path):
    opt = _parse_train_options(
        monkeypatch,
        tmp_path,
        ["--research_profile", "balanced_mri_sr_v1"],
    )

    assert opt.which_model_netG == "resunet_3d"
    assert opt.ngf == 48
    assert opt.which_model_netD == "n_layers"
    assert opt.ndf == 48
    assert opt.n_layers_D == 3
    assert opt.wgan_gp is True
    assert opt.gp_lambda == 10.0
    assert opt.n_critic == 5
    assert opt.lambda_A == 100.0
    assert opt.lambda_gan == 0.001
    assert opt.use_perceptual_loss is True
    assert opt.perceptual_backbone == "dinov3"
    assert opt.lambda_perceptual == 0.01
    assert opt.perceptual_distance == "l2"
    assert opt.scale_factor == 1


def test_explicit_cli_overrides_research_profile_defaults(monkeypatch, tmp_path):
    opt = _parse_train_options(
        monkeypatch,
        tmp_path,
        [
            "--research_profile",
            "balanced_mri_sr_v1",
            "--ngf",
            "64",
            "--lambda_perceptual",
            "0.2",
        ],
    )

    assert opt.research_profile == "balanced_mri_sr_v1"
    assert opt.which_model_netG == "resunet_3d"
    assert opt.ngf == 64
    assert opt.lambda_perceptual == 0.2


def test_legacy_generator_and_scale_are_forced_to_resunet(monkeypatch, tmp_path):
    opt = _parse_train_options(
        monkeypatch,
        tmp_path,
        [
            "--which_model_netG",
            "mdrn_3d",
            "--scale_factor",
            "4",
        ],
    )

    assert opt.which_model_netG == "resunet_3d"
    assert opt.scale_factor == 1
