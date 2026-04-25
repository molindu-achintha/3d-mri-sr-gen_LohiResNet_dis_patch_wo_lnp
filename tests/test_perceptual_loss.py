"""Tests for configurable perceptual-loss integration."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.perceptual_loss import (  # noqa: E402
    DinoV3FeatureExtractor,
    PerceptualLoss3D,
    SwinUNETRFeatureExtractor,
    build_perceptual_extractor,
)
from models.LohiResNet_dis_patch_wo_lnp import LohiResNet_dis_patch_wo_lnp  # noqa: E402


DINOV3_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "DINOV3"
    / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)


def _make_opt(tmp_path, use_perceptual=False, backbone="dinov3", lambda_gan=1.0):
    return SimpleNamespace(
        isTrain=True,
        gpu_ids=[],
        batchSize=1,
        input_nc=1,
        output_nc=1,
        depthSize=128,
        fineSize=128,
        scale_factor=1,
        ngf=1,
        ndf=2,
        which_model_netG="resunet_3d",
        which_model_netD="basic",
        norm="instance",
        no_dropout=True,
        gp_lambda=10.0,
        n_critic=1,
        wgan_gp=False,
        n_layers_D=3,
        lambda_gan=lambda_gan,
        continue_train=False,
        which_epoch="latest",
        pool_size=1,
        lr=2e-4,
        beta1=0.5,
        no_lsgan=False,
        lambda_A=10.0,
        which_direction="AtoB",
        checkpoints_dir=str(tmp_path / "checkpoints"),
        name="unit_test",
        niter_decay=10,
        use_perceptual_loss=use_perceptual,
        perceptual_backbone=backbone,
        lambda_perceptual=0.1,
        perceptual_distance="l2",
        perceptual_model_arch="",
        perceptual_model_ckpt="",
        perceptual_pretrained=False,
        perceptual_dinov3_repo="",
        perceptual_dino_input_size=32,
        perceptual_swin_feature_layers="encoder1,encoder2,encoder3,encoder4",
    )


class TinyPerceptual2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Recording2DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_shape = None

    def forward(self, x):
        self.last_shape = tuple(x.shape)
        return x.mean(dim=(2, 3))


class RecordingIntermediateDinoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_shape = None
        self.last_kwargs = None

    def get_intermediate_layers(self, x, *, n=1, reshape=False):
        self.last_shape = tuple(x.shape)
        self.last_kwargs = {"n": n, "reshape": reshape}
        return (torch.ones(x.size(0), 6, 2, 2),)

    def forward_features(self, _x):
        raise AssertionError("forward_features should not run when get_intermediate_layers is available")


class PatchTokenFallbackModel(nn.Module):
    patch_size = 16

    def __init__(self):
        super().__init__()
        self.last_shape = None

    def forward_features(self, x):
        self.last_shape = tuple(x.shape)
        batch = x.size(0)
        patch_tokens = torch.arange(batch * 4 * 8, dtype=x.dtype, device=x.device).view(batch, 4, 8)
        return {
            "x_norm_patchtokens": patch_tokens,
            "x_norm_clstoken": torch.zeros(batch, 8, dtype=x.dtype, device=x.device),
        }


class TinySwinLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv3d(4, 8, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.head = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.encoder1(x))
        x = torch.relu(self.encoder2(x))
        x = torch.relu(self.encoder3(x))
        x = torch.relu(self.encoder4(x))
        return self.head(x)


def test_perceptual_disabled_has_zero_component(tmp_path):
    opt = _make_opt(tmp_path, use_perceptual=False, lambda_gan=1.0)
    model = LohiResNet_dis_patch_wo_lnp()
    model.initialize(opt)
    data = {
        "A": torch.randn(1, 1, opt.depthSize, opt.fineSize, opt.fineSize),
        "B": torch.randn(1, 1, opt.depthSize, opt.fineSize, opt.fineSize),
    }
    model.set_input(data)
    model.optimize_parameters()

    assert model.loss_G_Perc.item() == pytest.approx(0.0, abs=1e-8)
    errors = model.get_current_errors()
    assert "G_Perc" in errors
    assert errors["G_Perc"] == pytest.approx(0.0, abs=1e-8)
    assert model.loss_G.item() == pytest.approx(
        ((opt.lambda_gan * model.loss_G_GAN) + model.loss_G_L1).item(), rel=1e-5
    )


def test_lambda_gan_zero_excludes_gan_term_from_total_loss(tmp_path):
    opt = _make_opt(tmp_path, use_perceptual=True, backbone="dinov3", lambda_gan=0.0)
    model = LohiResNet_dis_patch_wo_lnp()
    model.initialize(opt, perceptual_model=TinyPerceptual2D())
    data = {
        "A": torch.randn(1, 1, opt.depthSize, opt.fineSize, opt.fineSize),
        "B": torch.randn(1, 1, opt.depthSize, opt.fineSize, opt.fineSize),
    }
    model.set_input(data)
    model.forward()
    model.optimizer_G.zero_grad()
    model.backward_G()

    expected = (model.loss_G_L1 + model.loss_G_Perc).item()
    assert model.loss_G.item() == pytest.approx(expected, rel=1e-5)


def test_injected_perceptual_model_contributes_and_backprops(tmp_path):
    opt = _make_opt(tmp_path, use_perceptual=True, backbone="dinov3")
    model = LohiResNet_dis_patch_wo_lnp()
    model.initialize(opt, perceptual_model=TinyPerceptual2D())
    data = {
        "A": torch.randn(1, 1, opt.depthSize, opt.fineSize, opt.fineSize),
        "B": torch.randn(1, 1, opt.depthSize, opt.fineSize, opt.fineSize),
    }
    model.set_input(data)
    model.forward()
    model.optimizer_G.zero_grad()
    model.backward_G()

    assert model.loss_G_Perc.item() > 0.0
    grads = [p.grad for p in model.netG.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_pix2pix_optimizes_one_64_cube_patch(tmp_path):
    opt = _make_opt(tmp_path, use_perceptual=False)
    opt.depthSize = 64
    opt.fineSize = 64
    model = LohiResNet_dis_patch_wo_lnp()
    model.initialize(opt)
    data = {
        "A": torch.randn(1, 1, opt.depthSize, opt.fineSize, opt.fineSize),
        "B": torch.randn(1, 1, opt.depthSize, opt.fineSize, opt.fineSize),
    }
    model.set_input(data)
    model.optimize_parameters()

    assert tuple(model.fake_B.shape) == (1, 1, 64, 64, 64)
    assert model.loss_G_L1.item() > 0.0


def test_dino_slice_conversion_shape():
    recorder = Recording2DModel()
    extractor = DinoV3FeatureExtractor(recorder, input_size=32)
    x = torch.randn(2, 1, 5, 12, 16)
    features = extractor.extract_features(x)

    assert recorder.last_shape == (10, 3, 32, 32)
    assert features.shape[0] == 10


def test_dino_slice_conversion_shape_for_64_cube_patch():
    recorder = Recording2DModel()
    extractor = DinoV3FeatureExtractor(recorder, input_size=32)
    x = torch.randn(2, 1, 64, 64, 64)
    features = extractor.extract_features(x)

    assert recorder.last_shape == (128, 3, 32, 32)
    assert features.shape[0] == 128


def test_dino_prefers_intermediate_patch_maps():
    recorder = RecordingIntermediateDinoModel()
    extractor = DinoV3FeatureExtractor(recorder, input_size=32)
    x = torch.randn(2, 1, 5, 12, 16)
    features = extractor.extract_features(x)

    assert recorder.last_shape == (10, 3, 32, 32)
    assert recorder.last_kwargs == {"n": 1, "reshape": True}
    assert tuple(features.shape) == (10, 6, 2, 2)


def test_dino_forward_features_uses_patch_tokens_before_cls():
    recorder = PatchTokenFallbackModel()
    extractor = DinoV3FeatureExtractor(recorder, input_size=32)
    x = torch.randn(2, 1, 5, 12, 16)
    features = extractor.extract_features(x)

    assert recorder.last_shape == (10, 3, 32, 32)
    assert tuple(features.shape) == (10, 8, 2, 2)


def test_swin_multiscale_feature_aggregation():
    extractor = SwinUNETRFeatureExtractor(
        TinySwinLike(), ["encoder1", "encoder2", "encoder3", "encoder4"]
    )
    loss_fn = PerceptualLoss3D(extractor, distance="l2")
    fake = torch.randn(1, 1, 8, 16, 16)
    real = torch.randn(1, 1, 8, 16, 16)
    loss = loss_fn(fake, real)

    assert loss.dim() == 0
    assert loss.item() > 0.0


def test_missing_monai_dependency_raises_clear_error(monkeypatch):
    opt = SimpleNamespace(
        perceptual_backbone="swinunetr",
        perceptual_swin_feature_layers="encoder1,encoder2,encoder3,encoder4",
        perceptual_dino_input_size=224,
        perceptual_model_ckpt="",
        perceptual_pretrained=False,
        gpu_ids=[],
        scale_factor=1,
        depthSize=8,
        fineSize=8,
    )

    def _raise():
        raise RuntimeError("MONAI is required for SwinUNETR perceptual loss.")

    monkeypatch.setattr("models.perceptual_loss._get_monai_swinunetr_class", _raise)
    with pytest.raises(RuntimeError, match="MONAI"):
        build_perceptual_extractor(opt)


def test_invalid_perceptual_checkpoint_path_fails_fast():
    opt = SimpleNamespace(
        perceptual_backbone="dinov3",
        perceptual_swin_feature_layers="encoder1,encoder2,encoder3,encoder4",
        perceptual_dino_input_size=224,
        perceptual_model_ckpt="/tmp/definitely_missing_perceptual_ckpt.pth",
        perceptual_pretrained=False,
        perceptual_model_arch="vit_small_patch14_dinov2.lvd142m",
        gpu_ids=[],
    )
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        build_perceptual_extractor(opt, perceptual_model=TinyPerceptual2D())


def test_invalid_local_dinov3_repo_path_fails_fast(tmp_path):
    opt = SimpleNamespace(
        perceptual_backbone="dinov3",
        perceptual_model_arch="dinov3_vitb16",
        perceptual_swin_feature_layers="encoder1,encoder2,encoder3,encoder4",
        perceptual_dino_input_size=224,
        perceptual_model_ckpt="",
        perceptual_pretrained=False,
        perceptual_dinov3_repo=str(tmp_path / "missing_repo"),
        gpu_ids=[],
    )

    with pytest.raises(RuntimeError, match="requires a valid --perceptual_dinov3_repo"):
        build_perceptual_extractor(opt)


def test_local_dinov3_pretrained_requires_checkpoint(tmp_path):
    repo_path = tmp_path / "dinov3"
    (repo_path / "dinov3" / "hub").mkdir(parents=True)
    (repo_path / "dinov3" / "hub" / "backbones.py").write_text("# stub\n", encoding="utf-8")

    opt = SimpleNamespace(
        perceptual_backbone="dinov3",
        perceptual_model_arch="dinov3_vitb16",
        perceptual_swin_feature_layers="encoder1,encoder2,encoder3,encoder4",
        perceptual_dino_input_size=224,
        perceptual_model_ckpt="",
        perceptual_pretrained=True,
        perceptual_dinov3_repo=str(repo_path),
        gpu_ids=[],
    )

    with pytest.raises(RuntimeError, match="requires --perceptual_model_ckpt"):
        build_perceptual_extractor(opt)


def test_local_dinov3_repo_loader_bypasses_timm(tmp_path, monkeypatch):
    assert DINOV3_CHECKPOINT_PATH.exists()

    repo_path = tmp_path / "dinov3"
    (repo_path / "dinov3" / "hub").mkdir(parents=True)
    (repo_path / "dinov3" / "hub" / "backbones.py").write_text("# stub\n", encoding="utf-8")

    class FakeBackbones:
        @staticmethod
        def dinov3_vitb16(pretrained=False, weights=None):
            model = TinyPerceptual2D()
            model.factory_args = {"pretrained": pretrained, "weights": weights}
            return model

    monkeypatch.setattr("models.perceptual_loss._import_local_dinov3_backbones", lambda _repo: FakeBackbones)
    monkeypatch.setattr(
        "models.perceptual_loss._get_timm_module",
        lambda: (_ for _ in ()).throw(AssertionError("timm should not be used")),
    )
    loaded_checkpoints = []
    monkeypatch.setattr(
        "models.perceptual_loss._load_weights",
        lambda _model, ckpt_path: loaded_checkpoints.append(str(ckpt_path)),
    )

    opt = SimpleNamespace(
        perceptual_backbone="dinov3",
        perceptual_model_arch="dinov3_vitb16",
        perceptual_swin_feature_layers="encoder1,encoder2,encoder3,encoder4",
        perceptual_dino_input_size=224,
        perceptual_model_ckpt=str(DINOV3_CHECKPOINT_PATH),
        perceptual_pretrained=True,
        perceptual_dinov3_repo=str(repo_path),
        gpu_ids=[],
    )

    extractor = build_perceptual_extractor(opt)

    assert isinstance(extractor, DinoV3FeatureExtractor)
    assert isinstance(extractor.model, TinyPerceptual2D)
    assert extractor.model.factory_args == {
        "pretrained": True,
        "weights": str(DINOV3_CHECKPOINT_PATH),
    }
    assert loaded_checkpoints == [str(DINOV3_CHECKPOINT_PATH)]


def test_unknown_local_dinov3_arch_raises_clear_error(tmp_path, monkeypatch):
    repo_path = tmp_path / "dinov3"
    (repo_path / "dinov3" / "hub").mkdir(parents=True)
    (repo_path / "dinov3" / "hub" / "backbones.py").write_text("# stub\n", encoding="utf-8")

    class FakeBackbones:
        @staticmethod
        def dinov3_vitb16(pretrained=False, weights=None):
            return TinyPerceptual2D()

    monkeypatch.setattr("models.perceptual_loss._import_local_dinov3_backbones", lambda _repo: FakeBackbones)

    opt = SimpleNamespace(
        perceptual_backbone="dinov3",
        perceptual_model_arch="dinov3_missing",
        perceptual_swin_feature_layers="encoder1,encoder2,encoder3,encoder4",
        perceptual_dino_input_size=224,
        perceptual_model_ckpt="",
        perceptual_pretrained=False,
        perceptual_dinov3_repo=str(repo_path),
        gpu_ids=[],
    )

    with pytest.raises(RuntimeError, match="Unknown local DINOv3 architecture 'dinov3_missing'"):
        build_perceptual_extractor(opt)


def test_invalid_dino_arch_raises_with_version_hint(monkeypatch):
    class FakeTimm:
        __version__ = "0.9.2"

        @staticmethod
        def list_models():
            return [
                "vit_small_patch14_dinov2.lvd142m",
                "vit_base_patch14_dinov2.lvd142m",
            ]

        @staticmethod
        def create_model(*_args, **_kwargs):
            raise AssertionError("create_model should not be called for invalid arch")

    opt = SimpleNamespace(
        perceptual_backbone="dinov3",
        perceptual_model_arch="this_model_does_not_exist",
        perceptual_swin_feature_layers="encoder1,encoder2,encoder3,encoder4",
        perceptual_dino_input_size=224,
        perceptual_model_ckpt="",
        perceptual_pretrained=False,
        perceptual_dinov3_repo="",
        gpu_ids=[],
    )

    monkeypatch.setattr("models.perceptual_loss._get_timm_module", lambda: FakeTimm())
    with pytest.raises(RuntimeError, match="Unknown DINO architecture 'this_model_does_not_exist'"):
        build_perceptual_extractor(opt)
