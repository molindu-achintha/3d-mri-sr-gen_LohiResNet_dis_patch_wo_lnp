"""Unit tests for the active 128^3 ResUNet generator."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.networks3d import ResUNetGenerator3D, define_G  # noqa: E402


def test_resunet3d_shape_128():
    gen = ResUNetGenerator3D(in_channels=1, out_channels=1, base_filters=1)
    x = torch.randn(1, 1, 128, 128, 128)

    with torch.no_grad():
        y = gen(x)

    assert y.shape == (1, 1, 128, 128, 128)


def test_resunet3d_multichannel():
    gen = ResUNetGenerator3D(in_channels=3, out_channels=2, base_filters=1)
    x = torch.randn(1, 3, 128, 128, 128)

    with torch.no_grad():
        y = gen(x)

    assert y.shape == (1, 2, 128, 128, 128)


def test_resunet3d_gradient_flow():
    gen = ResUNetGenerator3D(in_channels=1, out_channels=1, base_filters=1)
    x = torch.randn(1, 1, 128, 128, 128)

    y = gen(x)
    y.mean().backward()

    no_grad = [name for name, p in gen.named_parameters() if p.requires_grad and p.grad is None]
    assert no_grad == []


def test_define_g_returns_resunet_for_legacy_name():
    gen = define_G(
        input_nc=1,
        output_nc=1,
        ngf=1,
        which_model_netG="mdrn_3d",
        norm="instance",
        use_dropout=False,
        gpu_ids=[],
        scale_factor=4,
        n_fmdrb=2,
        skip_compress_ratio=0.5,
    )
    x = torch.randn(1, 1, 128, 128, 128)

    assert isinstance(gen, ResUNetGenerator3D)
    with torch.no_grad():
        assert gen(x).shape == x.shape


if __name__ == "__main__":
    test_resunet3d_shape_128()
    test_resunet3d_multichannel()
    test_resunet3d_gradient_flow()
    test_define_g_returns_resunet_for_legacy_name()
    print("All ResUNet3D generator tests passed.")
