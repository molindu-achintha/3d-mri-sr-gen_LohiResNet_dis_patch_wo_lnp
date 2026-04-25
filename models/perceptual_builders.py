from __future__ import annotations

import inspect
from pathlib import Path

import torch.nn as nn

from .perceptual_checkpoints import _load_weights
from .perceptual_dependencies import (
    _freeze_module,
    _get_monai_swinunetr_class,
    _get_timm_module,
    _import_local_dinov3_backbones,
    _parse_feature_layers,
    _resolve_device,
    _resolve_local_dinov3_repo,
    _suggest_dino_arches,
    _suggest_local_dino_arches,
    _validate_dino_architecture,
)
from .perceptual_extractors import DinoV3FeatureExtractor, SwinUNETRFeatureExtractor


def _build_swinunetr_from_config(opt) -> nn.Module:
    SwinUNETR = _get_monai_swinunetr_class()
    signature = inspect.signature(SwinUNETR.__init__)
    params = signature.parameters

    scale = int(getattr(opt, "scale_factor", 1))
    depth = int(getattr(opt, "depthSize", 96)) * scale
    size = int(getattr(opt, "fineSize", 96)) * scale

    kwargs = {
        "in_channels": 1,
        "out_channels": 1,
    }
    if "img_size" in params:
        kwargs["img_size"] = (depth, size, size)
    if "feature_size" in params:
        kwargs["feature_size"] = 48
    if "spatial_dims" in params:
        kwargs["spatial_dims"] = 3
    if "use_checkpoint" in params:
        kwargs["use_checkpoint"] = False

    try:
        return SwinUNETR(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize SwinUNETR for perceptual loss with kwargs={kwargs}."
        ) from exc


def _build_timm_dino_from_config(opt, arch: str) -> nn.Module:
    timm = _get_timm_module()
    _validate_dino_architecture(timm, arch)
    use_pretrained = bool(getattr(opt, "perceptual_pretrained", True)) and not getattr(
        opt, "perceptual_model_ckpt", ""
    )
    try:
        return timm.create_model(arch, pretrained=use_pretrained, num_classes=0)
    except Exception as exc:
        version = getattr(timm, "__version__", "unknown")
        suggestions = ", ".join(_suggest_dino_arches(timm))
        raise RuntimeError(
            f"Failed to create DINO backbone '{arch}' with timm=={version}. "
            f"Try --no_perceptual_pretrained, provide --perceptual_model_ckpt, or use one of: {suggestions}."
        ) from exc


def _build_local_dinov3_from_config(opt, repo_path: Path, arch: str) -> nn.Module:
    repo_path = repo_path.expanduser()
    if not repo_path.exists():
        raise RuntimeError(
            f"Local DINOv3 architecture '{arch}' requires a valid --perceptual_dinov3_repo. "
            f"Missing path: {repo_path}"
        )
    expected_backbones = repo_path / "dinov3" / "hub" / "backbones.py"
    if not expected_backbones.exists():
        raise RuntimeError(
            f"--perceptual_dinov3_repo={repo_path} does not look like a DINOv3 checkout "
            f"(missing {expected_backbones})."
        )

    ckpt_path = getattr(opt, "perceptual_model_ckpt", "")
    use_pretrained = bool(getattr(opt, "perceptual_pretrained", True))
    if not ckpt_path and use_pretrained:
        raise RuntimeError(
            "Local DINOv3 loading requires --perceptual_model_ckpt when pretrained weights are requested."
        )

    backbones_module = _import_local_dinov3_backbones(repo_path)
    factory = getattr(backbones_module, arch, None)
    if not callable(factory):
        suggestions = ", ".join(_suggest_local_dino_arches(backbones_module)[:6])
        raise RuntimeError(
            f"Unknown local DINOv3 architecture '{arch}'. "
            f"Try one of: {suggestions}."
        )

    kwargs = {"pretrained": False}
    if ckpt_path:
        ckpt = Path(ckpt_path).expanduser()
        if not ckpt.exists():
            raise FileNotFoundError(f"Perceptual checkpoint not found: {ckpt}")
        kwargs = {"pretrained": True, "weights": str(ckpt)}

    try:
        return factory(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create local DINOv3 backbone '{arch}' from repo {repo_path}."
        ) from exc


def _build_dinov3_from_config(opt) -> nn.Module:
    arch = getattr(opt, "perceptual_model_arch", "") or "dinov3_vitb16"
    repo_path = _resolve_local_dinov3_repo(opt)
    if arch.startswith("dinov3_"):
        if repo_path is None:
            raise RuntimeError(
                f"Local DINOv3 architecture '{arch}' requires --perceptual_dinov3_repo "
                "or a sibling ../dinov3 checkout."
            )
        return _build_local_dinov3_from_config(opt, repo_path, arch)
    return _build_timm_dino_from_config(opt, arch)


def build_perceptual_extractor(opt, perceptual_model: nn.Module = None, device=None):
    """Build a frozen feature extractor with a unified extract_features API."""

    backbone = getattr(opt, "perceptual_backbone", "swinunetr").lower()
    gpu_ids = getattr(opt, "gpu_ids", [])
    resolved_device = _resolve_device(device, gpu_ids)
    feature_layers = _parse_feature_layers(
        getattr(opt, "perceptual_swin_feature_layers", "encoder1,encoder2,encoder3,encoder4"),
        defaults=("encoder1", "encoder2", "encoder3", "encoder4"),
    )
    dino_input_size = int(getattr(opt, "perceptual_dino_input_size", 224))
    ckpt_path = getattr(opt, "perceptual_model_ckpt", "")
    use_pretrained = bool(getattr(opt, "perceptual_pretrained", True))

    if perceptual_model is not None and hasattr(perceptual_model, "extract_features"):
        extractor = perceptual_model
    elif backbone == "swinunetr":
        model = perceptual_model if perceptual_model is not None else _build_swinunetr_from_config(opt)
        if ckpt_path:
            _load_weights(model, ckpt_path)
        elif perceptual_model is None and use_pretrained:
            raise RuntimeError(
                "Automatic pretrained SwinUNETR weights are unavailable in this setup. "
                "Provide --perceptual_model_ckpt or inject a preloaded perceptual model."
            )
        extractor = SwinUNETRFeatureExtractor(model, layer_names=feature_layers)
    elif backbone == "dinov3":
        model = perceptual_model if perceptual_model is not None else _build_dinov3_from_config(opt)
        if ckpt_path:
            _load_weights(model, ckpt_path)
        extractor = DinoV3FeatureExtractor(model, input_size=dino_input_size)
    else:
        raise ValueError(
            f"Unknown perceptual backbone '{backbone}'. Choose from 'swinunetr' or 'dinov3'."
        )

    if isinstance(extractor, nn.Module):
        extractor = extractor.to(resolved_device)
        _freeze_module(extractor)
    return extractor
