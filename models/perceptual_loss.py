"""Perceptual-loss helpers for 3D super-resolution training.

Supports two backbones:
1) SwinUNETR (3D) with multi-scale encoder feature taps
2) DINO-like ViTs (2D) applied slice-wise across 3D depth
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_monai_swinunetr_class():
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as exc:
        raise RuntimeError(
            "MONAI is required for SwinUNETR perceptual loss. Install with `pip install monai`."
        ) from exc
    return SwinUNETR


def _get_timm_module():
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "timm is required for DINO perceptual loss. Install or upgrade with `pip install -U timm`."
        ) from exc
    return timm


def _default_dinov3_repo_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "dinov3"
    return str(candidate) if candidate.exists() else ""


def _resolve_device(device, gpu_ids: Sequence[int]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if gpu_ids and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cpu")


def _freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def _parse_feature_layers(raw_layers: str, defaults: Sequence[str]) -> List[str]:
    if not raw_layers:
        return list(defaults)
    layers = [name.strip() for name in raw_layers.split(",") if name.strip()]
    return layers or list(defaults)


def _extract_state_dict(blob):
    if isinstance(blob, dict):
        for key in ("state_dict", "model_state_dict", "model", "network"):
            if key in blob and isinstance(blob[key], dict):
                return blob[key]
        # If this already looks like a state dict, return as-is.
        if blob and all(torch.is_tensor(v) for v in blob.values()):
            return blob
    raise RuntimeError("Unsupported checkpoint format. Expected a state_dict-like object.")


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _load_weights(model: nn.Module, ckpt_path: str) -> None:
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Perceptual checkpoint not found: {ckpt}")
    payload = torch.load(ckpt, map_location="cpu")
    state_dict = _clean_state_dict_keys(_extract_state_dict(payload))
    model.load_state_dict(state_dict, strict=False)


def _dino_fallback_arches() -> List[str]:
    return [
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m",
    ]


def _suggest_dino_arches(timm_module) -> List[str]:
    catalog = sorted([name for name in timm_module.list_models() if "dino" in name.lower()])
    if not catalog:
        return _dino_fallback_arches()
    preferred = [name for name in _dino_fallback_arches() if name in catalog]
    if preferred:
        return preferred
    return catalog[:5]


def _validate_dino_architecture(timm_module, arch: str) -> None:
    catalog = set(timm_module.list_models())
    if arch in catalog:
        return

    suggestions = _suggest_dino_arches(timm_module)
    version = getattr(timm_module, "__version__", "unknown")
    suggestion_text = ", ".join(suggestions)
    raise RuntimeError(
        f"Unknown DINO architecture '{arch}' for timm=={version}. "
        f"Try one of: {suggestion_text}. If this model should exist, upgrade timm via `pip install -U timm`."
    )


def _resolve_local_dinov3_repo(opt) -> Path | None:
    raw_repo = getattr(opt, "perceptual_dinov3_repo", "") or _default_dinov3_repo_path()
    if not raw_repo:
        return None
    return Path(raw_repo).expanduser()


def _module_belongs_to_repo(module: ModuleType, repo_path: Path) -> bool:
    module_file = getattr(module, "__file__", "")
    if not module_file:
        return False
    try:
        resolved_file = Path(module_file).resolve()
    except OSError:
        return False
    return repo_path.resolve() in resolved_file.parents


def _import_local_dinov3_backbones(repo_path: Path) -> ModuleType:
    repo_path = repo_path.resolve()
    existing_pkg = sys.modules.get("dinov3")
    if existing_pkg is not None and not _module_belongs_to_repo(existing_pkg, repo_path):
        raise RuntimeError(
            "A different 'dinov3' package is already imported from "
            f"{getattr(existing_pkg, '__file__', 'unknown')}. Restart with only "
            f"--perceptual_dinov3_repo={repo_path} available on PYTHONPATH."
        )

    sys.path.insert(0, str(repo_path))
    try:
        module = importlib.import_module("dinov3.hub.backbones")
    finally:
        try:
            sys.path.remove(str(repo_path))
        except ValueError:
            pass

    if not _module_belongs_to_repo(module, repo_path):
        raise RuntimeError(
            f"Imported dinov3.hub.backbones from {getattr(module, '__file__', 'unknown')} "
            f"instead of {repo_path}."
        )
    return module


def _suggest_local_dino_arches(backbones_module: ModuleType) -> List[str]:
    names = []
    for name in dir(backbones_module):
        if name.startswith("dinov3_") and callable(getattr(backbones_module, name)):
            names.append(name)
    return sorted(names)


class SwinUNETRFeatureExtractor(nn.Module):
    """Collect multi-scale features from configured module names via hooks."""

    def __init__(self, model: nn.Module, layer_names: Sequence[str]):
        super().__init__()
        self.model = model
        self.layer_names = list(layer_names)
        self._hooks = []
        self._features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        named_modules = dict(self.model.named_modules())
        missing = [name for name in self.layer_names if name not in named_modules]
        if missing:
            available = ", ".join(sorted(list(named_modules.keys()))[:20])
            raise ValueError(
                f"Swin feature layers not found: {missing}. "
                f"Available examples: {available}"
            )

        def _make_hook(name):
            def _hook(_module, _inputs, outputs):
                tensor = self._to_tensor(outputs)
                self._features[name] = tensor

            return _hook

        for name in self.layer_names:
            handle = named_modules[name].register_forward_hook(_make_hook(name))
            self._hooks.append(handle)

    @staticmethod
    def _to_tensor(outputs) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return outputs
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if torch.is_tensor(item):
                    return item
        if isinstance(outputs, dict):
            for item in outputs.values():
                if torch.is_tensor(item):
                    return item
        raise TypeError("Hook output does not contain a tensor.")

    @staticmethod
    def _adapt_channels(x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            return x
        return x.mean(dim=1, keepdim=True)

    def extract_features(self, x_3d: torch.Tensor) -> List[torch.Tensor]:
        if x_3d.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B,C,D,H,W], got shape {tuple(x_3d.shape)}")
        x_3d = self._adapt_channels(x_3d)
        self._features = {}
        _ = self.model(x_3d)
        missing = [name for name in self.layer_names if name not in self._features]
        if missing:
            raise RuntimeError(f"Missing hooked features for layers: {missing}")
        return [self._features[name] for name in self.layer_names]


class DinoV3FeatureExtractor(nn.Module):
    """Slice-wise 3D-to-2D feature extractor using a DINO-like backbone."""

    def __init__(self, model: nn.Module, input_size: int = 224):
        super().__init__()
        self.model = model
        self.input_size = int(input_size)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _volume_to_slices(x_3d: torch.Tensor) -> torch.Tensor:
        # [B, C, D, H, W] -> [B*D, C, H, W]
        batch, ch, depth, height, width = x_3d.shape
        return x_3d.permute(0, 2, 1, 3, 4).contiguous().view(batch * depth, ch, height, width)

    @staticmethod
    def _adapt_to_three_channels(x_2d: torch.Tensor) -> torch.Tensor:
        channels = x_2d.size(1)
        if channels == 3:
            return x_2d
        if channels == 1:
            return x_2d.repeat(1, 3, 1, 1)
        if channels == 2:
            return torch.cat([x_2d, x_2d[:, :1]], dim=1)
        return x_2d[:, :3]

    def _prepare_slices(self, x_3d: torch.Tensor) -> torch.Tensor:
        x_2d = self._volume_to_slices(x_3d)
        x_2d = self._adapt_to_three_channels(x_2d)
        x_2d = (x_2d + 1.0) * 0.5
        x_2d = F.interpolate(
            x_2d,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return (x_2d - self.imagenet_mean) / self.imagenet_std

    def _reshape_patch_tokens(self, features: torch.Tensor, image_shape: torch.Size) -> torch.Tensor:
        if features.dim() != 3:
            return features

        patch_size = getattr(self.model, "patch_size", None)
        if isinstance(patch_size, tuple):
            if len(patch_size) < 2:
                return features
            patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
        elif isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            return features

        if patch_h <= 0 or patch_w <= 0:
            return features

        grid_h = int(image_shape[-2]) // patch_h
        grid_w = int(image_shape[-1]) // patch_w
        if grid_h <= 0 or grid_w <= 0 or (grid_h * grid_w) != features.size(1):
            return features

        return (
            features.transpose(1, 2)
            .contiguous()
            .view(features.size(0), features.size(2), grid_h, grid_w)
        )

    def _pick_tensor(self, output, image_shape: torch.Size) -> torch.Tensor:
        if torch.is_tensor(output):
            return self._reshape_patch_tokens(output, image_shape)
        if isinstance(output, (list, tuple)):
            for item in output:
                if torch.is_tensor(item):
                    return self._reshape_patch_tokens(item, image_shape)
                if isinstance(item, (list, tuple, dict)):
                    try:
                        return self._pick_tensor(item, image_shape)
                    except TypeError:
                        continue
        if isinstance(output, dict):
            # Prefer patch/token maps before global embeddings for spatial perceptual loss.
            for key in (
                "x_norm_patchtokens",
                "patch_tokens",
                "features",
                "last_hidden_state",
                "x_norm_clstoken",
                "x_cls",
            ):
                if key in output and torch.is_tensor(output[key]):
                    return self._reshape_patch_tokens(output[key], image_shape)
            for item in output.values():
                if torch.is_tensor(item):
                    return self._reshape_patch_tokens(item, image_shape)
        raise TypeError("DINO forward output does not contain a tensor.")

    def extract_features(self, x_3d: torch.Tensor) -> torch.Tensor:
        if x_3d.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B,C,D,H,W], got shape {tuple(x_3d.shape)}")
        x_2d = self._prepare_slices(x_3d)
        if hasattr(self.model, "get_intermediate_layers"):
            try:
                intermediate = self.model.get_intermediate_layers(x_2d, n=1, reshape=True)
            except TypeError:
                intermediate = None
            else:
                if isinstance(intermediate, (list, tuple)) and intermediate:
                    first_item = intermediate[0]
                    if torch.is_tensor(first_item):
                        return first_item
                    if isinstance(first_item, (list, tuple)):
                        for item in first_item:
                            if torch.is_tensor(item):
                                return item
        if hasattr(self.model, "forward_features"):
            raw = self.model.forward_features(x_2d)
        else:
            raw = self.model(x_2d)
        return self._pick_tensor(raw, x_2d.shape)


class PerceptualLoss3D(nn.Module):
    """Compute perceptual feature distance between fake and real 3D volumes."""

    def __init__(self, extractor: Union[SwinUNETRFeatureExtractor, DinoV3FeatureExtractor], distance: str = "l2"):
        super().__init__()
        if distance not in {"l1", "l2"}:
            raise ValueError(f"Unsupported perceptual distance: {distance}")
        self.extractor = extractor
        self.distance = distance

    @staticmethod
    def _as_list(features: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
        if torch.is_tensor(features):
            return [features]
        return list(features)

    def _distance(self, fake_feat: torch.Tensor, real_feat: torch.Tensor) -> torch.Tensor:
        if self.distance == "l1":
            return torch.mean(torch.abs(fake_feat - real_feat))
        diff = fake_feat - real_feat
        return torch.mean(diff * diff)

    def forward(self, fake_b: torch.Tensor, real_b: torch.Tensor) -> torch.Tensor:
        fake_features = self._as_list(self.extractor.extract_features(fake_b))
        with torch.no_grad():
            real_features = self._as_list(self.extractor.extract_features(real_b))
        if len(fake_features) != len(real_features):
            raise RuntimeError(
                f"Perceptual feature mismatch: {len(fake_features)} fake vs {len(real_features)} real"
            )
        losses = [self._distance(fake_f, real_f) for fake_f, real_f in zip(fake_features, real_features)]
        return torch.stack(losses).mean()


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
