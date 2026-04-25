from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import List, Sequence

import torch
import torch.nn as nn


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
