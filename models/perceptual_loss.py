"""Perceptual-loss facade for 3D super-resolution training.

The implementation is split by concern across:
- perceptual_dependencies.py
- perceptual_checkpoints.py
- perceptual_extractors.py
- perceptual_builders.py

This module keeps the public imports stable for callers and tests.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import torch
import torch.nn as nn

from .perceptual_builders import (
    _build_dinov3_from_config,
    _build_local_dinov3_from_config,
    _build_swinunetr_from_config,
    _build_timm_dino_from_config,
    build_perceptual_extractor,
)
from .perceptual_checkpoints import _clean_state_dict_keys, _extract_state_dict, _load_weights
from .perceptual_dependencies import (
    _default_dinov3_repo_path,
    _dino_fallback_arches,
    _freeze_module,
    _get_monai_swinunetr_class,
    _get_timm_module,
    _import_local_dinov3_backbones,
    _module_belongs_to_repo,
    _parse_feature_layers,
    _resolve_device,
    _resolve_local_dinov3_repo,
    _suggest_dino_arches,
    _suggest_local_dino_arches,
    _validate_dino_architecture,
)
from .perceptual_extractors import DinoV3FeatureExtractor, SwinUNETRFeatureExtractor


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


__all__ = [
    "DinoV3FeatureExtractor",
    "PerceptualLoss3D",
    "SwinUNETRFeatureExtractor",
    "build_perceptual_extractor",
    "_build_dinov3_from_config",
    "_build_local_dinov3_from_config",
    "_build_swinunetr_from_config",
    "_build_timm_dino_from_config",
    "_clean_state_dict_keys",
    "_default_dinov3_repo_path",
    "_dino_fallback_arches",
    "_extract_state_dict",
    "_freeze_module",
    "_get_monai_swinunetr_class",
    "_get_timm_module",
    "_import_local_dinov3_backbones",
    "_load_weights",
    "_module_belongs_to_repo",
    "_parse_feature_layers",
    "_resolve_device",
    "_resolve_local_dinov3_repo",
    "_suggest_dino_arches",
    "_suggest_local_dino_arches",
    "_validate_dino_architecture",
]
