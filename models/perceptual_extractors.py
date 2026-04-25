from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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
