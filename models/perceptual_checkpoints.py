from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


def _extract_state_dict(blob):
    if isinstance(blob, dict):
        for key in ("state_dict", "model_state_dict", "model", "network"):
            if key in blob and isinstance(blob[key], dict):
                return blob[key]
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
