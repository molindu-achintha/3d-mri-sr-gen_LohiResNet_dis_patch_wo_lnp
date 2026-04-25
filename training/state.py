from pathlib import Path
from typing import Tuple

import torch


def save_training_state(model, state_path: Path, epoch: int, next_epoch: int, total_steps: int) -> None:
    state = {
        "epoch": epoch,
        "next_epoch": next_epoch,
        "total_steps": total_steps,
    }
    if hasattr(model, "old_lr"):
        state["old_lr"] = model.old_lr
    if hasattr(model, "optimizer_G"):
        state["optimizer_G"] = model.optimizer_G.state_dict()
    if hasattr(model, "optimizer_D"):
        state["optimizer_D"] = model.optimizer_D.state_dict()
    torch.save(state, state_path)


def try_resume_state(model, state_path: Path, default_epoch: int) -> Tuple[int, int]:
    if not state_path.exists():
        return default_epoch, 0

    state = torch.load(state_path, map_location="cpu")
    if hasattr(model, "optimizer_G") and "optimizer_G" in state:
        model.optimizer_G.load_state_dict(state["optimizer_G"])
    if hasattr(model, "optimizer_D") and "optimizer_D" in state:
        model.optimizer_D.load_state_dict(state["optimizer_D"])
    if hasattr(model, "old_lr") and "old_lr" in state:
        model.old_lr = state["old_lr"]

    start_epoch = int(state.get("next_epoch", default_epoch))
    total_steps = int(state.get("total_steps", 0))
    print(f"Resumed training state from {state_path} (start_epoch={start_epoch}, total_steps={total_steps})")
    return start_epoch, total_steps
