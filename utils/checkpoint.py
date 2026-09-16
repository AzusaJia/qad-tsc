from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def checkpoint_experiment_name(
    loss_lambda: float,
    seed: int,
) -> str:
    return (
        f"quantile_pointwise_freq_mask_all_patch_seed{seed}"
        f"_lambda{loss_lambda:g}"
    )


def checkpoint_directory(
    root: str | Path,
    loss_lambda: float,
    seed: int,
    dataset_name: str,
) -> Path:
    return Path(root) / checkpoint_experiment_name(loss_lambda, seed) / dataset_name


def find_checkpoint_path(
    root: str | Path,
    loss_lambda: float,
    seed: int,
    dataset_name: str,
    filename: str = "best.pt",
) -> Path:
    path = checkpoint_directory(root, loss_lambda, seed, dataset_name) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def save_training_state(
    path: str | Path,
    student: nn.Module,
    teacher: nn.Module,
    student_projector: nn.Module,
    teacher_projector: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_state_dict": student.state_dict(),
            "teacher_state_dict": teacher.state_dict(),
            "student_ppv_projector_state_dict": student_projector.state_dict(),
            "teacher_ppv_projector_state_dict": teacher_projector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            **metadata,
        },
        path,
    )


def read_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> dict:
    return torch.load(Path(path), map_location=device, weights_only=False)


def load_training_state(
    path: str | Path,
    student: nn.Module,
    teacher: nn.Module,
    student_projector: nn.Module,
    teacher_projector: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    restore_rng: bool = False,
) -> dict:
    payload = read_checkpoint(path, device)
    student.load_state_dict(payload["student_state_dict"])
    teacher.load_state_dict(payload["teacher_state_dict"])
    student_projector.load_state_dict(payload["student_ppv_projector_state_dict"])
    teacher_projector.load_state_dict(payload["teacher_ppv_projector_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    rng_state = payload.get("rng_state")
    if restore_rng and rng_state:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].cpu())
        cuda_state = rng_state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
    return payload
