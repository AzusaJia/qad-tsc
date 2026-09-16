from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from losses.quantile_loss import patch_alignment_loss
from models.teacher import EMA_MOMENTUM, update_ema_teacher
from utils.checkpoint import load_training_state, read_checkpoint, save_training_state


WEIGHT_DECAY = 0.0001


class SSLTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        student_projector: nn.Module,
        teacher_projector: nn.Module,
        device: torch.device,
        learning_rate: float,
        loss_lambda: float,
    ) -> None:
        self.student = student
        self.teacher = teacher
        self.student_projector = student_projector
        self.teacher_projector = teacher_projector
        self.device = device
        self.loss_lambda = loss_lambda
        parameters = list(student.parameters()) + list(student_projector.parameters())
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.student.train()
        self.teacher.eval()
        self.student_projector.train()
        self.teacher_projector.eval()
        totals = {"loss": 0.0, "quantile_loss": 0.0, "pointwise_loss": 0.0}
        batches = 0
        for clean_values, padding_mask in loader:
            clean_values = clean_values.to(self.device)
            padding_mask = padding_mask.to(self.device)
            loss, details = patch_alignment_loss(
                self.student,
                self.teacher,
                self.student_projector,
                self.teacher_projector,
                clean_values,
                padding_mask,
                self.loss_lambda,
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            update_ema_teacher(self.student, self.teacher, EMA_MOMENTUM)
            update_ema_teacher(
                self.student_projector, self.teacher_projector, EMA_MOMENTUM
            )
            totals["loss"] += float(loss.detach().item())
            totals["quantile_loss"] += details["quantile_loss"]
            totals["pointwise_loss"] += details["pointwise_loss"]
            batches += 1
        return {key: value / batches for key, value in totals.items()}

    def fit(
        self,
        loader: torch.utils.data.DataLoader,
        epochs: int,
        checkpoint_directory: str | Path,
        metadata: dict,
        resume_checkpoint: str | Path | None = None,
    ) -> dict[str, float | int | str]:
        checkpoint_directory = Path(checkpoint_directory)
        best_loss = float("inf")
        best_epoch = 0
        start_epoch = 1
        last_stats: dict[str, float] = {}
        logger = logging.getLogger("qad")

        if resume_checkpoint is not None:
            resumed = load_training_state(
                resume_checkpoint,
                self.student,
                self.teacher,
                self.student_projector,
                self.teacher_projector,
                self.optimizer,
                self.device,
                restore_rng=True,
            )
            completed_epoch = int(resumed.get("epoch", 0))
            start_epoch = completed_epoch + 1
            best_path = checkpoint_directory / "best.pt"
            if best_path.is_file():
                previous_best = read_checkpoint(best_path, "cpu")
                best_loss = float(previous_best.get("best_loss", float("inf")))
                best_epoch = int(
                    previous_best.get("best_epoch", previous_best.get("epoch", 0))
                )
            else:
                best_loss = float(resumed.get("best_loss", float("inf")))
                best_epoch = int(resumed.get("best_epoch", resumed.get("epoch", 0)))
            logger.info(
                "action=resume_training checkpoint=%s completed_epoch=%d "
                "next_epoch=%d target_epoch=%d",
                resume_checkpoint,
                completed_epoch,
                start_epoch,
                epochs,
            )

        for epoch in range(start_epoch, epochs + 1):
            stats = self.train_epoch(loader)
            last_stats = stats
            logger.info(
                "epoch=%d/%d loss=%.6f quantile=%.6f pointwise=%.6f",
                epoch,
                epochs,
                stats["loss"],
                stats["quantile_loss"],
                stats["pointwise_loss"],
            )
            is_best = stats["loss"] < best_loss
            if is_best:
                best_loss = stats["loss"]
                best_epoch = epoch
            payload = {
                **metadata,
                "epoch": epoch,
                "train_loss": stats["loss"],
                "best_epoch": best_epoch,
                "best_loss": best_loss,
            }
            save_training_state(
                checkpoint_directory / "last.pt",
                self.student,
                self.teacher,
                self.student_projector,
                self.teacher_projector,
                self.optimizer,
                payload,
            )
            if is_best:
                save_training_state(
                    checkpoint_directory / "best.pt",
                    self.student,
                    self.teacher,
                    self.student_projector,
                    self.teacher_projector,
                    self.optimizer,
                    payload,
                )

        selected = checkpoint_directory / "best.pt"
        selected_payload = load_training_state(
            selected,
            self.student,
            self.teacher,
            self.student_projector,
            self.teacher_projector,
            self.optimizer,
            self.device,
        )
        selected_epoch = int(
            selected_payload.get("best_epoch", selected_payload.get("epoch", epochs))
        )
        return {
            **last_stats,
            "selected_epoch": selected_epoch,
            "selected_checkpoint": str(selected),
        }
