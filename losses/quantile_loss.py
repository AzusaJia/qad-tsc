from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from data.preprocessing import (
    PATCH_SCALES,
    apply_frequency_mask,
    make_multiscale_masks,
    make_multiscale_patches,
)


QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
SOFT_QUANTILE_TEMPERATURE = 1.0


def sorted_soft_quantile(
    values: torch.Tensor,
    levels: tuple[float, ...] = QUANTILE_LEVELS,
    temperature: float = SOFT_QUANTILE_TEMPERATURE,
) -> torch.Tensor:
    """Differentiable batched channel-wise quantiles for [M,N,D]."""
    sorted_values = torch.sort(values, dim=1).values
    ranks = torch.arange(
        sorted_values.shape[1], device=values.device, dtype=values.dtype
    )
    quantiles = torch.as_tensor(levels, device=values.device, dtype=values.dtype)
    target_ranks = quantiles * (sorted_values.shape[1] - 1)
    distance = ranks.unsqueeze(0) - target_ranks.unsqueeze(1)
    weights = torch.softmax(-0.5 * (distance / temperature).pow(2), dim=1)
    return torch.einsum("mnd,qn->mdq", sorted_values, weights)


def quantile_alignment(
    student_values: torch.Tensor,
    teacher_values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    levels = torch.as_tensor(
        QUANTILE_LEVELS,
        device=teacher_values.device,
        dtype=teacher_values.dtype,
    )
    if bool(valid_mask.all()):
        student_quantiles = sorted_soft_quantile(student_values)
        with torch.no_grad():
            teacher_quantiles = torch.quantile(teacher_values, levels, dim=1).permute(
                1, 2, 0
            )
    else:
        student_parts = []
        teacher_parts = []
        for row in range(student_values.shape[0]):
            valid = valid_mask[row]
            if not valid.any():
                continue
            student_row = student_values[row][valid]
            teacher_row = teacher_values[row][valid]
            student_parts.append(sorted_soft_quantile(student_row.unsqueeze(0))[0])
            with torch.no_grad():
                teacher_parts.append(
                    torch.quantile(teacher_row, levels, dim=0).transpose(0, 1)
                )
        if not student_parts:
            return student_values.sum() * 0.0
        student_quantiles = torch.stack(student_parts)
        teacher_quantiles = torch.stack(teacher_parts)
    return F.smooth_l1_loss(
        student_quantiles,
        teacher_quantiles.detach(),
        reduction="mean",
        beta=1.0,
    )


def pointwise_alignment(
    student_values: torch.Tensor,
    teacher_values: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """The source patch ablation: same-position SmoothL1 over valid tokens."""
    if not valid_mask.any():
        return student_values.sum() * 0.0
    return F.smooth_l1_loss(
        student_values[valid_mask],
        teacher_values[valid_mask].detach(),
        reduction="mean",
        beta=1.0,
    )


def _project_tokens(values: torch.Tensor, projector: nn.Module) -> torch.Tensor:
    values = F.layer_norm(values, (values.shape[-1],))
    values = projector(values)
    return F.layer_norm(values, (values.shape[-1],))


def patch_alignment_loss(
    student: nn.Module,
    teacher: nn.Module,
    student_projector: nn.Module,
    teacher_projector: nn.Module,
    clean_values: torch.Tensor,
    padding_mask: torch.Tensor,
    loss_lambda: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute quantile loss plus weighted pointwise loss."""

    patches = make_multiscale_patches(clean_values, PATCH_SCALES)
    masks = make_multiscale_masks(padding_mask, PATCH_SCALES)
    quantile_losses = []
    pointwise_losses = []

    for scale in PATCH_SCALES:
        patch_values = patches[scale]
        patch_mask = masks[scale].bool()
        student_values = apply_frequency_mask(patch_values, patch_mask)

        student_tokens = student(student_values, patch_mask)
        student_projected = _project_tokens(student_tokens, student_projector)
        with torch.no_grad():
            teacher_tokens = teacher(patch_values, patch_mask)
            teacher_projected = _project_tokens(teacher_tokens, teacher_projector)

        quantile_losses.append(
            quantile_alignment(student_projected, teacher_projected, patch_mask)
        )
        pointwise_losses.append(
            pointwise_alignment(student_projected, teacher_projected, patch_mask)
        )

    quantile_loss = torch.stack(quantile_losses).mean()
    pointwise_loss = torch.stack(pointwise_losses).mean()
    total = quantile_loss + loss_lambda * pointwise_loss
    return total, {
        "quantile_loss": float(quantile_loss.detach().item()),
        "pointwise_loss": float(pointwise_loss.detach().item()),
    }
