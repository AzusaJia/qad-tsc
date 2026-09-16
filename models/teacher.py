from __future__ import annotations

import copy

import torch
from torch import nn


EMA_MOMENTUM = 0.996


def initialize_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def update_ema_teacher(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float = EMA_MOMENTUM,
) -> None:
    """Update parameters by EMA and copy all buffers, matching the source."""
    with torch.no_grad():
        for student_parameter, teacher_parameter in zip(
            student.parameters(), teacher.parameters()
        ):
            teacher_parameter.data.mul_(momentum).add_(
                student_parameter.data, alpha=1.0 - momentum
            )
        for (_, student_buffer), (_, teacher_buffer) in zip(
            student.named_buffers(), teacher.named_buffers()
        ):
            teacher_buffer.copy_(student_buffer)
