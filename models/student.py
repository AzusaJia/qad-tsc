from __future__ import annotations

import torch
from torch import nn

from models.backbone import ENCODER_OUTPUT_DIM, TCNBackbone


PROJECTOR_HIDDEN_DIM = 512
PROJECTOR_OUTPUT_DIM = 256


class ProjectionHead(nn.Module):
    """Token projector shared by the source quantile/point-wise objectives."""

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(ENCODER_OUTPUT_DIM, PROJECTOR_HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(PROJECTOR_HIDDEN_DIM)
        self.act1 = nn.GELU()
        self.linear2 = nn.Linear(PROJECTOR_HIDDEN_DIM, PROJECTOR_OUTPUT_DIM)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act1(self.ln1(self.linear1(values))))


def build_student(input_channels: int, max_length: int) -> TCNBackbone:
    return TCNBackbone(input_channels, max_length)
