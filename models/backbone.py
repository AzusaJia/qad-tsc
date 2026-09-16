from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


D_MODEL = 128
TCN_HIDDEN_CHANNELS = 80
ENCODER_OUTPUT_DIM = 320
NUM_LAYERS = 5
DROPOUT = 0.1


class FixedPositionalEncoding(nn.Module):
    def __init__(self, dimension: int, dropout: float, max_length: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        encoding = torch.zeros(max_length, dimension)
        position = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0).transpose(0, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.dropout(values + self.pe[: values.size(0)])


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dilation: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.residual_proj = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Linear(input_channels, output_channels)
        )
        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(values)
        output = self.conv1(values.transpose(1, 2)).transpose(1, 2)
        output = self.dropout(F.gelu(output))
        output = self.conv2(output.transpose(1, 2)).transpose(1, 2)
        output = self.dropout(F.gelu(output))
        if output.size(1) > residual.size(1):
            output = output[:, : residual.size(1)]
        elif output.size(1) < residual.size(1):
            output = F.pad(output, (0, 0, 0, residual.size(1) - output.size(1)))
        return residual + self.residual_scale * output


class TCNBackbone(nn.Module):
    """The five-block TCN encoder used by QAD."""

    def __init__(self, input_channels: int, max_length: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_channels, D_MODEL)
        self.pos_enc = FixedPositionalEncoding(D_MODEL, DROPOUT, max_length)
        channels = (
            [D_MODEL] + [TCN_HIDDEN_CHANNELS] * (NUM_LAYERS - 1) + [ENCODER_OUTPUT_DIM]
        )
        self.blocks = nn.ModuleList(
            [
                TCNResidualBlock(
                    channels[index],
                    channels[index + 1],
                    dilation=2**index,
                    dropout=DROPOUT,
                    residual_scale=1.0 / math.sqrt(NUM_LAYERS),
                )
                for index in range(NUM_LAYERS)
            ]
        )
        self.block_norms = nn.ModuleList(
            [
                nn.LayerNorm(channels[index + 1], elementwise_affine=False)
                for index in range(NUM_LAYERS)
            ]
        )
        self.dropout1 = nn.Dropout(DROPOUT)

    @property
    def project_inp(self) -> nn.Module:
        return self.input_projection

    def forward(
        self,
        values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.input_projection(values) * math.sqrt(D_MODEL)
        output = self.pos_enc(output.transpose(0, 1)).transpose(0, 1)
        for block, normalization in zip(self.blocks, self.block_norms):
            output = block(output)
            output = output.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
            output = normalization(output)
            output = output.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        return self.dropout1(F.gelu(output))
