from __future__ import annotations

import math

import numpy as np
import torch


PATCH_SCALES = (1, 2, 4, 8)
FREQUENCY_BANDS = 8
MASKED_FREQUENCY_BANDS = 1
EXCLUDE_DC = True
FREQUENCY_MASK_PROBABILITY = 1.0
FREQUENCY_MASK_PER_CHANNEL = False


def normalize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
    train_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel standardization fitted only on valid training positions."""
    sample_count, _, channels = x_train.shape
    mean = np.zeros(channels, dtype=np.float64)
    std = np.ones(channels, dtype=np.float64)
    for channel in range(channels):
        values = np.concatenate(
            [
                x_train[index, : int(train_lengths[index]), channel].astype(np.float64)
                for index in range(sample_count)
            ]
        )
        mean[channel] = values.mean()
        channel_std = values.std()
        std[channel] = channel_std if channel_std > 1e-8 else 1.0
    mean32 = mean.astype(np.float32)
    std32 = std.astype(np.float32)
    train = ((x_train - mean32[None, None, :]) / std32[None, None, :]).astype(
        np.float32
    )
    test = ((x_test - mean32[None, None, :]) / std32[None, None, :]).astype(np.float32)
    return train, test, mean32, std32


def make_multiscale_patches(
    values: torch.Tensor,
    scales: tuple[int, ...] = PATCH_SCALES,
) -> dict[int, torch.Tensor]:
    """Split [B,L,C] into the original non-overlapping patch hierarchy."""
    if values.ndim != 3:
        raise ValueError(f"values must be [B,L,C], got {list(values.shape)}")
    batch, length, channels = values.shape
    result: dict[int, torch.Tensor] = {}
    for scale in scales:
        patch_length = length // scale
        if patch_length <= 0:
            continue
        usable = patch_length * scale
        result[scale] = (
            values[:, :usable]
            .contiguous()
            .reshape(batch, scale, patch_length, channels)
            .reshape(batch * scale, patch_length, channels)
        )
    if not result:
        raise ValueError(f"No patch scale is valid for sequence length {length}")
    return result


def make_multiscale_masks(
    padding_mask: torch.Tensor,
    scales: tuple[int, ...] = PATCH_SCALES,
) -> dict[int, torch.Tensor]:
    batch, length = padding_mask.shape
    result: dict[int, torch.Tensor] = {}
    for scale in scales:
        patch_length = length // scale
        if patch_length <= 0:
            continue
        usable = patch_length * scale
        result[scale] = (
            padding_mask[:, :usable]
            .contiguous()
            .reshape(batch, scale, patch_length)
            .reshape(batch * scale, patch_length)
        )
    return result


def apply_frequency_mask(
    values: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    num_bands: int = FREQUENCY_BANDS,
    mask_bands: int = MASKED_FREQUENCY_BANDS,
    exclude_dc: bool = EXCLUDE_DC,
    probability: float = FREQUENCY_MASK_PROBABILITY,
    per_channel: bool = FREQUENCY_MASK_PER_CHANNEL,
) -> torch.Tensor:
    """Apply the original random contiguous rFFT-band dropout."""
    if values.ndim != 3:
        raise ValueError(f"values must be [B,P,C], got {list(values.shape)}")
    items, patch_length, channels = values.shape
    spectrum = torch.fft.rfft(values, dim=1)
    fft_bins = int(spectrum.shape[1])
    first_bin = 1 if exclude_dc else 0
    if fft_bins <= first_bin or probability <= 0.0:
        return values

    maskable = torch.arange(first_bin, fft_bins, device=values.device)
    count = int(maskable.numel())
    effective_band_count = min(num_bands, count)
    bands = []
    for band_index in range(effective_band_count):
        start = int(math.floor(band_index * count / effective_band_count))
        end = int(math.floor((band_index + 1) * count / effective_band_count))
        if end > start:
            bands.append(maskable[start:end])
    if not bands:
        return values

    effective_band_count = len(bands)
    selected_count = min(mask_bands, effective_band_count)
    frequency_mask = torch.zeros(
        items, fft_bins, channels, dtype=torch.bool, device=values.device
    )
    apply_flags = torch.rand(items, device=values.device) < probability
    for item in range(items):
        if not bool(apply_flags[item].item()):
            continue
        if per_channel:
            for channel in range(channels):
                selected = torch.randperm(effective_band_count, device=values.device)[
                    :selected_count
                ]
                for band_id in selected.tolist():
                    frequency_mask[item, bands[band_id], channel] = True
        else:
            selected = torch.randperm(effective_band_count, device=values.device)[
                :selected_count
            ]
            for band_id in selected.tolist():
                frequency_mask[item, bands[band_id], :] = True

    if frequency_mask.any():
        spectrum = spectrum.masked_fill(frequency_mask, 0)
        corrupted = torch.fft.irfft(spectrum, n=patch_length, dim=1).to(values.dtype)
    else:
        corrupted = values
    if padding_mask is not None:
        corrupted = torch.where(padding_mask.unsqueeze(-1), corrupted, values)
    return corrupted
