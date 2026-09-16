from __future__ import annotations

import random
from functools import partial
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DATASET_ORDER = ("Epilepsy", "HAR", "PAMAP2", "Skoda", "Sleep")
TRAIN_CROP_LENGTH = 3000


def _encode_labels(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_strings = [str(value) for value in np.asarray(train_labels).reshape(-1)]
    test_strings = [str(value) for value in np.asarray(test_labels).reshape(-1)]
    classes = sorted(set(train_strings + test_strings))
    mapping = {label: index for index, label in enumerate(classes)}
    y_train = np.asarray([mapping[label] for label in train_strings], dtype=np.int64)
    y_test = np.asarray([mapping[label] for label in test_strings], dtype=np.int64)
    return y_train, y_test, classes


def _to_time_major(values: np.ndarray, split: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError(f"{split} must be [N,C,L], got {values.shape}")
    return np.ascontiguousarray(np.transpose(values, (0, 2, 1)), dtype=np.float32)


def load_dataset(
    dataset_root: str | Path,
    dataset_name: str,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """Load the original cached train/test split from <root>/<name>/<name>.npy."""
    if dataset_name not in DATASET_ORDER:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}; choose from {DATASET_ORDER}"
        )
    path = Path(dataset_root) / dataset_name / f"{dataset_name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset cache not found: {path}")
    payload = np.load(path, allow_pickle=True).item()
    required = ("train_data", "train_label", "test_data", "test_label")
    missing = [key for key in required if key not in payload or payload[key] is None]
    if missing:
        raise KeyError(f"{path} is missing keys: {missing}")

    x_train = _to_time_major(payload["train_data"], "train_data")
    x_test = _to_time_major(payload["test_data"], "test_data")
    y_train, y_test, classes = _encode_labels(
        payload["train_label"], payload["test_label"]
    )
    train_lengths = np.full(x_train.shape[0], x_train.shape[1], dtype=np.int64)
    test_lengths = np.full(x_test.shape[0], x_test.shape[1], dtype=np.int64)
    return (
        x_train,
        y_train,
        x_test,
        y_test,
        train_lengths,
        test_lengths,
        classes,
    )


def pad_splits_to_common_length(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = max(x_train.shape[1], x_test.shape[1])
    if x_train.shape[1] < target:
        x_train = np.pad(x_train, ((0, 0), (0, target - x_train.shape[1]), (0, 0)))
    if x_test.shape[1] < target:
        x_test = np.pad(x_test, ((0, 0), (0, target - x_test.shape[1]), (0, 0)))
    return x_train, x_test


class SSLDataset(Dataset):
    """The original random-crop training dataset, restricted to the used mode."""

    def __init__(
        self,
        values: np.ndarray,
        padding_mask: torch.Tensor,
        crop_length: int = TRAIN_CROP_LENGTH,
    ) -> None:
        self.values = torch.from_numpy(values)
        self.padding_mask = padding_mask
        self.crop_length = crop_length

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        valid_length = int(self.padding_mask[index].sum().item())
        if valid_length <= 0:
            return self.values[index, :1], self.padding_mask[index, :1]
        return (
            self.values[index, :valid_length],
            self.padding_mask[index, :valid_length],
        )


def _collate_random_crop(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
    crop_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_lengths = [int(mask.sum().item()) for _, mask in batch]
    minimum = min(valid_lengths)
    start = random.randint(0, minimum - crop_length) if minimum > crop_length else 0
    cropped = []
    for values, mask in batch:
        end = min(start + crop_length, values.shape[0])
        cropped.append((values[start:end], mask[start:end]))

    max_length = max(values.shape[0] for values, _ in cropped)
    channels = cropped[0][0].shape[1]
    values_out = torch.zeros(len(cropped), max_length, channels, dtype=torch.float32)
    mask_out = torch.zeros(len(cropped), max_length, dtype=torch.bool)
    for index, (values, mask) in enumerate(cropped):
        length = values.shape[0]
        values_out[index, :length] = values
        mask_out[index, :length] = mask
    return values_out, mask_out


def build_ssl_loader(
    x_train_normalized: np.ndarray,
    train_lengths: np.ndarray,
    batch_size: int,
) -> DataLoader:
    length = x_train_normalized.shape[1]
    padding_mask = (
        torch.arange(length)[None, :] < torch.from_numpy(train_lengths)[:, None]
    )
    dataset = SSLDataset(x_train_normalized, padding_mask)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(_collate_random_crop, crop_length=TRAIN_CROP_LENGTH),
        drop_last=False,
    )
