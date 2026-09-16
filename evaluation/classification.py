from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.preprocessing import (
    PATCH_SCALES,
    make_multiscale_masks,
    make_multiscale_patches,
)


# Compatibility used by the source project for its NumPy 2.x environment.
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
from sklearn.linear_model import RidgeClassifierCV  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
RESULT_FIELDS = ("dataset", "linear_accuracy", "linear_f1_macro")


def _padding_mask(
    lengths: np.ndarray, max_length: int, device: torch.device
) -> torch.Tensor:
    return (
        torch.arange(max_length, device=device)[None, :]
        < torch.as_tensor(lengths, device=device)[:, None]
    )


def _pool_patch_tokens(tokens: np.ndarray, masks: np.ndarray) -> np.ndarray:
    mask_float = masks.astype(np.float32)[..., None]
    denominator = np.maximum(mask_float.sum(axis=2), 1.0)
    mean = (tokens * mask_float).sum(axis=2) / denominator
    maximum = np.where(masks[..., None], tokens, -np.inf).max(axis=2).astype(np.float32)
    maximum = np.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0)
    empty = ~masks.any(axis=2)
    mean[empty] = 0.0
    maximum[empty] = 0.0
    return np.concatenate([mean, maximum], axis=-1).astype(np.float32, copy=False)


def _aggregate_patches(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    patch_valid = masks.any(axis=2)
    denominator = np.maximum(patch_valid.sum(axis=1).astype(np.float32), 1.0)
    mean = (features * patch_valid[..., None].astype(np.float32)).sum(
        axis=1
    ) / denominator[:, None]
    maximum = np.where(patch_valid[..., None], features, -np.inf).max(axis=1)
    maximum = np.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0)
    empty = ~patch_valid.any(axis=1)
    mean[empty] = 0.0
    maximum[empty] = 0.0
    return np.concatenate([mean, maximum], axis=1).astype(np.float32, copy=False)


def extract_features(
    encoder: nn.Module,
    values: np.ndarray,
    lengths: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Extract fixed multi-scale all-patch features (5120-D)."""
    encoder.eval()
    outputs = []
    full_length = values.shape[1]
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            batch = torch.from_numpy(values[start:end]).to(device)
            full_mask = _padding_mask(lengths[start:end], full_length, device)
            patches = make_multiscale_patches(batch, PATCH_SCALES)
            masks = make_multiscale_masks(full_mask, PATCH_SCALES)
            scale_outputs = []
            current_batch = end - start
            for scale in PATCH_SCALES:
                patch_mask = masks[scale]
                tokens = encoder(patches[scale], patch_mask)
                tokens = F.layer_norm(tokens, (tokens.shape[-1],))
                token_array = (
                    tokens.reshape(
                        current_batch, scale, tokens.shape[1], tokens.shape[2]
                    )
                    .cpu()
                    .numpy()
                )
                mask_array = (
                    patch_mask.reshape(current_batch, scale, patch_mask.shape[1])
                    .cpu()
                    .numpy()
                    .astype(bool)
                )
                pooled = _pool_patch_tokens(token_array, mask_array)
                scale_outputs.append(_aggregate_patches(pooled, mask_array))
            outputs.append(np.concatenate(scale_outputs, axis=1))
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def evaluate_classifier(
    encoder: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_lengths: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_lengths: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    train_features = extract_features(
        encoder, x_train, train_lengths, device, batch_size
    )
    test_features = extract_features(
        encoder, x_test, test_lengths, device, batch_size
    )
    return evaluate_feature_classifier(
        train_features,
        y_train,
        test_features,
        y_test,
    )


def evaluate_feature_classifier(
    train_features: np.ndarray,
    y_train: np.ndarray,
    test_features: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Fit the source-compatible frozen Ridge probe on extracted features."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    classifier = RidgeClassifierCV(alphas=RIDGE_ALPHAS, class_weight=None)
    classifier.fit(train_scaled, y_train)
    predictions = classifier.predict(test_scaled)
    return {
        "linear_accuracy": float(accuracy_score(y_test, predictions)),
        "linear_f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "ridge_alpha": float(classifier.alpha_),
        "feature_dim": int(train_features.shape[1]),
    }


def write_results_csv(
    path: str | Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    """Write source-compatible metrics and aggregation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for row in rows:
        accuracy = float(row["linear_accuracy"])
        f1 = float(row["linear_f1_macro"])
        metric_rows.append(
            {
                "dataset": row["dataset"],
                "linear_accuracy": accuracy,
                "linear_f1_macro": f1,
            }
        )
    if metric_rows:
        mean_accuracy = sum(float(row["linear_accuracy"]) for row in metric_rows) / len(
            metric_rows
        )
        mean_f1 = sum(float(row["linear_f1_macro"]) for row in metric_rows) / len(
            metric_rows
        )
        mean_accuracy = round(mean_accuracy, 6)
        mean_f1 = round(mean_f1, 6)
        metric_rows.append(
            {
                "dataset": "mean",
                "linear_accuracy": mean_accuracy,
                "linear_f1_macro": mean_f1,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(metric_rows)
