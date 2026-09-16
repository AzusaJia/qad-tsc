from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
import torch

from data.datasets import DATASET_ORDER, load_dataset, pad_splits_to_common_length
from data.preprocessing import normalize_train_test
from evaluation.classification import (
    evaluate_feature_classifier,
    extract_features,
    write_results_csv,
)
from models.student import build_student
from utils.checkpoint import find_checkpoint_path, read_checkpoint
from utils.logging import configure_logging
from utils.seed import set_seed


def select_few_shot_indices(
    labels: np.ndarray,
    class_count: int,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selections = []
    for class_index in range(class_count):
        indices = np.flatnonzero(labels == class_index)
        size = min(samples_per_class, len(indices))
        selections.append(rng.choice(indices, size=size, replace=False))
    return np.sort(np.concatenate(selections))


def result_stem(loss_lambda: float) -> str:
    return f"quantile_pointwise_freq_mask_all_patch_lambda{loss_lambda:g}"


def result_path(output_path: Path, stem: str, seed: int, shot: int) -> Path:
    return output_path / f"{stem}_seed{seed}_fewshot{shot}.csv"


def read_results(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                "dataset": row["dataset"],
                "linear_accuracy": float(row["linear_accuracy"]),
                "linear_f1_macro": float(row["linear_f1_macro"]),
            }
            for row in csv.DictReader(handle)
            if row["dataset"] != "mean"
        ]


def write_seed_results(
    output_path: Path,
    stem: str,
    seed: int,
    rows_by_shot: dict[int, list[dict[str, float | int | str]]],
) -> None:
    for shot, rows in rows_by_shot.items():
        rows.sort(key=lambda row: DATASET_ORDER.index(str(row["dataset"])))
        write_results_csv(result_path(output_path, stem, seed, shot), rows)


def run_seed(
    data_path: Path,
    checkpoint_root: Path,
    output_path: Path,
    datasets: list[str],
    shots: tuple[int, ...],
    seed: int,
    batch_size: int,
    loss_lambda: float,
    device: torch.device,
    resume: bool,
) -> None:
    stem = result_stem(loss_lambda)
    logger = configure_logging(
        output_path / f"{stem}_seed{seed}_fewshot.log",
        mode="a" if resume else "w",
    )
    rows_by_shot = {
        shot: read_results(result_path(output_path, stem, seed, shot)) if resume else []
        for shot in shots
    }

    for dataset_name in datasets:
        completed = {
            shot: {str(row["dataset"]) for row in rows}
            for shot, rows in rows_by_shot.items()
        }
        if resume and all(dataset_name in completed[shot] for shot in shots):
            continue

        set_seed(seed)
        (
            x_train,
            y_train,
            x_test,
            y_test,
            train_lengths,
            test_lengths,
            classes,
        ) = load_dataset(data_path, dataset_name)
        x_train, x_test = pad_splits_to_common_length(x_train, x_test)
        x_train, x_test, _, _ = normalize_train_test(x_train, x_test, train_lengths)

        selected_by_shot = {
            shot: select_few_shot_indices(y_train, len(classes), shot, seed)
            for shot in shots
        }
        selected_union = np.unique(np.concatenate(tuple(selected_by_shot.values())))
        checkpoint_path = find_checkpoint_path(
            checkpoint_root,
            loss_lambda,
            seed,
            dataset_name,
        )
        checkpoint = read_checkpoint(checkpoint_path)
        encoder = build_student(x_train.shape[2], x_train.shape[1]).to(device)
        encoder.load_state_dict(checkpoint["teacher_state_dict"])
        encoder.requires_grad_(False)

        train_features = extract_features(
            encoder,
            x_train[selected_union],
            train_lengths[selected_union],
            device,
            batch_size,
        )
        test_features = extract_features(
            encoder,
            x_test,
            test_lengths,
            device,
            batch_size,
        )

        for shot, selected in selected_by_shot.items():
            positions = np.searchsorted(selected_union, selected)
            metrics = evaluate_feature_classifier(
                train_features[positions],
                y_train[selected],
                test_features,
                y_test,
            )
            rows_by_shot[shot] = [
                row for row in rows_by_shot[shot] if row["dataset"] != dataset_name
            ]
            rows_by_shot[shot].append({"dataset": dataset_name, **metrics})
            logger.info(
                "dataset=%s fewshot=%d accuracy=%.6f f1_macro=%.6f",
                dataset_name,
                shot,
                metrics["linear_accuracy"],
                metrics["linear_f1_macro"],
            )
        write_seed_results(output_path, stem, seed, rows_by_shot)

        del encoder, checkpoint, x_train, x_test, train_features, test_features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_few_shot(
    data_path: Path,
    checkpoint_root: Path,
    output_path: Path,
    datasets: list[str],
    shots: tuple[int, ...],
    seed: int,
    batch_size: int,
    loss_lambda: float,
    device: torch.device,
    resume: bool,
) -> None:
    run_seed(
        data_path,
        checkpoint_root,
        output_path,
        datasets,
        shots,
        seed,
        batch_size,
        loss_lambda,
        device,
        resume,
    )
