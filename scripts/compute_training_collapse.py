from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import (  # noqa: E402
    DATASET_ORDER,
    load_dataset,
    pad_splits_to_common_length,
)
from data.preprocessing import normalize_train_test  # noqa: E402
from evaluation.classification import extract_features  # noqa: E402
from models.student import build_student  # noqa: E402
from utils.checkpoint import find_checkpoint_path, read_checkpoint  # noqa: E402
from utils.seed import set_seed  # noqa: E402


DETAIL_FIELDS = (
    "dataset",
    "seed",
    "n_train",
    "feature_dim",
    "feature_std_ddof",
    "mean_feature_std",
    "best_epoch",
    "best_train_loss",
    "checkpoint",
)
SUMMARY_FIELDS = (
    "dataset",
    "num_seeds",
    "mean_feature_std_mean",
    "mean_feature_std_std_population",
    "mean_feature_std_std_sample",
    "mean_feature_std_min",
    "mean_feature_std_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure feature collapse in pretrained representations"
    )
    parser.add_argument("--data-path", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("result/checkpoints"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("result_all/collapse"),
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ORDER, default=DATASET_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=range(40, 45))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-std-ddof", type=int, default=0)
    parser.add_argument("--loss-lambda", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).absolute()


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for dataset_name in DATASET_ORDER:
        values = np.asarray(
            [
                float(row["mean_feature_std"])
                for row in rows
                if row["dataset"] == dataset_name
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        summaries.append(
            {
                "dataset": dataset_name,
                "num_seeds": int(values.size),
                "mean_feature_std_mean": float(values.mean()),
                "mean_feature_std_std_population": float(values.std(ddof=0)),
                "mean_feature_std_std_sample": (
                    float(values.std(ddof=1)) if values.size > 1 else 0.0
                ),
                "mean_feature_std_min": float(values.min()),
                "mean_feature_std_max": float(values.max()),
            }
        )
    return summaries


def main() -> None:
    arguments = parse_args()
    data_path = resolve_path(arguments.data_path)
    checkpoint_root = resolve_path(arguments.checkpoint_root)
    output_path = resolve_path(arguments.output_path)
    device = torch.device(arguments.device)
    rows = []

    for dataset_name in arguments.datasets:
        for seed in arguments.seeds:
            set_seed(seed)
            (
                x_train,
                _,
                x_test,
                _,
                train_lengths,
                _,
                _,
            ) = load_dataset(data_path, dataset_name)
            x_train, x_test = pad_splits_to_common_length(x_train, x_test)
            x_train, _, _, _ = normalize_train_test(
                x_train, x_test, train_lengths
            )

            checkpoint_path = find_checkpoint_path(
                checkpoint_root,
                arguments.loss_lambda,
                seed,
                dataset_name,
            )
            checkpoint = read_checkpoint(checkpoint_path, device)
            encoder = build_student(x_train.shape[2], x_train.shape[1]).to(device)
            encoder.load_state_dict(checkpoint["teacher_state_dict"])
            encoder.requires_grad_(False)
            features = extract_features(
                encoder,
                x_train,
                train_lengths,
                device,
                arguments.batch_size,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "n_train": int(features.shape[0]),
                    "feature_dim": int(features.shape[1]),
                    "feature_std_ddof": arguments.feature_std_ddof,
                    "mean_feature_std": float(
                        features.std(axis=0, ddof=arguments.feature_std_ddof).mean()
                    ),
                    "best_epoch": checkpoint.get("best_epoch", checkpoint.get("epoch")),
                    "best_train_loss": checkpoint.get(
                        "best_loss", checkpoint.get("train_loss")
                    ),
                    "checkpoint": os.path.relpath(checkpoint_path, PROJECT_ROOT),
                }
            )
            print(
                f"dataset={dataset_name} seed={seed} "
                f"mean_feature_std={rows[-1]['mean_feature_std']:.12g}"
            )
            del encoder, checkpoint, features, x_train, x_test
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(output_path / "training_feature_std_by_seed.csv", DETAIL_FIELDS, rows)
    write_csv(
        output_path / "training_feature_std_summary.csv",
        SUMMARY_FIELDS,
        summarize(rows),
    )


if __name__ == "__main__":
    main()
