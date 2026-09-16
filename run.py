from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch

from data.datasets import (
    DATASET_ORDER,
    build_ssl_loader,
    load_dataset,
    pad_splits_to_common_length,
)
from data.preprocessing import normalize_train_test
from evaluation.classification import evaluate_classifier, write_results_csv
from models.student import ProjectionHead, build_student
from models.teacher import initialize_teacher
from trainers.ssl_trainer import SSLTrainer
from utils.checkpoint import (
    checkpoint_directory,
    find_checkpoint_path,
    read_checkpoint,
)
from utils.logging import configure_logging
from utils.seed import set_seed
from workflows.fewshot import run_few_shot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate QAD")
    parser.add_argument("--mode", default="pretrain", choices=("pretrain", "downstream-only", "few-shot"), help="workflow to run")
    parser.add_argument("--resume", action="store_true", help="keep completed CSV rows, skip finished datasets, and reuse matching checkpoints")
    parser.add_argument("--checkpoint", type=Path, help="path to a specific checkpoint to load")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("result/checkpoints"), help="root directory for saving and finding checkpoints")
    parser.add_argument("--data-path", type=Path, default=Path("datasets"))
    parser.add_argument("--output-path", type=Path, default=Path("result"), help="directory for the result CSV and log")
    parser.add_argument("--result-filename", default="result.csv", help="result CSV filename; no experiment settings are added")
    parser.add_argument("--few-shot-output-path", type=Path, default=Path("result/fewshot"))
    parser.add_argument("--dataset", default="all", help="all, one dataset, or comma-separated names")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--loss-lambda", type=float, default=0.5, help="pointwise loss weight relative to quantile loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few-shot-values", nargs="+", type=int, default=(1, 5, 10, 50, 100, 500))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (
        path
        if path.is_absolute()
        else (Path(__file__).resolve().parent / path).resolve()
    )


def resolve_datasets(value: str) -> list[str]:
    if value == "all":
        return list(DATASET_ORDER)
    requested = {part.strip() for part in value.split(",") if part.strip()}
    unknown = sorted(requested - set(DATASET_ORDER))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    return [name for name in DATASET_ORDER if name in requested]


def read_completed_results(path: Path) -> list[dict[str, float | str]]:
    if not path.is_file():
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


def run_full_dataset(arguments: argparse.Namespace) -> None:
    epochs = arguments.epochs
    batch_size = arguments.batch_size
    loss_lambda = arguments.loss_lambda
    seed = arguments.seed

    data_path = resolve_path(arguments.data_path)
    result_filename = Path(arguments.result_filename)
    if len(result_filename.parts) != 1 or result_filename.name in ("", ".", ".."):
        raise ValueError("--result-filename must be a filename, not a path")
    result_path = resolve_path(arguments.output_path) / result_filename
    checkpoint_root = resolve_path(arguments.checkpoint_root)
    datasets = resolve_datasets(arguments.dataset)
    downstream_only = arguments.mode == "downstream-only"
    explicit_checkpoint: Path | None = None
    if arguments.checkpoint is not None:
        if not downstream_only:
            raise ValueError("--checkpoint is only valid with --mode downstream-only")
        if len(datasets) != 1:
            raise ValueError("--checkpoint requires exactly one dataset")
        explicit_checkpoint = resolve_path(arguments.checkpoint)

    device = torch.device(arguments.device)
    logger = configure_logging(
        result_path.with_suffix(".log"),
        mode="a" if arguments.resume else "w",
    )
    logger.info(
        "action=%s mask=freq_mask probe=all_patch epochs=%d batch_size=%d "
        "loss_lambda=%g seed=%d",
        "evaluate"
        if downstream_only
        else "resume"
        if arguments.resume
        else "train",
        epochs,
        batch_size,
        loss_lambda,
        seed,
    )
    results: list[dict[str, float | int | str]] = (
        read_completed_results(result_path) if arguments.resume else []
    )
    completed_datasets = {str(row["dataset"]) for row in results}
    if arguments.resume:
        logger.info(
            "resume_result=%s completed=%s",
            result_path,
            ",".join(name for name in DATASET_ORDER if name in completed_datasets)
            or "none",
        )

    for dataset_name in datasets:
        directory = checkpoint_directory(
            checkpoint_root,
            loss_lambda,
            seed,
            dataset_name,
        )
        resume_training_checkpoint: Path | None = None
        completed_checkpoint: dict | None = None
        completed_checkpoint_path: Path | None = None
        if arguments.resume and not downstream_only:
            last_path = directory / "last.pt"
            best_path = directory / "best.pt"
            candidate_path = last_path if last_path.is_file() else best_path
            if candidate_path.is_file():
                candidate = read_checkpoint(candidate_path, "cpu")
                completed_epoch = int(candidate["epoch"])
                if completed_epoch < epochs:
                    resume_training_checkpoint = candidate_path
                else:
                    completed_checkpoint_path = (
                        best_path if best_path.is_file() else candidate_path
                    )
                    completed_checkpoint = read_checkpoint(completed_checkpoint_path, "cpu")

        if dataset_name in completed_datasets:
            if downstream_only or resume_training_checkpoint is None:
                logger.info("dataset=%s action=skip_completed", dataset_name)
                continue
            logger.info(
                "dataset=%s action=extend_training checkpoint=%s target_epoch=%d",
                dataset_name,
                resume_training_checkpoint,
                epochs,
            )
            results = [row for row in results if row["dataset"] != dataset_name]
            completed_datasets.remove(dataset_name)
            write_results_csv(result_path, results)

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
        x_train, x_test, train_mean, train_std = normalize_train_test(
            x_train, x_test, train_lengths
        )
        channels = x_train.shape[2]
        sequence_length = x_train.shape[1]
        student = build_student(channels, sequence_length).to(device)
        teacher = initialize_teacher(student).to(device)
        selected_epoch: int | None = None

        trained = False
        checkpoint = completed_checkpoint
        if checkpoint is not None:
            logger.info(
                "dataset=%s action=reuse_checkpoint checkpoint=%s",
                dataset_name,
                completed_checkpoint_path,
            )

        if downstream_only:
            checkpoint_path = explicit_checkpoint or find_checkpoint_path(
                checkpoint_root,
                loss_lambda,
                seed,
                dataset_name,
            )
            checkpoint = read_checkpoint(checkpoint_path, "cpu")
            teacher.load_state_dict(checkpoint["teacher_state_dict"])
            selected_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch"))
        elif checkpoint is not None:
            teacher.load_state_dict(checkpoint["teacher_state_dict"])
            selected_epoch = checkpoint.get("best_epoch", checkpoint.get("epoch"))
        else:
            trained = True
            loader = build_ssl_loader(x_train, train_lengths, batch_size)
            student_projector = ProjectionHead().to(device)
            teacher_projector = initialize_teacher(student_projector).to(device)
            trainer = SSLTrainer(
                student,
                teacher,
                student_projector,
                teacher_projector,
                device,
                arguments.learning_rate,
                loss_lambda,
            )
            training = trainer.fit(
                loader,
                epochs,
                directory,
                {
                    "dataset": dataset_name,
                    "dataset_info": {
                        "L": sequence_length,
                        "C": channels,
                        "n_classes": len(classes),
                    },
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "classes": classes,
                    "pretrain_actual_epochs": epochs,
                    "loss_lambda": loss_lambda,
                    "batch_size": batch_size,
                    "seed": seed,
                },
                resume_checkpoint=resume_training_checkpoint,
            )
            selected_epoch = int(training["selected_epoch"])

        metrics = evaluate_classifier(
            teacher,
            x_train,
            y_train,
            train_lengths,
            x_test,
            y_test,
            test_lengths,
            device,
            batch_size,
        )
        results.append({"dataset": dataset_name, **metrics})
        results.sort(key=lambda row: DATASET_ORDER.index(str(row["dataset"])))
        write_results_csv(result_path, results)
        logger.info(
            "dataset=%s accuracy=%.6f f1_macro=%.6f feature_dim=%d "
            "selected_epoch=%s",
            dataset_name,
            metrics["linear_accuracy"],
            metrics["linear_f1_macro"],
            metrics["feature_dim"],
            selected_epoch,
        )

        del student, teacher, x_train, x_test
        if trained:
            del loader, trainer, student_projector, teacher_projector
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("results=%s", result_path)


def main() -> None:
    arguments = parse_args()
    if arguments.mode == "few-shot":
        run_few_shot(
            data_path=resolve_path(arguments.data_path),
            checkpoint_root=resolve_path(arguments.checkpoint_root),
            output_path=resolve_path(arguments.few_shot_output_path),
            datasets=resolve_datasets(arguments.dataset),
            shots=tuple(arguments.few_shot_values),
            seed=arguments.seed,
            batch_size=arguments.batch_size,
            loss_lambda=arguments.loss_lambda,
            device=torch.device(arguments.device),
            resume=arguments.resume,
        )
        return
    run_full_dataset(arguments)


if __name__ == "__main__":
    main()
