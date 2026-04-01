"""moonshot-enhanced-setting cnn runner with grouped file validation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from smelt.datasets import (
    MOONSHOT_TRACK,
    build_base_class_vocab_manifest,
    load_base_sensor_dataset,
    prepare_moonshot_window_splits,
    stack_window_labels,
    write_base_class_vocab_manifest,
    write_moonshot_view_manifest,
)
from smelt.evaluation import (
    FILE_LEVEL_AGGREGATORS,
    FileLevelAggregationResult,
    aggregate_file_level_metrics,
    build_window_prediction_bundle,
    export_classification_report,
    export_file_level_report,
    load_category_mapping,
    write_window_prediction_bundle,
)
from smelt.models import ExactUpstreamCnnClassifier
from smelt.preprocessing import stack_window_values

from .run import (
    CnnModelConfig,
    build_dataloader,
    build_run_dir,
    collect_evaluation_outputs,
    expand_env_values,
    resolve_device,
    set_seed,
    validate_reference_manifest,
    validate_required_reference,
    write_resolved_config,
    write_run_metadata,
)


class MoonshotRunError(Exception):
    """raised when the moonshot run cannot proceed safely."""


@dataclass(slots=True)
class MoonshotRunConfig:
    track: str
    experiment_name: str
    config_path: str
    data_root: str
    output_root: str
    category_map_path: str
    class_vocab_manifest_path: str
    exact_upstream_regression_path: str
    seed: int
    device: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    diff_period: int
    window_size: int
    stride: int | None
    num_workers: int
    validation_files_per_class: int
    label_smoothing: float
    scheduler_name: str
    scheduler_t_max: int
    scheduler_eta_min: float
    primary_file_aggregator: str
    validation_file_aggregator: str
    model_name: str
    model: CnnModelConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = asdict(self.model)
        return payload


@dataclass(slots=True)
class MoonshotPreparedTensors:
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    train_windows: np.ndarray
    train_labels: np.ndarray
    validation_windows: np.ndarray
    validation_labels: np.ndarray
    test_windows: np.ndarray
    test_labels: np.ndarray
    train_window_count: int
    validation_window_count: int
    test_window_count: int
    train_standardization_shape: tuple[int, int]
    standardized_train_split: Any
    standardized_validation_split: Any
    standardized_test_split: Any
    view_manifest: dict[str, Any]


@dataclass(slots=True)
class MoonshotHistoryRow:
    epoch: int
    learning_rate: float
    train_loss: float
    train_acc_at_1: float
    validation_window_acc_at_1: float
    validation_window_f1: float
    validation_file_acc_at_1: float
    validation_file_f1: float
    is_best: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir, window_metrics, checkpoint_path, prepared, file_metrics = run_moonshot(args.config)
    print(f"run_dir: {run_dir}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"window_acc@1: {window_metrics.acc_at_1}")
    print(f"window_acc@5: {window_metrics.acc_at_5}")
    print(f"window_precision_macro: {window_metrics.precision_macro}")
    print(f"window_recall_macro: {window_metrics.recall_macro}")
    print(f"window_f1_macro: {window_metrics.f1_macro}")
    print(f"file_acc@1: {file_metrics.acc_at_1}")
    print(f"file_acc@5: {file_metrics.acc_at_5}")
    print(f"file_precision_macro: {file_metrics.precision_macro}")
    print(f"file_recall_macro: {file_metrics.recall_macro}")
    print(f"file_f1_macro: {file_metrics.f1_macro}")
    print(f"train_window_count: {prepared.train_window_count}")
    print(f"validation_window_count: {prepared.validation_window_count}")
    print(f"test_window_count: {prepared.test_window_count}")
    print(f"feature_count: {len(prepared.feature_names)}")
    return 0


def run_moonshot(
    config_path: Path,
) -> tuple[Path, Any, Path, MoonshotPreparedTensors, Any]:
    config = load_moonshot_run_config(config_path)
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = build_run_dir(Path(config.output_root), config.experiment_name, config.to_dict())
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_base_sensor_dataset(Path(config.data_root))
    vocab_manifest = build_base_class_vocab_manifest(dataset)
    validate_reference_manifest(config.class_vocab_manifest_path, vocab_manifest.to_dict())
    validate_required_reference(config.exact_upstream_regression_path)
    category_mapping = load_category_mapping(Path(config.category_map_path))

    prepared = prepare_moonshot_tensors(dataset, config)
    model = ExactUpstreamCnnClassifier(
        in_channels=prepared.train_windows.shape[2],
        num_classes=len(prepared.class_names),
        channels=config.model.channels,
        kernel_size=config.model.kernel_size,
        dropout=config.model.dropout,
        use_batchnorm=config.model.use_batchnorm,
    ).to(device)

    train_loader = build_dataloader(
        prepared.train_windows,
        prepared.train_labels,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = build_dataloader(
        prepared.validation_windows,
        prepared.validation_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = build_dataloader(
        prepared.test_windows,
        prepared.test_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    best_checkpoint_path = run_dir / "checkpoint_best.pt"
    history = train_moonshot_classifier(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_windows=prepared.standardized_validation_split.windows,
        class_names=prepared.class_names,
        category_mapping=category_mapping,
        device=device,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        label_smoothing=config.label_smoothing,
        scheduler_name=config.scheduler_name,
        scheduler_t_max=config.scheduler_t_max,
        scheduler_eta_min=config.scheduler_eta_min,
        validation_file_aggregator=config.validation_file_aggregator,
        checkpoint_path=best_checkpoint_path,
        config_payload=config.to_dict(),
    )

    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    checkpoint_path = run_dir / "checkpoint_final.pt"
    torch.save(best_checkpoint, checkpoint_path)

    test_evaluation = collect_evaluation_outputs(
        model=model,
        data_loader=test_loader,
        device=device,
        class_names=prepared.class_names,
        category_mapping=category_mapping,
    )
    export_classification_report(
        output_root=run_dir.parent,
        run_name=run_dir.name,
        metrics=test_evaluation.metrics,
        methods_summary={
            "track": MOONSHOT_TRACK,
            "view_mode": "diff_all12",
            "exact_upstream_regression_path": str(
                Path(config.exact_upstream_regression_path).resolve()
            ),
            "train_window_count": prepared.train_window_count,
            "validation_window_count": prepared.validation_window_count,
            "test_window_count": prepared.test_window_count,
            "train_standardization_shape": list(prepared.train_standardization_shape),
            "feature_names": list(prepared.feature_names),
            "retained_columns": list(prepared.feature_names),
            "validation_files_per_class": config.validation_files_per_class,
            "primary_file_aggregator": config.primary_file_aggregator,
            "validation_file_aggregator": config.validation_file_aggregator,
            "best_validation_epoch": int(best_checkpoint.get("epoch", 0)),
            "best_validation_window_acc@1": float(
                best_checkpoint.get("best_validation_window_acc@1", 0.0)
            ),
            "best_validation_window_acc@5": float(
                best_checkpoint.get("best_validation_window_acc@5", 0.0)
            ),
            "best_validation_window_f1": float(
                best_checkpoint.get("best_validation_window_f1", 0.0)
            ),
            "best_validation_file_acc@1": float(
                best_checkpoint.get("best_validation_file_acc@1", 0.0)
            ),
            "best_validation_file_acc@5": float(
                best_checkpoint.get("best_validation_file_acc@5", 0.0)
            ),
            "best_validation_file_f1": float(best_checkpoint.get("best_validation_file_f1", 0.0)),
            "setting_note": MOONSHOT_TRACK,
        },
        overwrite=True,
    )
    test_bundle = build_window_prediction_bundle(
        class_names=test_evaluation.metrics.class_names,
        true_labels=test_evaluation.true_labels,
        predicted_labels=test_evaluation.predicted_labels,
        topk_indices=test_evaluation.topk_indices,
        logits=test_evaluation.logits,
        windows=prepared.standardized_test_split.windows,
    )
    write_window_prediction_bundle(run_dir / "predictions.npz", test_bundle)

    file_level_rows: list[dict[str, str]] = []
    primary_file_metrics = None
    primary_paths: dict[str, str] | None = None
    for aggregator in FILE_LEVEL_AGGREGATORS:
        file_result = aggregate_file_level_metrics(
            bundle=test_bundle,
            category_mapping=category_mapping,
            aggregator=aggregator,
        )
        report_paths = export_file_level_report(
            output_root=run_dir / "file_level",
            run_name=aggregator,
            result=file_result,
            methods_summary={
                "track": MOONSHOT_TRACK,
                "view_mode": "diff_all12",
                "setting_note": MOONSHOT_TRACK,
                "window_level_acc@1": test_evaluation.metrics.acc_at_1,
                "window_level_acc@5": test_evaluation.metrics.acc_at_5,
                "window_level_f1_macro": test_evaluation.metrics.f1_macro,
            },
        )
        file_level_rows.append(
            build_file_level_summary_row(
                run_id=run_dir.name,
                track=MOONSHOT_TRACK,
                model_family=config.model_name,
                view_mode="diff_all12",
                diff_period=config.diff_period,
                window_size=config.window_size,
                stride=prepared.standardized_test_split.stride,
                window_metrics=test_evaluation.metrics,
                file_result=file_result,
                report_paths=report_paths,
            ),
        )
        if aggregator == config.primary_file_aggregator:
            primary_file_metrics = file_result.metrics
            primary_paths = report_paths.to_dict()
    if primary_file_metrics is None or primary_paths is None:
        raise MoonshotRunError("primary file aggregator report was not produced")

    file_level_csv = run_dir / "file_level_metrics_comparison.csv"
    file_level_json = run_dir / "file_level_metrics_comparison.json"
    validation_split_json = run_dir / "validation_split.json"
    write_dict_rows_csv(file_level_csv, file_level_rows)
    write_json(file_level_json, {"rows": file_level_rows})
    write_json(
        validation_split_json,
        {
            "train_relative_paths": [
                window.relative_path
                for window in prepared.standardized_train_split.windows
                if window.window_index == 0
            ],
            "validation_relative_paths": [
                window.relative_path
                for window in prepared.standardized_validation_split.windows
                if window.window_index == 0
            ],
        },
    )
    write_training_history(run_dir / "training_history.csv", history)
    write_resolved_config(run_dir / "resolved_config.yaml", config)
    write_run_metadata(
        run_dir / "run_metadata.json",
        {
            "track": MOONSHOT_TRACK,
            "mode": "supervised_classification",
            "view_mode": "diff_all12",
            "reference_artifacts": {
                "category_map_path": str(Path(config.category_map_path).resolve()),
                "class_vocab_manifest_path": str(Path(config.class_vocab_manifest_path).resolve()),
                "exact_upstream_regression_path": str(
                    Path(config.exact_upstream_regression_path).resolve()
                ),
            },
            "window_counts": {
                "train": prepared.train_window_count,
                "validation": prepared.validation_window_count,
                "test": prepared.test_window_count,
            },
            "feature_names": list(prepared.feature_names),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "best_checkpoint_path": str(best_checkpoint_path.resolve()),
            "validation_files_per_class": config.validation_files_per_class,
            "primary_file_aggregator": config.primary_file_aggregator,
            "validation_file_aggregator": config.validation_file_aggregator,
            "best_validation_summary": {
                "epoch": int(best_checkpoint.get("epoch", 0)),
                "window_acc@1": float(best_checkpoint.get("best_validation_window_acc@1", 0.0)),
                "window_acc@5": float(best_checkpoint.get("best_validation_window_acc@5", 0.0)),
                "window_macro_f1": float(best_checkpoint.get("best_validation_window_f1", 0.0)),
                "file_acc@1": float(best_checkpoint.get("best_validation_file_acc@1", 0.0)),
                "file_acc@5": float(best_checkpoint.get("best_validation_file_acc@5", 0.0)),
                "file_macro_f1": float(best_checkpoint.get("best_validation_file_f1", 0.0)),
            },
            "file_level_metrics_path": str(file_level_json.resolve()),
            "file_level_primary_report": primary_paths,
            "validation_split_path": str(validation_split_json.resolve()),
            "setting_note": MOONSHOT_TRACK,
        },
    )
    write_base_class_vocab_manifest(run_dir / "base_class_vocab.json", vocab_manifest)
    write_moonshot_view_manifest(run_dir / "moonshot_view_manifest.json", prepared.view_manifest)
    return run_dir, test_evaluation.metrics, checkpoint_path, prepared, primary_file_metrics


def run_moonshot_cnn(
    config_path: Path,
) -> tuple[Path, Any, Any, MoonshotPreparedTensors]:
    run_dir, window_metrics, _checkpoint_path, prepared, file_metrics = run_moonshot(config_path)
    return run_dir, window_metrics, file_metrics, prepared


def load_moonshot_run_config(config_path: Path) -> MoonshotRunConfig:
    try:
        raw_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MoonshotRunError(f"unable to read config file: {config_path}") from exc
    if not isinstance(raw_payload, dict):
        raise MoonshotRunError("moonshot run config must deserialize to a mapping")
    payload = expand_env_values(raw_payload)
    required_keys = {
        "track",
        "experiment_name",
        "data_root",
        "output_root",
        "category_map_path",
        "class_vocab_manifest_path",
        "exact_upstream_regression_path",
        "seed",
        "device",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "diff_period",
        "window_size",
        "stride",
        "num_workers",
        "validation_files_per_class",
        "label_smoothing",
        "scheduler_name",
        "scheduler_t_max",
        "scheduler_eta_min",
        "primary_file_aggregator",
        "validation_file_aggregator",
        "model_name",
        "model",
    }
    missing = sorted(required_keys - set(payload))
    if missing:
        raise MoonshotRunError(f"moonshot config is missing required keys: {missing}")
    if payload["track"] != MOONSHOT_TRACK:
        raise MoonshotRunError(f"unsupported track for moonshot runner: {payload['track']!r}")
    if payload["model_name"] != "cnn":
        raise MoonshotRunError("moonshot m01 only supports cnn model_name")
    for aggregator_key in ("primary_file_aggregator", "validation_file_aggregator"):
        if payload[aggregator_key] not in FILE_LEVEL_AGGREGATORS:
            raise MoonshotRunError(f"{aggregator_key} must be one of {FILE_LEVEL_AGGREGATORS}")
    model_payload = payload["model"]
    if not isinstance(model_payload, dict):
        raise MoonshotRunError("moonshot cnn model config must be a mapping")
    raw_channels = model_payload["channels"]
    if not isinstance(raw_channels, list) or not raw_channels:
        raise MoonshotRunError("moonshot cnn channels must be a non-empty list")
    return MoonshotRunConfig(
        track=str(payload["track"]),
        experiment_name=str(payload["experiment_name"]),
        config_path=str(config_path.resolve()),
        data_root=str(payload["data_root"]),
        output_root=str(payload["output_root"]),
        category_map_path=str(payload["category_map_path"]),
        class_vocab_manifest_path=str(payload["class_vocab_manifest_path"]),
        exact_upstream_regression_path=str(payload["exact_upstream_regression_path"]),
        seed=int(payload["seed"]),
        device=str(payload["device"]),
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        lr=float(payload["lr"]),
        weight_decay=float(payload["weight_decay"]),
        grad_clip=float(payload["grad_clip"]),
        diff_period=int(payload["diff_period"]),
        window_size=int(payload["window_size"]),
        stride=None if payload["stride"] is None else int(payload["stride"]),
        num_workers=int(payload["num_workers"]),
        validation_files_per_class=int(payload["validation_files_per_class"]),
        label_smoothing=float(payload["label_smoothing"]),
        scheduler_name=str(payload["scheduler_name"]),
        scheduler_t_max=int(payload["scheduler_t_max"]),
        scheduler_eta_min=float(payload["scheduler_eta_min"]),
        primary_file_aggregator=str(payload["primary_file_aggregator"]),
        validation_file_aggregator=str(payload["validation_file_aggregator"]),
        model_name=str(payload["model_name"]),
        model=CnnModelConfig(
            channels=tuple(int(channel) for channel in raw_channels),
            kernel_size=int(model_payload["kernel_size"]),
            dropout=float(model_payload["dropout"]),
            use_batchnorm=bool(model_payload["use_batchnorm"]),
        ),
    )


def prepare_moonshot_tensors(
    dataset: Any,
    config: MoonshotRunConfig,
) -> MoonshotPreparedTensors:
    prepared = prepare_moonshot_window_splits(
        dataset,
        diff_period=config.diff_period,
        window_size=config.window_size,
        stride=config.stride,
        validation_files_per_class=config.validation_files_per_class,
    )
    train_values = stack_window_values(prepared.standardized_train_split.windows).astype(
        np.float32,
        copy=False,
    )
    validation_values = stack_window_values(prepared.standardized_validation_split.windows).astype(
        np.float32,
        copy=False,
    )
    test_values = stack_window_values(prepared.standardized_test_split.windows).astype(
        np.float32,
        copy=False,
    )
    return MoonshotPreparedTensors(
        class_names=prepared.class_names,
        feature_names=prepared.feature_names,
        train_windows=train_values,
        train_labels=stack_window_labels(prepared.standardized_train_split, prepared.class_names),
        validation_windows=validation_values,
        validation_labels=stack_window_labels(
            prepared.standardized_validation_split,
            prepared.class_names,
        ),
        test_windows=test_values,
        test_labels=stack_window_labels(prepared.standardized_test_split, prepared.class_names),
        train_window_count=prepared.standardized_train_split.window_count,
        validation_window_count=prepared.standardized_validation_split.window_count,
        test_window_count=prepared.standardized_test_split.window_count,
        train_standardization_shape=(
            prepared.standardizer.sample_count,
            prepared.standardizer.feature_count,
        ),
        standardized_train_split=prepared.standardized_train_split,
        standardized_validation_split=prepared.standardized_validation_split,
        standardized_test_split=prepared.standardized_test_split,
        view_manifest=prepared.view_manifest,
    )


def train_moonshot_classifier(
    *,
    model: nn.Module,
    train_loader: Any,
    validation_loader: Any,
    validation_windows: tuple[Any, ...],
    class_names: tuple[str, ...],
    category_mapping: dict[str, str],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    label_smoothing: float,
    scheduler_name: str,
    scheduler_t_max: int,
    scheduler_eta_min: float,
    validation_file_aggregator: str,
    checkpoint_path: Path,
    config_payload: dict[str, Any],
) -> list[MoonshotHistoryRow]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        scheduler_t_max=scheduler_t_max,
        scheduler_eta_min=scheduler_eta_min,
    )
    best_validation_acc = float("-inf")
    history: list[MoonshotHistoryRow] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_size = batch_y.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total_examples += batch_size

        validation_outputs = collect_evaluation_outputs(
            model=model,
            data_loader=validation_loader,
            device=device,
            class_names=class_names,
            category_mapping=category_mapping,
        )
        validation_file_metrics = aggregate_file_level_metrics(
            bundle=build_window_prediction_bundle(
                class_names=validation_outputs.metrics.class_names,
                true_labels=validation_outputs.true_labels,
                predicted_labels=validation_outputs.predicted_labels,
                topk_indices=validation_outputs.topk_indices,
                logits=validation_outputs.logits,
                windows=validation_windows,
            ),
            category_mapping=category_mapping,
            aggregator=validation_file_aggregator,
        ).metrics
        current_lr = float(optimizer.param_groups[0]["lr"])
        is_best = validation_file_metrics.acc_at_1 > best_validation_acc
        if is_best:
            best_validation_acc = validation_file_metrics.acc_at_1
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config_payload,
                    "best_validation_file_acc@1": validation_file_metrics.acc_at_1,
                    "best_validation_file_acc@5": validation_file_metrics.acc_at_5,
                    "best_validation_file_f1": validation_file_metrics.f1_macro,
                    "best_validation_window_acc@1": validation_outputs.metrics.acc_at_1,
                    "best_validation_window_acc@5": validation_outputs.metrics.acc_at_5,
                    "best_validation_window_f1": validation_outputs.metrics.f1_macro,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        history.append(
            MoonshotHistoryRow(
                epoch=epoch,
                learning_rate=current_lr,
                train_loss=total_loss / max(total_examples, 1),
                train_acc_at_1=100.0 * total_correct / max(total_examples, 1),
                validation_window_acc_at_1=validation_outputs.metrics.acc_at_1,
                validation_window_f1=validation_outputs.metrics.f1_macro,
                validation_file_acc_at_1=validation_file_metrics.acc_at_1,
                validation_file_f1=validation_file_metrics.f1_macro,
                is_best=is_best,
            )
        )
        if scheduler is not None:
            scheduler.step()
    if not checkpoint_path.is_file():
        raise MoonshotRunError("moonshot training did not produce a best checkpoint")
    return history


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    scheduler_t_max: int,
    scheduler_eta_min: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if scheduler_name == "":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_t_max,
            eta_min=scheduler_eta_min,
        )
    raise MoonshotRunError(f"unsupported scheduler_name: {scheduler_name!r}")


def write_training_history(output_path: Path, history: list[MoonshotHistoryRow]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "learning_rate",
                "train_loss",
                "train_acc@1",
                "validation_window_acc@1",
                "validation_window_f1",
                "validation_file_acc@1",
                "validation_file_f1",
                "is_best",
            ]
        )
        for row in history:
            writer.writerow(
                [
                    row.epoch,
                    row.learning_rate,
                    row.train_loss,
                    row.train_acc_at_1,
                    row.validation_window_acc_at_1,
                    row.validation_window_f1,
                    row.validation_file_acc_at_1,
                    row.validation_file_f1,
                    str(row.is_best).lower(),
                ]
            )


def build_file_level_summary_row(
    *,
    run_id: str,
    track: str,
    model_family: str,
    view_mode: str,
    diff_period: int,
    window_size: int,
    stride: int,
    window_metrics: Any,
    file_result: FileLevelAggregationResult,
    report_paths: Any,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "track": track,
        "model_family": model_family,
        "view_mode": view_mode,
        "g": str(diff_period),
        "window_size": str(window_size),
        "stride": str(stride),
        "aggregator": file_result.aggregator,
        "window_acc@1": str(window_metrics.acc_at_1),
        "window_acc@5": str(window_metrics.acc_at_5),
        "window_macro_f1": str(window_metrics.f1_macro),
        "file_acc@1": str(file_result.metrics.acc_at_1),
        "file_acc@5": str(file_result.metrics.acc_at_5),
        "file_macro_precision": str(file_result.metrics.precision_macro),
        "file_macro_recall": str(file_result.metrics.recall_macro),
        "file_macro_f1": str(file_result.metrics.f1_macro),
        "file_summary_metrics_path": report_paths.summary_json,
        "file_confusion_matrix_path": report_paths.confusion_matrix_csv,
        "file_per_category_accuracy_path": report_paths.per_category_accuracy_csv,
        "file_predictions_path": report_paths.per_file_predictions_csv,
    }


def write_dict_rows_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    import json

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
