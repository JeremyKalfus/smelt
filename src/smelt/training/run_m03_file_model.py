"""train a frozen-encoder file-level model for m03."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from smelt.evaluation import (
    build_file_level_result_from_predictions,
    export_classification_report,
    export_file_level_report,
    load_category_mapping,
    write_dict_rows_csv,
    write_json,
)
from smelt.models import AttentionDeepSetsClassifier

from .m03 import build_file_groups, load_window_feature_bundle
from .run import (
    build_run_dir,
    resolve_device,
    set_seed,
    validate_required_reference,
    write_resolved_config,
    write_run_metadata,
)


class M03FileModelRunError(Exception):
    """raised when the m03 learned file-level model cannot proceed safely."""


@dataclass(slots=True)
class M03FileModelConfig:
    track: str
    experiment_name: str
    config_path: str
    output_root: str
    category_map_path: str
    class_vocab_manifest_path: str
    exact_upstream_regression_path: str
    frozen_encoder_run_id: str
    frozen_embedding_bundle_path: str
    frozen_encoder_selection_path: str
    protocol_definition_path: str
    seed: int
    device: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float
    scheduler_name: str
    scheduler_t_max: int
    scheduler_eta_min: float
    label_smoothing: float
    channel_set: str
    diff_period: int
    window_size: int
    stride: int
    view_mode: str
    model_name: str
    model: AttentionDeepSetsConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = asdict(self.model)
        return payload


@dataclass(slots=True)
class AttentionDeepSetsConfig:
    hidden_dim: int
    dropout: float


@dataclass(slots=True)
class FileGroupExample:
    split: str
    relative_path: str
    absolute_path: str
    true_label: int
    num_windows: int
    features: np.ndarray


@dataclass(slots=True)
class FileModelHistoryRow:
    epoch: int
    learning_rate: float
    train_loss: float
    validation_file_acc_at_1: float
    validation_file_f1: float
    is_best: bool


@dataclass(slots=True)
class FileCheckpointSelection:
    epoch: int
    checkpoint_path: Path
    validation_file_acc_at_1: float
    validation_file_acc_at_5: float
    validation_file_f1: float


class FileGroupDataset(Dataset[FileGroupExample]):
    def __init__(self, examples: tuple[FileGroupExample, ...]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> FileGroupExample:
        return self.examples[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir, metrics, best_checkpoint = run_m03_file_model(args.config)
    print(f"run_dir: {run_dir}")
    print(f"checkpoint_path: {best_checkpoint}")
    print(f"file_acc@1: {metrics.acc_at_1}")
    print(f"file_acc@5: {metrics.acc_at_5}")
    print(f"file_precision_macro: {metrics.precision_macro}")
    print(f"file_recall_macro: {metrics.recall_macro}")
    print(f"file_f1_macro: {metrics.f1_macro}")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    print(f"frozen_encoder_run_id: {metadata.get('encoder_source_run_id', '')}")
    print(f"file_level_model_family: {metadata.get('file_level_model_family', '')}")
    print(f"encoder_frozen: {metadata.get('encoder_frozen', '')}")
    print(f"device: {metadata.get('device', '')}")
    return 0


def run_m03_file_model(config_path: Path) -> tuple[Path, Any, Path]:
    config = load_m03_file_model_config(config_path)
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = build_run_dir(Path(config.output_root), config.experiment_name, config.to_dict())
    run_dir.mkdir(parents=True, exist_ok=False)

    validate_required_reference(config.exact_upstream_regression_path)
    validate_required_reference(config.class_vocab_manifest_path)
    validate_required_reference(config.frozen_encoder_selection_path)
    validate_required_reference(config.protocol_definition_path)
    category_mapping = load_category_mapping(Path(config.category_map_path))
    embedding_bundle = load_window_feature_bundle(Path(config.frozen_embedding_bundle_path))
    train_examples = tuple(build_group_examples(embedding_bundle, split_name="train"))
    validation_examples = tuple(build_group_examples(embedding_bundle, split_name="validation"))
    test_examples = tuple(build_group_examples(embedding_bundle, split_name="test"))
    if not train_examples or not validation_examples or not test_examples:
        raise M03FileModelRunError("frozen embedding bundle did not provide all required splits")

    model = AttentionDeepSetsClassifier(
        input_dim=int(embedding_bundle.embedding_dim),
        hidden_dim=config.model.hidden_dim,
        num_classes=len(embedding_bundle.class_names),
        dropout=config.model.dropout,
    ).to(device)

    train_loader = DataLoader(
        FileGroupDataset(train_examples),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_file_group_batch,
    )
    validation_loader = DataLoader(
        FileGroupDataset(validation_examples),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_file_group_batch,
    )
    test_loader = DataLoader(
        FileGroupDataset(test_examples),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_file_group_batch,
    )

    history, best_selection = train_file_level_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        class_names=embedding_bundle.class_names,
        category_mapping=category_mapping,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        scheduler_name=config.scheduler_name,
        scheduler_t_max=config.scheduler_t_max,
        scheduler_eta_min=config.scheduler_eta_min,
        label_smoothing=config.label_smoothing,
        checkpoint_dir=run_dir / "checkpoint_candidates",
        config_payload=config.to_dict(),
    )
    best_checkpoint = torch.load(best_selection.checkpoint_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    best_checkpoint_path = run_dir / "checkpoint_best.pt"
    torch.save(best_checkpoint, best_checkpoint_path)
    final_checkpoint_path = run_dir / "checkpoint_final.pt"
    torch.save(best_checkpoint, final_checkpoint_path)

    validation_result = evaluate_file_level_model(
        model=model,
        data_loader=validation_loader,
        device=device,
        class_names=embedding_bundle.class_names,
        category_mapping=category_mapping,
        aggregator=config.model_name,
    )
    validation_report = export_file_level_report(
        output_root=run_dir / "validation_file_level",
        run_name=config.model_name,
        result=validation_result,
        methods_summary={
            "track": config.track,
            "mode": "file_level_classification",
            "view_mode": config.view_mode,
            "channel_set": config.channel_set,
            "encoder_source_run_id": config.frozen_encoder_run_id,
            "encoder_frozen": True,
        },
    )
    test_result = evaluate_file_level_model(
        model=model,
        data_loader=test_loader,
        device=device,
        class_names=embedding_bundle.class_names,
        category_mapping=category_mapping,
        aggregator=config.model_name,
    )
    export_classification_report(
        output_root=run_dir.parent,
        run_name=run_dir.name,
        metrics=test_result.metrics,
        methods_summary={
            "track": config.track,
            "mode": "file_level_classification",
            "view_mode": config.view_mode,
            "channel_set": config.channel_set,
            "encoder_source_run_id": config.frozen_encoder_run_id,
            "encoder_frozen": True,
            "primary_checkpoint_selection_metric": (
                "validation_file_acc@1_then_validation_file_macro_f1"
            ),
            "metric_scope": "file_level",
        },
        overwrite=True,
    )
    test_report = export_file_level_report(
        output_root=run_dir / "file_level",
        run_name=config.model_name,
        result=test_result,
        methods_summary={
            "track": config.track,
            "mode": "file_level_classification",
            "view_mode": config.view_mode,
            "channel_set": config.channel_set,
            "encoder_source_run_id": config.frozen_encoder_run_id,
            "encoder_frozen": True,
        },
    )
    comparison_rows = [
        build_file_level_row(
            run_id=run_dir.name,
            config=config,
            split_name="validation",
            metrics=validation_result.metrics,
            report_paths=validation_report.to_dict(),
        ),
        build_file_level_row(
            run_id=run_dir.name,
            config=config,
            split_name="test",
            metrics=test_result.metrics,
            report_paths=test_report.to_dict(),
        ),
    ]
    write_dict_rows_csv(
        run_dir / "validation_file_level_metrics_comparison.csv",
        [comparison_rows[0]],
    )
    write_json(
        run_dir / "validation_file_level_metrics_comparison.json",
        {"rows": [comparison_rows[0]]},
    )
    write_dict_rows_csv(run_dir / "file_level_metrics_comparison.csv", [comparison_rows[1]])
    write_json(run_dir / "file_level_metrics_comparison.json", {"rows": [comparison_rows[1]]})
    write_training_history(run_dir / "training_history.csv", history)
    write_resolved_config(run_dir / "resolved_config.yaml", config)
    write_run_metadata(
        run_dir / "run_metadata.json",
        {
            "track": config.track,
            "mode": "file_level_classification",
            "view_mode": config.view_mode,
            "channel_set": config.channel_set,
            "encoder_source_run_id": config.frozen_encoder_run_id,
            "encoder_source_bundle_path": str(Path(config.frozen_embedding_bundle_path).resolve()),
            "encoder_frozen": True,
            "file_level_model_family": config.model_name,
            "device": str(device),
            "batch_size": config.batch_size,
            "parameter_count": int(
                sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                )
            ),
            "window_counts": {
                "train": int(sum(example.num_windows for example in train_examples)),
                "validation": int(sum(example.num_windows for example in validation_examples)),
                "test": int(sum(example.num_windows for example in test_examples)),
            },
            "file_counts": {
                "train": len(train_examples),
                "validation": len(validation_examples),
                "test": len(test_examples),
            },
            "locked_primary_aggregator": config.model_name,
            "primary_checkpoint_selection_metric": (
                "validation_file_acc@1_then_validation_file_macro_f1"
            ),
            "aggregator_selection_source": "validation_only",
            "best_validation_summary": {
                "epoch": int(best_checkpoint.get("epoch", 0)),
                "file_acc@1": float(best_checkpoint.get("best_validation_file_acc@1", 0.0)),
                "file_acc@5": float(best_checkpoint.get("best_validation_file_acc@5", 0.0)),
                "file_macro_f1": float(best_checkpoint.get("best_validation_file_f1", 0.0)),
            },
            "checkpoint_path": str(final_checkpoint_path.resolve()),
            "best_checkpoint_path": str(best_checkpoint_path.resolve()),
            "file_level_primary_report": test_report.to_dict(),
            "validation_file_level_primary_report": validation_report.to_dict(),
            "reference_artifacts": {
                "category_map_path": str(Path(config.category_map_path).resolve()),
                "class_vocab_manifest_path": str(Path(config.class_vocab_manifest_path).resolve()),
                "exact_upstream_regression_path": str(
                    Path(config.exact_upstream_regression_path).resolve()
                ),
                "frozen_encoder_selection_path": str(
                    Path(config.frozen_encoder_selection_path).resolve()
                ),
                "protocol_definition_path": str(Path(config.protocol_definition_path).resolve()),
            },
        },
    )
    shutil.copyfile(config.class_vocab_manifest_path, run_dir / "base_class_vocab.json")
    return run_dir, test_result.metrics, best_checkpoint_path


def load_m03_file_model_config(config_path: Path) -> M03FileModelConfig:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise M03FileModelRunError(f"unable to read config file: {config_path}") from exc
    if not isinstance(payload, dict):
        raise M03FileModelRunError("m03 file model config must deserialize to a mapping")
    required = {
        "track",
        "experiment_name",
        "output_root",
        "category_map_path",
        "class_vocab_manifest_path",
        "exact_upstream_regression_path",
        "frozen_encoder_run_id",
        "frozen_embedding_bundle_path",
        "frozen_encoder_selection_path",
        "protocol_definition_path",
        "seed",
        "device",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "scheduler_name",
        "scheduler_t_max",
        "scheduler_eta_min",
        "label_smoothing",
        "channel_set",
        "diff_period",
        "window_size",
        "stride",
        "view_mode",
        "model_name",
        "model",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise M03FileModelRunError(f"m03 file model config is missing required keys: {missing}")
    if payload["track"] != "moonshot-enhanced-setting":
        raise M03FileModelRunError(f"unsupported track {payload['track']!r}")
    if payload["model_name"] != "attention_deepsets":
        raise M03FileModelRunError("m03 file model only supports model_name=attention_deepsets")
    model_payload = payload["model"]
    if not isinstance(model_payload, dict):
        raise M03FileModelRunError("m03 file model config 'model' must be a mapping")
    return M03FileModelConfig(
        track=str(payload["track"]),
        experiment_name=str(payload["experiment_name"]),
        config_path=str(config_path.resolve()),
        output_root=str(payload["output_root"]),
        category_map_path=str(payload["category_map_path"]),
        class_vocab_manifest_path=str(payload["class_vocab_manifest_path"]),
        exact_upstream_regression_path=str(payload["exact_upstream_regression_path"]),
        frozen_encoder_run_id=str(payload["frozen_encoder_run_id"]),
        frozen_embedding_bundle_path=str(payload["frozen_embedding_bundle_path"]),
        frozen_encoder_selection_path=str(payload["frozen_encoder_selection_path"]),
        protocol_definition_path=str(payload["protocol_definition_path"]),
        seed=int(payload["seed"]),
        device=str(payload["device"]),
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        lr=float(payload["lr"]),
        weight_decay=float(payload["weight_decay"]),
        grad_clip=float(payload["grad_clip"]),
        scheduler_name=str(payload["scheduler_name"]),
        scheduler_t_max=int(payload["scheduler_t_max"]),
        scheduler_eta_min=float(payload["scheduler_eta_min"]),
        label_smoothing=float(payload["label_smoothing"]),
        channel_set=str(payload["channel_set"]),
        diff_period=int(payload["diff_period"]),
        window_size=int(payload["window_size"]),
        stride=int(payload["stride"]),
        view_mode=str(payload["view_mode"]),
        model_name=str(payload["model_name"]),
        model=AttentionDeepSetsConfig(
            hidden_dim=int(model_payload["hidden_dim"]),
            dropout=float(model_payload["dropout"]),
        ),
    )


def build_group_examples(
    embedding_bundle: Any,
    *,
    split_name: str,
) -> list[FileGroupExample]:
    return [
        FileGroupExample(
            split=str(group["split"]),
            relative_path=str(group["relative_path"]),
            absolute_path=str(group["absolute_path"]),
            true_label=int(group["true_label"]),
            num_windows=int(group["num_windows"]),
            features=np.asarray(group["features"], dtype=np.float32),
        )
        for group in build_file_groups(embedding_bundle, split_name=split_name)
    ]


def collate_file_group_batch(
    batch: list[FileGroupExample],
) -> dict[str, Any]:
    max_windows = max(example.features.shape[0] for example in batch)
    feature_dim = batch[0].features.shape[1]
    features = torch.zeros(len(batch), max_windows, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_windows, dtype=torch.bool)
    labels = torch.zeros(len(batch), dtype=torch.long)
    relative_paths: list[str] = []
    absolute_paths: list[str] = []
    num_windows: list[int] = []
    splits: list[str] = []
    for index, example in enumerate(batch):
        window_count = int(example.features.shape[0])
        features[index, :window_count] = torch.from_numpy(example.features)
        mask[index, :window_count] = True
        labels[index] = int(example.true_label)
        relative_paths.append(example.relative_path)
        absolute_paths.append(example.absolute_path)
        num_windows.append(example.num_windows)
        splits.append(example.split)
    return {
        "features": features,
        "mask": mask,
        "labels": labels,
        "relative_paths": tuple(relative_paths),
        "absolute_paths": tuple(absolute_paths),
        "num_windows": np.asarray(num_windows, dtype=np.int64),
        "splits": tuple(splits),
    }


def train_file_level_model(
    *,
    model: AttentionDeepSetsClassifier,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    device: torch.device,
    class_names: tuple[str, ...],
    category_mapping: dict[str, str],
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    scheduler_name: str,
    scheduler_t_max: int,
    scheduler_eta_min: float,
    label_smoothing: float,
    checkpoint_dir: Path,
    config_payload: dict[str, Any],
) -> tuple[list[FileModelHistoryRow], FileCheckpointSelection]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        scheduler_t_max=scheduler_t_max,
        scheduler_eta_min=scheduler_eta_min,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[FileModelHistoryRow] = []
    best_selection: FileCheckpointSelection | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(features, mask)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
        validation_result = evaluate_file_level_model(
            model=model,
            data_loader=validation_loader,
            device=device,
            class_names=class_names,
            category_mapping=category_mapping,
            aggregator="attention_deepsets",
        )
        candidate = FileCheckpointSelection(
            epoch=epoch,
            checkpoint_path=checkpoint_dir / "attention_deepsets.pt",
            validation_file_acc_at_1=validation_result.metrics.acc_at_1,
            validation_file_acc_at_5=validation_result.metrics.acc_at_5,
            validation_file_f1=validation_result.metrics.f1_macro,
        )
        is_best = is_better_file_checkpoint(candidate, best_selection)
        if is_best:
            best_selection = candidate
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config_payload,
                    "epoch": epoch,
                    "best_validation_file_acc@1": candidate.validation_file_acc_at_1,
                    "best_validation_file_acc@5": candidate.validation_file_acc_at_5,
                    "best_validation_file_f1": candidate.validation_file_f1,
                },
                candidate.checkpoint_path,
            )
        history.append(
            FileModelHistoryRow(
                epoch=epoch,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                train_loss=total_loss / max(total_examples, 1),
                validation_file_acc_at_1=validation_result.metrics.acc_at_1,
                validation_file_f1=validation_result.metrics.f1_macro,
                is_best=is_best,
            )
        )
        if scheduler is not None:
            scheduler.step()
    if best_selection is None or not best_selection.checkpoint_path.is_file():
        raise M03FileModelRunError("m03 learned file model did not produce a best checkpoint")
    return history, best_selection


def evaluate_file_level_model(
    *,
    model: AttentionDeepSetsClassifier,
    data_loader: DataLoader[Any],
    device: torch.device,
    class_names: tuple[str, ...],
    category_mapping: dict[str, str],
    aggregator: str,
) -> Any:
    model.eval()
    logits_batches: list[np.ndarray] = []
    labels_batches: list[np.ndarray] = []
    split_names: list[str] = []
    relative_paths: list[str] = []
    absolute_paths: list[str] = []
    num_windows: list[int] = []
    with torch.no_grad():
        for batch in data_loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            mask = batch["mask"].to(device=device)
            logits = model(features, mask)
            logits_batches.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
            labels_batches.append(batch["labels"].numpy().astype(np.int64, copy=False))
            split_names.extend(batch["splits"])
            relative_paths.extend(batch["relative_paths"])
            absolute_paths.extend(batch["absolute_paths"])
            num_windows.extend(batch["num_windows"].tolist())
    logits = np.concatenate(logits_batches, axis=0)
    true_labels = np.concatenate(labels_batches, axis=0)
    topk_indices = stable_topk_per_row(logits)
    predicted_labels = np.asarray(topk_indices[:, 0], dtype=np.int64)
    return build_file_level_result_from_predictions(
        aggregator=aggregator,
        class_names=class_names,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        topk_indices=topk_indices,
        split_names=tuple(split_names),
        relative_paths=tuple(relative_paths),
        absolute_paths=tuple(absolute_paths),
        num_windows=np.asarray(num_windows, dtype=np.int64),
        category_mapping=category_mapping,
    )


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
    raise M03FileModelRunError(f"unsupported scheduler_name {scheduler_name!r}")


def is_better_file_checkpoint(
    candidate: FileCheckpointSelection,
    incumbent: FileCheckpointSelection | None,
) -> bool:
    if incumbent is None:
        return True
    if candidate.validation_file_acc_at_1 > incumbent.validation_file_acc_at_1:
        return True
    if candidate.validation_file_acc_at_1 < incumbent.validation_file_acc_at_1:
        return False
    if candidate.validation_file_f1 > incumbent.validation_file_f1:
        return True
    return False


def write_training_history(output_path: Path, history: list[FileModelHistoryRow]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "learning_rate",
                "train_loss",
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
                    row.validation_file_acc_at_1,
                    row.validation_file_f1,
                    str(row.is_best).lower(),
                ]
            )


def build_file_level_row(
    *,
    run_id: str,
    config: M03FileModelConfig,
    split_name: str,
    metrics: Any,
    report_paths: dict[str, str],
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "track": config.track,
        "encoder_source_run_id": config.frozen_encoder_run_id,
        "file_level_model_family": config.model_name,
        "encoder_frozen": "true",
        "selection_rules": "validation_only_checkpoint_selection",
        "split": split_name,
        "channel_set": config.channel_set,
        "view_mode": config.view_mode,
        "g": str(config.diff_period),
        "window_size": str(config.window_size),
        "stride": str(config.stride),
        "file_acc@1": str(metrics.acc_at_1),
        "file_acc@5": str(metrics.acc_at_5),
        "file_macro_precision": str(metrics.precision_macro),
        "file_macro_recall": str(metrics.recall_macro),
        "file_macro_f1": str(metrics.f1_macro),
        "summary_json": report_paths.get("summary_json", ""),
        "confusion_matrix_csv": report_paths.get("confusion_matrix_csv", ""),
        "per_category_accuracy_csv": report_paths.get("per_category_accuracy_csv", ""),
        "per_file_predictions_csv": report_paths.get("per_file_predictions_csv", ""),
    }


def stable_topk_per_row(logits: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.argsort(-row, kind="mergesort")[: min(5, row.shape[0])] for row in logits],
        axis=0,
    ).astype(np.int64, copy=False)


if __name__ == "__main__":
    raise SystemExit(main())
