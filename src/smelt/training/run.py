"""exact-upstream classification runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from smelt.datasets import (
    build_base_class_vocab_manifest,
    load_base_sensor_dataset,
    write_base_class_vocab_manifest,
)
from smelt.evaluation import (
    ClassificationMetrics,
    compute_classification_metrics,
    export_classification_report,
    load_category_mapping,
)
from smelt.models import ExactUpstreamCnnClassifier, ExactUpstreamTransformerClassifier
from smelt.preprocessing import (
    EXACT_UPSTREAM_DROPPED_COLUMNS,
    apply_window_standardizer,
    fit_window_standardizer,
    generate_split_windows,
    preprocess_split_records,
    stack_window_values,
)


class ExactUpstreamRunError(Exception):
    """raised when the exact-upstream transformer run cannot proceed safely."""


@dataclass(slots=True)
class TransformerModelConfig:
    model_dim: int = 256
    num_heads: int = 8
    num_layers: int = 3
    dropout: float = 0.1


@dataclass(slots=True)
class CnnModelConfig:
    channels: tuple[int, ...] = (64, 128, 256)
    kernel_size: int = 5
    dropout: float = 0.2
    use_batchnorm: bool = True


@dataclass(slots=True)
class ExactUpstreamRunConfig:
    track: str
    experiment_name: str
    config_path: str
    data_root: str
    output_root: str
    category_map_path: str
    preprocessing_summary_path: str
    class_vocab_manifest_path: str
    gcms_class_map_manifest_path: str
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
    shuffle_train_labels: bool
    model_name: str
    model: TransformerModelConfig | CnnModelConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = asdict(self.model)
        return payload


@dataclass(slots=True)
class PreparedWindowTensors:
    class_names: tuple[str, ...]
    train_windows: np.ndarray
    train_labels: np.ndarray
    test_windows: np.ndarray
    test_labels: np.ndarray
    train_window_count: int
    test_window_count: int
    train_standardization_shape: tuple[int, int]


@dataclass(slots=True)
class TrainingHistoryRow:
    epoch: int
    train_loss: float
    train_acc_at_1: float


@dataclass(slots=True)
class EvaluationOutputs:
    metrics: ClassificationMetrics
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    topk_indices: np.ndarray
    logits: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir, metrics, checkpoint_path, prepared = run_exact_upstream_transformer(args.config)
    print(f"run_dir: {run_dir}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"acc@1: {metrics.acc_at_1}")
    print(f"acc@5: {metrics.acc_at_5}")
    print(f"precision_macro: {metrics.precision_macro}")
    print(f"recall_macro: {metrics.recall_macro}")
    print(f"f1_macro: {metrics.f1_macro}")
    print(f"train_window_count: {prepared.train_window_count}")
    print(f"test_window_count: {prepared.test_window_count}")
    return 0


def run_exact_upstream_transformer(
    config_path: Path,
) -> tuple[Path, Any, Path, PreparedWindowTensors]:
    config = load_run_config(config_path)
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = build_run_dir(Path(config.output_root), config.experiment_name, config.to_dict())
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_base_sensor_dataset(Path(config.data_root))
    vocab_manifest = build_base_class_vocab_manifest(dataset)
    validate_reference_manifest(config.class_vocab_manifest_path, vocab_manifest.to_dict())
    validate_required_reference(config.gcms_class_map_manifest_path)
    validate_required_reference(config.preprocessing_summary_path)
    category_mapping = load_category_mapping(Path(config.category_map_path))

    prepared = prepare_window_tensors(dataset, config)
    model = build_classifier_model(
        config=config,
        input_dim=prepared.train_windows.shape[2],
        num_classes=len(prepared.class_names),
    ).to(device)

    train_loader = build_dataloader(
        prepared.train_windows,
        maybe_shuffle_train_labels(prepared.train_labels, config),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = build_dataloader(
        prepared.test_windows,
        prepared.test_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    history = train_classifier(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
    )
    checkpoint_path = run_dir / "checkpoint_final.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict()},
        checkpoint_path,
    )

    evaluation = collect_evaluation_outputs(
        model=model,
        data_loader=test_loader,
        device=device,
        class_names=prepared.class_names,
        category_mapping=category_mapping,
    )
    metrics = evaluation.metrics
    export_classification_report(
        output_root=run_dir.parent,
        run_name=run_dir.name,
        metrics=metrics,
        methods_summary={
            "preprocessing_summary_path": str(Path(config.preprocessing_summary_path).resolve()),
            "class_vocab_manifest_path": str(Path(config.class_vocab_manifest_path).resolve()),
            "gcms_class_map_manifest_path": str(
                Path(config.gcms_class_map_manifest_path).resolve()
            ),
            "train_window_count": prepared.train_window_count,
            "test_window_count": prepared.test_window_count,
            "train_standardization_shape": list(prepared.train_standardization_shape),
        },
        overwrite=True,
    )
    write_prediction_bundle(run_dir / "predictions.npz", evaluation)
    write_training_history(run_dir / "training_history.csv", history)
    write_resolved_config(run_dir / "resolved_config.yaml", config)
    write_run_metadata(
        run_dir / "run_metadata.json",
        {
            "track": "exact-upstream",
            "mode": "classification",
            "reference_artifacts": {
                "preprocessing_summary_path": str(
                    Path(config.preprocessing_summary_path).resolve()
                ),
                "class_vocab_manifest_path": str(Path(config.class_vocab_manifest_path).resolve()),
                "gcms_class_map_manifest_path": str(
                    Path(config.gcms_class_map_manifest_path).resolve()
                ),
                "category_map_path": str(Path(config.category_map_path).resolve()),
            },
            "window_counts": {
                "train": prepared.train_window_count,
                "test": prepared.test_window_count,
            },
            "checkpoint_path": str(checkpoint_path.resolve()),
            "shuffle_train_labels": config.shuffle_train_labels,
        },
    )
    write_base_class_vocab_manifest(run_dir / "base_class_vocab.json", vocab_manifest)
    return run_dir, metrics, checkpoint_path, prepared


def load_run_config(config_path: Path) -> ExactUpstreamRunConfig:
    try:
        raw_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ExactUpstreamRunError(f"unable to read config file: {config_path}") from exc
    if not isinstance(raw_payload, dict):
        raise ExactUpstreamRunError("run config must deserialize to a mapping")

    payload = expand_env_values(raw_payload)
    required_keys = {
        "track",
        "experiment_name",
        "data_root",
        "output_root",
        "category_map_path",
        "preprocessing_summary_path",
        "class_vocab_manifest_path",
        "gcms_class_map_manifest_path",
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
        "model",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        raise ExactUpstreamRunError(f"run config is missing required keys: {missing_keys}")
    if payload["track"] != "exact-upstream":
        raise ExactUpstreamRunError(f"unsupported track for this runner: {payload['track']!r}")
    model_payload = payload["model"]
    if not isinstance(model_payload, dict):
        raise ExactUpstreamRunError("model config must be a mapping")
    model_name = resolve_model_name(payload.get("model_name"), model_payload)
    if model_name == "transformer":
        model_config: TransformerModelConfig | CnnModelConfig = TransformerModelConfig(
            model_dim=int(model_payload["model_dim"]),
            num_heads=int(model_payload["num_heads"]),
            num_layers=int(model_payload["num_layers"]),
            dropout=float(model_payload["dropout"]),
        )
    elif model_name == "cnn":
        raw_channels = model_payload["channels"]
        if not isinstance(raw_channels, list) or not raw_channels:
            raise ExactUpstreamRunError("cnn model channels must be a non-empty list")
        model_config = CnnModelConfig(
            channels=tuple(int(channel) for channel in raw_channels),
            kernel_size=int(model_payload["kernel_size"]),
            dropout=float(model_payload["dropout"]),
            use_batchnorm=bool(model_payload["use_batchnorm"]),
        )
    else:
        raise ExactUpstreamRunError(f"unsupported model_name for this runner: {model_name!r}")

    return ExactUpstreamRunConfig(
        track=str(payload["track"]),
        experiment_name=str(payload["experiment_name"]),
        config_path=str(config_path.resolve()),
        data_root=str(payload["data_root"]),
        output_root=str(payload["output_root"]),
        category_map_path=str(payload["category_map_path"]),
        preprocessing_summary_path=str(payload["preprocessing_summary_path"]),
        class_vocab_manifest_path=str(payload["class_vocab_manifest_path"]),
        gcms_class_map_manifest_path=str(payload["gcms_class_map_manifest_path"]),
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
        shuffle_train_labels=bool(payload.get("shuffle_train_labels", False)),
        model_name=model_name,
        model=model_config,
    )


def resolve_model_name(raw_value: Any, model_payload: dict[str, Any]) -> str:
    if raw_value is not None:
        return str(raw_value)
    model_keys = set(model_payload)
    if {"model_dim", "num_heads", "num_layers", "dropout"} <= model_keys:
        return "transformer"
    if {"channels", "kernel_size", "dropout", "use_batchnorm"} <= model_keys:
        return "cnn"
    raise ExactUpstreamRunError(
        "model_name is missing and could not be inferred from the model config keys"
    )


def expand_env_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_values(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [expand_env_values(inner_value) for inner_value in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ExactUpstreamRunError(f"unresolved environment variable in config value: {value}")
        return expanded
    return value


def validate_required_reference(path_value: str) -> None:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ExactUpstreamRunError(f"required reference artifact is missing: {path}")


def validate_reference_manifest(path_value: str, expected_payload: dict[str, Any]) -> None:
    path = Path(path_value).expanduser().resolve()
    validate_required_reference(str(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("class_vocab") != expected_payload.get("class_vocab"):
        raise ExactUpstreamRunError(
            f"class vocab manifest does not match the resolved dataset vocab: {path}"
        )


def prepare_window_tensors(
    dataset: Any,
    config: ExactUpstreamRunConfig,
) -> PreparedWindowTensors:
    class_names = build_base_class_vocab_manifest(dataset).class_vocab
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    train_records = preprocess_split_records(
        dataset.train_records,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=config.diff_period,
    )
    test_records = preprocess_split_records(
        dataset.test_records,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=config.diff_period,
    )
    train_windows = generate_split_windows(
        train_records,
        window_size=config.window_size,
        stride=config.stride,
    )
    test_windows = generate_split_windows(
        test_records,
        window_size=config.window_size,
        stride=config.stride,
    )
    if train_windows.window_count == 0:
        raise ExactUpstreamRunError("train split produced zero windows")
    if test_windows.window_count == 0:
        raise ExactUpstreamRunError("test split produced zero windows")

    standardizer = fit_window_standardizer(train_windows)
    standardized_train = apply_window_standardizer(train_windows, standardizer)
    standardized_test = apply_window_standardizer(test_windows, standardizer)

    train_values = stack_window_values(standardized_train.windows).astype(np.float32, copy=False)
    test_values = stack_window_values(standardized_test.windows).astype(np.float32, copy=False)
    train_labels = np.asarray(
        [class_to_index[window.class_name] for window in standardized_train.windows],
        dtype=np.int64,
    )
    test_labels = np.asarray(
        [class_to_index[window.class_name] for window in standardized_test.windows],
        dtype=np.int64,
    )
    return PreparedWindowTensors(
        class_names=class_names,
        train_windows=train_values,
        train_labels=train_labels,
        test_windows=test_values,
        test_labels=test_labels,
        train_window_count=standardized_train.window_count,
        test_window_count=standardized_test.window_count,
        train_standardization_shape=(
            standardizer.window_count * standardizer.window_size,
            standardizer.feature_count,
        ),
    )


def build_classifier_model(
    *,
    config: ExactUpstreamRunConfig,
    input_dim: int,
    num_classes: int,
) -> nn.Module:
    if config.model_name == "transformer":
        if not isinstance(config.model, TransformerModelConfig):
            raise ExactUpstreamRunError("transformer config is malformed")
        return ExactUpstreamTransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            model_dim=config.model.model_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
        )
    if config.model_name == "cnn":
        if not isinstance(config.model, CnnModelConfig):
            raise ExactUpstreamRunError("cnn config is malformed")
        return ExactUpstreamCnnClassifier(
            in_channels=input_dim,
            num_classes=num_classes,
            channels=config.model.channels,
            kernel_size=config.model.kernel_size,
            dropout=config.model.dropout,
            use_batchnorm=config.model.use_batchnorm,
        )
    raise ExactUpstreamRunError(f"unsupported model_name for build: {config.model_name!r}")


def maybe_shuffle_train_labels(
    labels: np.ndarray,
    config: ExactUpstreamRunConfig,
) -> np.ndarray:
    labels_copy = labels.copy()
    if not config.shuffle_train_labels:
        return labels_copy
    rng = np.random.default_rng(config.seed)
    rng.shuffle(labels_copy)
    return labels_copy


def build_dataloader(
    windows: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[Any]:
    dataset = TensorDataset(torch.from_numpy(windows), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_classifier(
    *,
    model: nn.Module,
    train_loader: DataLoader[Any],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
) -> list[TrainingHistoryRow]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history: list[TrainingHistoryRow] = []
    for epoch in range(1, epochs + 1):
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
        history.append(
            TrainingHistoryRow(
                epoch=epoch,
                train_loss=total_loss / max(total_examples, 1),
                train_acc_at_1=100.0 * total_correct / max(total_examples, 1),
            )
        )
    return history


def evaluate_classifier(
    *,
    model: nn.Module,
    data_loader: DataLoader[Any],
    device: torch.device,
    class_names: tuple[str, ...],
    category_mapping: dict[str, str],
) -> Any:
    return collect_evaluation_outputs(
        model=model,
        data_loader=data_loader,
        device=device,
        class_names=class_names,
        category_mapping=category_mapping,
    ).metrics


def collect_evaluation_outputs(
    *,
    model: nn.Module,
    data_loader: DataLoader[Any],
    device: torch.device,
    class_names: tuple[str, ...],
    category_mapping: dict[str, str],
) -> EvaluationOutputs:
    model.eval()
    predicted_labels: list[np.ndarray] = []
    topk_indices: list[np.ndarray] = []
    true_labels: list[np.ndarray] = []
    logits_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            logits = model(batch_x)
            logits_cpu = logits.cpu().numpy()
            predicted = logits.argmax(dim=1).cpu().numpy()
            topk = torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices.cpu().numpy()
            predicted_labels.append(predicted)
            topk_indices.append(topk)
            true_labels.append(batch_y.numpy())
            logits_rows.append(logits_cpu)
    true_array = np.concatenate(true_labels, axis=0)
    predicted_array = np.concatenate(predicted_labels, axis=0)
    topk_array = np.concatenate(topk_indices, axis=0)
    logits_array = np.concatenate(logits_rows, axis=0)
    metrics = compute_classification_metrics(
        class_names=class_names,
        true_labels=true_array,
        predicted_labels=predicted_array,
        topk_indices=topk_array,
        category_mapping=category_mapping,
    )
    return EvaluationOutputs(
        metrics=metrics,
        true_labels=true_array,
        predicted_labels=predicted_array,
        topk_indices=topk_array,
        logits=logits_array,
    )


def write_training_history(output_path: Path, history: list[TrainingHistoryRow]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc@1"])
        for row in history:
            writer.writerow([row.epoch, row.train_loss, row.train_acc_at_1])


def write_resolved_config(output_path: Path, config: ExactUpstreamRunConfig) -> None:
    output_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=True),
        encoding="utf-8",
    )


def write_run_metadata(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_prediction_bundle(output_path: Path, evaluation: EvaluationOutputs) -> None:
    np.savez_compressed(
        output_path,
        class_names=np.asarray(evaluation.metrics.class_names),
        true_labels=evaluation.true_labels,
        predicted_labels=evaluation.predicted_labels,
        topk_indices=evaluation.topk_indices,
        logits=evaluation.logits,
    )


def build_run_dir(output_root: Path, experiment_name: str, config_payload: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config_hash = hashlib.sha1(
        json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:8]
    return output_root / f"{experiment_name}-{timestamp}-{config_hash}"


def resolve_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ExactUpstreamRunError("config requested cuda but it is unavailable")
        return torch.device("cuda")
    if normalized == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ExactUpstreamRunError("config requested mps but it is unavailable")
        return torch.device("mps")
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ExactUpstreamRunError(f"unsupported device setting: {device_name!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
