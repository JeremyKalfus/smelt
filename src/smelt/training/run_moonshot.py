"""moonshot-enhanced-setting cnn runner with grouped file validation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from smelt.datasets import (
    MOONSHOT_ALL12_CHANNEL_SET,
    MOONSHOT_TRACK,
    build_base_class_vocab_manifest,
    load_base_sensor_dataset,
    prepare_moonshot_window_splits,
    resolve_moonshot_channel_set,
    stack_window_labels,
    write_base_class_vocab_manifest,
    write_moonshot_view_manifest,
)
from smelt.evaluation import (
    FILE_LEVEL_AGGREGATORS,
    AggregatorSelectionCandidate,
    FileLevelAggregationResult,
    aggregate_file_level_metrics,
    build_window_prediction_bundle,
    export_classification_report,
    export_file_level_report,
    load_category_mapping,
    normalize_aggregator_candidates,
    select_validation_locked_aggregator,
    write_window_prediction_bundle,
)
from smelt.models import (
    DeepTemporalResNet1D,
    ExactResearchInceptionClassifier,
    ExactUpstreamCnnClassifier,
    TemporalPatchTransformerClassifier,
    build_inception_model_summary,
)
from smelt.preprocessing import stack_window_values

from .run import (
    CnnModelConfig,
    build_dataloader,
    build_run_dir,
    collect_evaluation_outputs,
    expand_env_values,
    maybe_shuffle_train_labels,
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
    shuffle_train_labels: bool
    diff_period: int
    window_size: int
    stride: int | None
    num_workers: int
    validation_files_per_class: int
    label_smoothing: float
    scheduler_name: str
    scheduler_t_max: int
    scheduler_eta_min: float
    gradient_accumulation_steps: int
    locked_protocol: bool
    candidate_file_aggregators: tuple[str, ...]
    primary_file_aggregator: str
    validation_file_aggregator: str
    channel_set: str
    model_name: str
    model: (
        CnnModelConfig
        | TemporalResNetModelConfig
        | HInceptionModelConfig
        | PatchTransformerModelConfig
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = asdict(self.model)
        return payload


@dataclass(slots=True)
class MoonshotPreparedTensors:
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    channel_set: str
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
    validation_primary_aggregator: str
    is_best: bool


@dataclass(slots=True)
class AggregatorCheckpointSelection:
    aggregator: str
    epoch: int
    checkpoint_path: Path
    validation_file_acc_at_1: float
    validation_file_acc_at_5: float
    validation_file_f1: float
    validation_window_acc_at_1: float
    validation_window_acc_at_5: float
    validation_window_f1: float


@dataclass(slots=True)
class TemporalResNetModelConfig:
    stage_depths: tuple[int, ...]
    stage_widths: tuple[int, ...]
    stem_width: int
    kernel_size: int
    normalization: str
    groupnorm_groups: int
    se_reduction: int
    head_dropout: float
    stochastic_depth_probability: float


@dataclass(slots=True)
class HInceptionModelConfig:
    stem_channels: int
    branch_channels: int
    bottleneck_channels: int
    num_blocks: int
    residual_interval: int
    activation_name: str
    dropout: float
    head_hidden_dim: int


@dataclass(slots=True)
class PatchTransformerModelConfig:
    patch_size: int
    patch_stride: int
    model_dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: float
    dropout: float


@dataclass(slots=True)
class MoonshotDeviceSmokeResult:
    device: str
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    feature_count: int
    parameter_count: int
    model_family: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device-smoke-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.device_smoke_only:
        smoke = run_moonshot_device_smoke(args.config)
        print("status: ok")
        print(f"device: {smoke.device}")
        print(f"batch_size: {smoke.batch_size}")
        print(f"gradient_accumulation_steps: {smoke.gradient_accumulation_steps}")
        print(f"effective_batch_size: {smoke.effective_batch_size}")
        print(f"feature_count: {smoke.feature_count}")
        print(f"parameter_count: {smoke.parameter_count}")
        print(f"model_family: {smoke.model_family}")
        return 0
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
    print(f"channel_set: {prepared.channel_set}")
    metadata = run_dir / "run_metadata.json"
    if metadata.is_file():
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        print(f"locked_primary_aggregator: {payload.get('locked_primary_aggregator', '')}")
        print(f"device: {payload.get('device', '')}")
        print(f"parameter_count: {payload.get('parameter_count', '')}")
        print(f"gradient_accumulation_steps: {payload.get('gradient_accumulation_steps', '')}")
        print(f"effective_batch_size: {payload.get('effective_batch_size', '')}")
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
    model, architecture_summary = build_moonshot_model(
        config=config,
        input_dim=prepared.train_windows.shape[2],
        num_classes=len(prepared.class_names),
    )
    model = model.to(device)

    train_loader = build_dataloader(
        prepared.train_windows,
        maybe_shuffle_train_labels(prepared.train_labels, config),
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

    checkpoint_dir = run_dir / "checkpoint_candidates"
    training_result = train_moonshot_classifier(
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
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        locked_protocol=config.locked_protocol,
        candidate_file_aggregators=config.candidate_file_aggregators,
        validation_file_aggregator=config.validation_file_aggregator,
        checkpoint_dir=checkpoint_dir,
        config_payload=config.to_dict(),
    )
    history = training_result["history"]
    best_selection = training_result["best_selection"]
    locked_primary_aggregator = training_result["locked_primary_aggregator"]

    best_checkpoint_path = run_dir / "checkpoint_best.pt"
    best_checkpoint = torch.load(best_selection.checkpoint_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    torch.save(best_checkpoint, best_checkpoint_path)
    checkpoint_path = run_dir / "checkpoint_final.pt"
    torch.save(best_checkpoint, checkpoint_path)

    validation_evaluation = collect_evaluation_outputs(
        model=model,
        data_loader=validation_loader,
        device=device,
        class_names=prepared.class_names,
        category_mapping=category_mapping,
    )
    validation_bundle = build_window_prediction_bundle(
        class_names=validation_evaluation.metrics.class_names,
        true_labels=validation_evaluation.true_labels,
        predicted_labels=validation_evaluation.predicted_labels,
        topk_indices=validation_evaluation.topk_indices,
        logits=validation_evaluation.logits,
        windows=prepared.standardized_validation_split.windows,
    )
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
            "view_mode": f"diff_{config.channel_set}",
            "channel_set": config.channel_set,
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
            "locked_protocol": config.locked_protocol,
            "candidate_file_aggregators": list(config.candidate_file_aggregators),
            "primary_file_aggregator": locked_primary_aggregator,
            "validation_file_aggregator": locked_primary_aggregator,
            "primary_checkpoint_selection_metric": (
                "validation_file_acc@1_then_validation_file_macro_f1"
            ),
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

    validation_file_level_rows: list[dict[str, str]] = []
    file_level_rows: list[dict[str, str]] = []
    primary_file_metrics = None
    primary_paths: dict[str, str] | None = None
    validation_paths: dict[str, dict[str, str]] = {}
    for aggregator in config.candidate_file_aggregators:
        validation_result = aggregate_file_level_metrics(
            bundle=validation_bundle,
            category_mapping=category_mapping,
            aggregator=aggregator,
        )
        validation_report_paths = export_file_level_report(
            output_root=run_dir / "validation_file_level",
            run_name=aggregator,
            result=validation_result,
            methods_summary={
                "track": MOONSHOT_TRACK,
                "split": "validation",
                "view_mode": f"diff_{config.channel_set}",
                "channel_set": config.channel_set,
                "setting_note": MOONSHOT_TRACK,
                "locked_primary_aggregator": locked_primary_aggregator,
                "primary_checkpoint_selection_metric": (
                    "validation_file_acc@1_then_validation_file_macro_f1"
                ),
            },
        )
        validation_paths[aggregator] = validation_report_paths.to_dict()
        validation_file_level_rows.append(
            build_file_level_summary_row(
                run_id=run_dir.name,
                track=MOONSHOT_TRACK,
                model_family=config.model_name,
                view_mode=f"diff_{config.channel_set}",
                channel_set=config.channel_set,
                diff_period=config.diff_period,
                window_size=config.window_size,
                stride=prepared.standardized_validation_split.stride,
                window_metrics=validation_evaluation.metrics,
                file_result=validation_result,
                report_paths=validation_report_paths,
            ),
        )
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
                "view_mode": f"diff_{config.channel_set}",
                "channel_set": config.channel_set,
                "setting_note": MOONSHOT_TRACK,
                "locked_primary_aggregator": locked_primary_aggregator,
                "primary_checkpoint_selection_metric": (
                    "validation_file_acc@1_then_validation_file_macro_f1"
                ),
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
                view_mode=f"diff_{config.channel_set}",
                channel_set=config.channel_set,
                diff_period=config.diff_period,
                window_size=config.window_size,
                stride=prepared.standardized_test_split.stride,
                window_metrics=test_evaluation.metrics,
                file_result=file_result,
                report_paths=report_paths,
            ),
        )
        if aggregator == locked_primary_aggregator:
            primary_file_metrics = file_result.metrics
            primary_paths = report_paths.to_dict()
    if primary_file_metrics is None or primary_paths is None:
        raise MoonshotRunError("primary file aggregator report was not produced")

    file_level_csv = run_dir / "file_level_metrics_comparison.csv"
    file_level_json = run_dir / "file_level_metrics_comparison.json"
    validation_file_level_csv = run_dir / "validation_file_level_metrics_comparison.csv"
    validation_file_level_json = run_dir / "validation_file_level_metrics_comparison.json"
    selection_json = run_dir / "locked_aggregator_selection.json"
    validation_split_json = run_dir / "validation_split.json"
    write_dict_rows_csv(file_level_csv, file_level_rows)
    write_json(file_level_json, {"rows": file_level_rows})
    write_dict_rows_csv(validation_file_level_csv, validation_file_level_rows)
    write_json(validation_file_level_json, {"rows": validation_file_level_rows})
    write_json(
        selection_json,
        {
            "locked_protocol": config.locked_protocol,
            "candidate_file_aggregators": list(config.candidate_file_aggregators),
            "locked_primary_aggregator": locked_primary_aggregator,
            "primary_checkpoint_selection_metric": (
                "validation_file_acc@1_then_validation_file_macro_f1"
            ),
            "best_validation_candidates": [
                {
                    "aggregator": selection["aggregator"],
                    "epoch": int(selection["epoch"]),
                    "validation_file_acc@1": float(selection["validation_file_acc_at_1"]),
                    "validation_file_acc@5": float(selection["validation_file_acc_at_5"]),
                    "validation_file_macro_f1": float(selection["validation_file_f1"]),
                    "validation_window_acc@1": float(selection["validation_window_acc_at_1"]),
                    "validation_window_acc@5": float(selection["validation_window_acc_at_5"]),
                    "validation_window_macro_f1": float(selection["validation_window_f1"]),
                    "checkpoint_path": selection["checkpoint_path"],
                }
                for selection in training_result["best_candidates"]
            ],
        },
    )
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
            "model_family": config.model_name,
            "view_mode": f"diff_{config.channel_set}",
            "channel_set": config.channel_set,
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
            "device": str(device),
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
            "parameter_count": int(architecture_summary["parameter_count"]),
            "architecture_summary_path": str((run_dir / "architecture_summary.json").resolve()),
            "feature_names": list(prepared.feature_names),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "best_checkpoint_path": str(best_checkpoint_path.resolve()),
            "validation_files_per_class": config.validation_files_per_class,
            "shuffle_train_labels": config.shuffle_train_labels,
            "locked_protocol": config.locked_protocol,
            "candidate_file_aggregators": list(config.candidate_file_aggregators),
            "locked_primary_aggregator": locked_primary_aggregator,
            "primary_file_aggregator": locked_primary_aggregator,
            "validation_file_aggregator": locked_primary_aggregator,
            "primary_checkpoint_selection_metric": (
                "validation_file_acc@1_then_validation_file_macro_f1"
            ),
            "aggregator_selection_source": "validation_only",
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
            "validation_file_level_metrics_path": str(validation_file_level_json.resolve()),
            "file_level_primary_report": primary_paths,
            "validation_file_level_primary_report": validation_paths[locked_primary_aggregator],
            "locked_aggregator_selection_path": str(selection_json.resolve()),
            "validation_split_path": str(validation_split_json.resolve()),
            "setting_note": MOONSHOT_TRACK,
        },
    )
    write_json(
        run_dir / "architecture_summary.json",
        {
            **architecture_summary,
            "device": str(device),
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
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


def run_moonshot_device_smoke(config_path: Path) -> MoonshotDeviceSmokeResult:
    config = load_moonshot_run_config(config_path)
    set_seed(config.seed)
    device = resolve_device(config.device)
    dataset = load_base_sensor_dataset(Path(config.data_root))
    prepared = prepare_moonshot_tensors(dataset, config)
    model, architecture_summary = build_moonshot_model(
        config=config,
        input_dim=prepared.train_windows.shape[2],
        num_classes=len(prepared.class_names),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    train_loader = build_dataloader(
        prepared.train_windows,
        maybe_shuffle_train_labels(prepared.train_labels, config),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    batch_x, batch_y = next(iter(train_loader))
    batch_x = batch_x.to(device=device, dtype=torch.float32)
    batch_y = batch_y.to(device=device, dtype=torch.long)
    model.train()
    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    loss.backward()
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    return MoonshotDeviceSmokeResult(
        device=str(device),
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        effective_batch_size=config.batch_size * config.gradient_accumulation_steps,
        feature_count=len(prepared.feature_names),
        parameter_count=int(architecture_summary["parameter_count"]),
        model_family=config.model_name,
    )


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
        "model_name",
        "model",
    }
    missing = sorted(required_keys - set(payload))
    if missing:
        raise MoonshotRunError(f"moonshot config is missing required keys: {missing}")
    if payload["track"] != MOONSHOT_TRACK:
        raise MoonshotRunError(f"unsupported track for moonshot runner: {payload['track']!r}")
    if payload["model_name"] not in {
        "cnn",
        "deep_temporal_resnet",
        "hinception",
        "patch_transformer",
    }:
        raise MoonshotRunError(
            "moonshot runner supports only cnn, deep_temporal_resnet, hinception, "
            "and patch_transformer model_name values"
        )
    locked_protocol = bool(payload.get("locked_protocol", False))
    candidate_file_aggregators = normalize_aggregator_candidates(
        payload.get("candidate_file_aggregators", FILE_LEVEL_AGGREGATORS)
    )
    if locked_protocol:
        primary_file_aggregator = ""
        validation_file_aggregator = ""
    else:
        for aggregator_key in ("primary_file_aggregator", "validation_file_aggregator"):
            if aggregator_key not in payload:
                raise MoonshotRunError(
                    f"{aggregator_key} is required when locked_protocol is disabled"
                )
            if payload[aggregator_key] not in FILE_LEVEL_AGGREGATORS:
                raise MoonshotRunError(f"{aggregator_key} must be one of {FILE_LEVEL_AGGREGATORS}")
        primary_file_aggregator = str(payload["primary_file_aggregator"])
        validation_file_aggregator = str(payload["validation_file_aggregator"])
    model_payload = payload["model"]
    if not isinstance(model_payload, dict):
        raise MoonshotRunError("moonshot cnn model config must be a mapping")
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
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 1)),
        shuffle_train_labels=bool(payload.get("shuffle_train_labels", False)),
        diff_period=int(payload["diff_period"]),
        window_size=int(payload["window_size"]),
        stride=None if payload["stride"] is None else int(payload["stride"]),
        num_workers=int(payload["num_workers"]),
        validation_files_per_class=int(payload["validation_files_per_class"]),
        label_smoothing=float(payload["label_smoothing"]),
        scheduler_name=str(payload["scheduler_name"]),
        scheduler_t_max=int(payload["scheduler_t_max"]),
        scheduler_eta_min=float(payload["scheduler_eta_min"]),
        locked_protocol=locked_protocol,
        candidate_file_aggregators=candidate_file_aggregators,
        primary_file_aggregator=primary_file_aggregator,
        validation_file_aggregator=validation_file_aggregator,
        channel_set=resolve_moonshot_channel_set(
            str(payload.get("channel_set", MOONSHOT_ALL12_CHANNEL_SET))
        ),
        model_name=str(payload["model_name"]),
        model=build_moonshot_model_config(
            model_name=str(payload["model_name"]),
            model_payload=model_payload,
        ),
    )


def build_moonshot_model_config(
    *,
    model_name: str,
    model_payload: dict[str, Any],
) -> (
    CnnModelConfig | TemporalResNetModelConfig | HInceptionModelConfig | PatchTransformerModelConfig
):
    if model_name == "cnn":
        raw_channels = model_payload["channels"]
        if not isinstance(raw_channels, list) or not raw_channels:
            raise MoonshotRunError("moonshot cnn channels must be a non-empty list")
        return CnnModelConfig(
            channels=tuple(int(channel) for channel in raw_channels),
            kernel_size=int(model_payload["kernel_size"]),
            dropout=float(model_payload["dropout"]),
            use_batchnorm=bool(model_payload["use_batchnorm"]),
        )
    if model_name == "deep_temporal_resnet":
        raw_stage_depths = model_payload["stage_depths"]
        raw_stage_widths = model_payload["stage_widths"]
        if not isinstance(raw_stage_depths, list) or len(raw_stage_depths) != 4:
            raise MoonshotRunError("deep_temporal_resnet stage_depths must be a 4-item list")
        if not isinstance(raw_stage_widths, list) or len(raw_stage_widths) != 4:
            raise MoonshotRunError("deep_temporal_resnet stage_widths must be a 4-item list")
        return TemporalResNetModelConfig(
            stage_depths=tuple(int(value) for value in raw_stage_depths),
            stage_widths=tuple(int(value) for value in raw_stage_widths),
            stem_width=int(model_payload["stem_width"]),
            kernel_size=int(model_payload["kernel_size"]),
            normalization=str(model_payload["normalization"]),
            groupnorm_groups=int(model_payload["groupnorm_groups"]),
            se_reduction=int(model_payload["se_reduction"]),
            head_dropout=float(model_payload["head_dropout"]),
            stochastic_depth_probability=float(model_payload["stochastic_depth_probability"]),
        )
    if model_name == "hinception":
        return HInceptionModelConfig(
            stem_channels=int(model_payload["stem_channels"]),
            branch_channels=int(model_payload["branch_channels"]),
            bottleneck_channels=int(model_payload["bottleneck_channels"]),
            num_blocks=int(model_payload["num_blocks"]),
            residual_interval=int(model_payload["residual_interval"]),
            activation_name=str(model_payload["activation_name"]),
            dropout=float(model_payload["dropout"]),
            head_hidden_dim=int(model_payload["head_hidden_dim"]),
        )
    if model_name == "patch_transformer":
        return PatchTransformerModelConfig(
            patch_size=int(model_payload["patch_size"]),
            patch_stride=int(model_payload["patch_stride"]),
            model_dim=int(model_payload["model_dim"]),
            num_heads=int(model_payload["num_heads"]),
            num_layers=int(model_payload["num_layers"]),
            mlp_ratio=float(model_payload["mlp_ratio"]),
            dropout=float(model_payload["dropout"]),
        )
    raise MoonshotRunError(f"unsupported moonshot model_name: {model_name!r}")


def build_moonshot_model(
    *,
    config: MoonshotRunConfig,
    input_dim: int,
    num_classes: int,
) -> tuple[nn.Module, dict[str, Any]]:
    if config.model_name == "cnn":
        if not isinstance(config.model, CnnModelConfig):
            raise MoonshotRunError("moonshot cnn config is malformed")
        model = ExactUpstreamCnnClassifier(
            in_channels=input_dim,
            num_classes=num_classes,
            channels=config.model.channels,
            kernel_size=config.model.kernel_size,
            dropout=config.model.dropout,
            use_batchnorm=config.model.use_batchnorm,
        )
        parameter_count = int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        )
        return model, {
            "model_family": "cnn",
            "block_type": "conv_stack",
            "stage_depths": [len(config.model.channels)],
            "stage_widths": list(config.model.channels),
            "normalization": "batchnorm" if config.model.use_batchnorm else "none",
            "parameter_count": parameter_count,
            "input_feature_count": input_dim,
            "kernel_size": config.model.kernel_size,
            "head_dropout": config.model.dropout,
        }
    if config.model_name == "deep_temporal_resnet":
        if not isinstance(config.model, TemporalResNetModelConfig):
            raise MoonshotRunError("moonshot deep temporal resnet config is malformed")
        model = DeepTemporalResNet1D(
            in_channels=input_dim,
            num_classes=num_classes,
            stage_depths=config.model.stage_depths,
            stage_widths=config.model.stage_widths,
            stem_width=config.model.stem_width,
            kernel_size=config.model.kernel_size,
            normalization=config.model.normalization,
            groupnorm_groups=config.model.groupnorm_groups,
            se_reduction=config.model.se_reduction,
            head_dropout=config.model.head_dropout,
            stochastic_depth_probability=config.model.stochastic_depth_probability,
        )
        return model, model.architecture_summary(input_feature_count=input_dim).to_dict()
    if config.model_name == "hinception":
        if not isinstance(config.model, HInceptionModelConfig):
            raise MoonshotRunError("moonshot hinception config is malformed")
        model = ExactResearchInceptionClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            stem_channels=config.model.stem_channels,
            branch_channels=config.model.branch_channels,
            bottleneck_channels=config.model.bottleneck_channels,
            num_blocks=config.model.num_blocks,
            residual_interval=config.model.residual_interval,
            activation_name=config.model.activation_name,
            dropout=config.model.dropout,
            head_hidden_dim=config.model.head_hidden_dim,
        )
        return (
            model,
            build_inception_model_summary(
                model=model,
                input_dim=input_dim,
                stem_channels=config.model.stem_channels,
                branch_channels=config.model.branch_channels,
                bottleneck_channels=config.model.bottleneck_channels,
                num_blocks=config.model.num_blocks,
                residual_interval=config.model.residual_interval,
                activation_name=config.model.activation_name,
                dropout=config.model.dropout,
                head_hidden_dim=config.model.head_hidden_dim,
            ).to_dict(),
        )
    if config.model_name == "patch_transformer":
        if not isinstance(config.model, PatchTransformerModelConfig):
            raise MoonshotRunError("moonshot patch transformer config is malformed")
        model = TemporalPatchTransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            patch_size=config.model.patch_size,
            patch_stride=config.model.patch_stride,
            model_dim=config.model.model_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
        )
        return model, model.architecture_summary(input_feature_count=input_dim).to_dict()
    raise MoonshotRunError(f"unsupported moonshot model_name: {config.model_name!r}")


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
        channel_set=config.channel_set,
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
        channel_set=prepared.channel_set,
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
    gradient_accumulation_steps: int,
    locked_protocol: bool,
    candidate_file_aggregators: tuple[str, ...],
    validation_file_aggregator: str,
    checkpoint_dir: Path,
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    if gradient_accumulation_steps <= 0:
        raise MoonshotRunError("gradient_accumulation_steps must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        scheduler_t_max=scheduler_t_max,
        scheduler_eta_min=scheduler_eta_min,
    )
    best_candidates: dict[str, AggregatorCheckpointSelection] = {}
    history: list[MoonshotHistoryRow] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        optimizer.zero_grad()
        for batch_index, (batch_x, batch_y) in enumerate(train_loader, start=1):
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.long)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            (loss / gradient_accumulation_steps).backward()
            if batch_index % gradient_accumulation_steps == 0 or batch_index == len(train_loader):
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
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
        validation_bundle = build_window_prediction_bundle(
            class_names=validation_outputs.metrics.class_names,
            true_labels=validation_outputs.true_labels,
            predicted_labels=validation_outputs.predicted_labels,
            topk_indices=validation_outputs.topk_indices,
            logits=validation_outputs.logits,
            windows=validation_windows,
        )
        validation_file_results = {
            aggregator: aggregate_file_level_metrics(
                bundle=validation_bundle,
                category_mapping=category_mapping,
                aggregator=aggregator,
            )
            for aggregator in (
                candidate_file_aggregators if locked_protocol else (validation_file_aggregator,)
            )
        }
        epoch_primary_aggregator = (
            select_validation_locked_aggregator(
                [
                    AggregatorSelectionCandidate(
                        aggregator=aggregator,
                        acc_at_1=result.metrics.acc_at_1,
                        f1_macro=result.metrics.f1_macro,
                    )
                    for aggregator, result in validation_file_results.items()
                ]
            )
            if locked_protocol
            else validation_file_aggregator
        )
        validation_file_metrics = validation_file_results[epoch_primary_aggregator].metrics
        current_lr = float(optimizer.param_groups[0]["lr"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for aggregator, result in validation_file_results.items():
            candidate = AggregatorCheckpointSelection(
                aggregator=aggregator,
                epoch=epoch,
                checkpoint_path=checkpoint_dir / f"{aggregator}.pt",
                validation_file_acc_at_1=result.metrics.acc_at_1,
                validation_file_acc_at_5=result.metrics.acc_at_5,
                validation_file_f1=result.metrics.f1_macro,
                validation_window_acc_at_1=validation_outputs.metrics.acc_at_1,
                validation_window_acc_at_5=validation_outputs.metrics.acc_at_5,
                validation_window_f1=validation_outputs.metrics.f1_macro,
            )
            if is_better_validation_checkpoint(candidate, best_candidates.get(aggregator)):
                best_candidates[aggregator] = candidate
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config_payload,
                        "best_validation_file_acc@1": candidate.validation_file_acc_at_1,
                        "best_validation_file_acc@5": candidate.validation_file_acc_at_5,
                        "best_validation_file_f1": candidate.validation_file_f1,
                        "best_validation_window_acc@1": candidate.validation_window_acc_at_1,
                        "best_validation_window_acc@5": candidate.validation_window_acc_at_5,
                        "best_validation_window_f1": candidate.validation_window_f1,
                        "epoch": candidate.epoch,
                        "locked_primary_aggregator": aggregator,
                    },
                    candidate.checkpoint_path,
                )
        selected_candidate = select_best_validation_candidate(
            best_candidates=best_candidates,
            locked_protocol=locked_protocol,
            candidate_file_aggregators=candidate_file_aggregators,
            validation_file_aggregator=validation_file_aggregator,
        )
        is_best = (
            selected_candidate.epoch == epoch
            and selected_candidate.aggregator == epoch_primary_aggregator
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
                validation_primary_aggregator=epoch_primary_aggregator,
                is_best=is_best,
            )
        )
        if scheduler is not None:
            scheduler.step()
    best_selection = select_best_validation_candidate(
        best_candidates=best_candidates,
        locked_protocol=locked_protocol,
        candidate_file_aggregators=candidate_file_aggregators,
        validation_file_aggregator=validation_file_aggregator,
    )
    if not best_selection.checkpoint_path.is_file():
        raise MoonshotRunError("moonshot training did not produce a best checkpoint")
    return {
        "history": history,
        "locked_primary_aggregator": best_selection.aggregator,
        "best_selection": best_selection,
        "best_candidates": [
            {
                "aggregator": candidate.aggregator,
                "epoch": candidate.epoch,
                "checkpoint_path": str(candidate.checkpoint_path.resolve()),
                "validation_file_acc_at_1": candidate.validation_file_acc_at_1,
                "validation_file_acc_at_5": candidate.validation_file_acc_at_5,
                "validation_file_f1": candidate.validation_file_f1,
                "validation_window_acc_at_1": candidate.validation_window_acc_at_1,
                "validation_window_acc_at_5": candidate.validation_window_acc_at_5,
                "validation_window_f1": candidate.validation_window_f1,
            }
            for candidate in sorted(best_candidates.values(), key=lambda item: item.aggregator)
        ],
    }


def is_better_validation_checkpoint(
    candidate: AggregatorCheckpointSelection,
    incumbent: AggregatorCheckpointSelection | None,
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


def select_best_validation_candidate(
    *,
    best_candidates: dict[str, AggregatorCheckpointSelection],
    locked_protocol: bool,
    candidate_file_aggregators: tuple[str, ...],
    validation_file_aggregator: str,
) -> AggregatorCheckpointSelection:
    if not best_candidates:
        raise MoonshotRunError("moonshot training did not record validation checkpoints")
    if not locked_protocol:
        if validation_file_aggregator not in best_candidates:
            raise MoonshotRunError(
                f"missing validation checkpoint for aggregator {validation_file_aggregator!r}"
            )
        return best_candidates[validation_file_aggregator]
    locked_aggregator = select_validation_locked_aggregator(
        [
            AggregatorSelectionCandidate(
                aggregator=aggregator,
                acc_at_1=best_candidates[aggregator].validation_file_acc_at_1,
                f1_macro=best_candidates[aggregator].validation_file_f1,
            )
            for aggregator in candidate_file_aggregators
            if aggregator in best_candidates
        ]
    )
    return best_candidates[locked_aggregator]


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
                "validation_primary_aggregator",
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
                    row.validation_primary_aggregator,
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
    channel_set: str,
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
        "channel_set": channel_set,
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
