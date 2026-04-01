"""research-extension supervised classifier runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from smelt.datasets import (
    FUSED_RAW_DIFF_VIEW,
    RESEARCH_VIEW_MODES,
    build_base_class_vocab_manifest,
    load_base_sensor_dataset,
    preprocess_split_records_for_view,
    write_base_class_vocab_manifest,
)
from smelt.evaluation import export_classification_report, load_category_mapping
from smelt.models import ExactResearchInceptionClassifier
from smelt.preprocessing import (
    WindowedSplit,
    apply_window_standardizer,
    fit_window_standardizer,
    generate_split_windows,
    stack_window_values,
)

from .run import (
    build_dataloader,
    build_run_dir,
    collect_evaluation_outputs,
    expand_env_values,
    maybe_shuffle_train_labels,
    resolve_device,
    set_seed,
    train_classifier,
    validate_reference_manifest,
    validate_required_reference,
    write_prediction_bundle,
    write_resolved_config,
    write_run_metadata,
    write_training_history,
)


class ResearchRunError(Exception):
    """raised when a research-extension supervised run cannot proceed safely."""


@dataclass(slots=True)
class ResearchInceptionModelConfig:
    stem_channels: int = 64
    branch_channels: int = 32
    bottleneck_channels: int = 32
    num_blocks: int = 6
    residual_interval: int = 3
    activation_name: str = "gelu"
    dropout: float = 0.1
    head_hidden_dim: int = 128


@dataclass(slots=True)
class ResearchRunConfig:
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
    view_mode: str
    shuffle_train_labels: bool
    model: ResearchInceptionModelConfig

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = asdict(self.model)
        return payload


@dataclass(slots=True)
class PreparedResearchWindowTensors:
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    retained_columns: tuple[str, ...]
    train_windows: np.ndarray
    train_labels: np.ndarray
    test_windows: np.ndarray
    test_labels: np.ndarray
    train_window_count: int
    test_window_count: int
    train_standardization_shape: tuple[int, int]
    view_manifest: dict[str, Any]
    standardized_train_split: WindowedSplit
    standardized_test_split: WindowedSplit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir, metrics, checkpoint_path, prepared = run_research_supervised(args.config)
    print(f"run_dir: {run_dir}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"acc@1: {metrics.acc_at_1}")
    print(f"acc@5: {metrics.acc_at_5}")
    print(f"precision_macro: {metrics.precision_macro}")
    print(f"recall_macro: {metrics.recall_macro}")
    print(f"f1_macro: {metrics.f1_macro}")
    print(f"train_window_count: {prepared.train_window_count}")
    print(f"test_window_count: {prepared.test_window_count}")
    print(f"feature_count: {len(prepared.feature_names)}")
    return 0


def run_research_supervised(
    config_path: Path,
) -> tuple[Path, Any, Path, PreparedResearchWindowTensors]:
    config = load_research_run_config(config_path)
    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = build_run_dir(Path(config.output_root), config.experiment_name, config.to_dict())
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_base_sensor_dataset(Path(config.data_root))
    vocab_manifest = build_base_class_vocab_manifest(dataset)
    validate_reference_manifest(config.class_vocab_manifest_path, vocab_manifest.to_dict())
    validate_required_reference(config.exact_upstream_regression_path)
    category_mapping = load_category_mapping(Path(config.category_map_path))

    prepared = prepare_research_window_tensors(dataset, config)
    model = ExactResearchInceptionClassifier(
        input_dim=prepared.train_windows.shape[2],
        num_classes=len(prepared.class_names),
        stem_channels=config.model.stem_channels,
        branch_channels=config.model.branch_channels,
        bottleneck_channels=config.model.bottleneck_channels,
        num_blocks=config.model.num_blocks,
        residual_interval=config.model.residual_interval,
        activation_name=config.model.activation_name,
        dropout=config.model.dropout,
        head_hidden_dim=config.model.head_hidden_dim,
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
    import torch

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
            "track": "research-extension",
            "view_mode": config.view_mode,
            "exact_upstream_regression_path": str(
                Path(config.exact_upstream_regression_path).resolve()
            ),
            "train_window_count": prepared.train_window_count,
            "test_window_count": prepared.test_window_count,
            "train_standardization_shape": list(prepared.train_standardization_shape),
            "feature_names": list(prepared.feature_names),
            "retained_columns": list(prepared.retained_columns),
        },
        overwrite=True,
    )
    write_prediction_bundle(
        run_dir / "predictions.npz",
        evaluation,
        windows=prepared.standardized_test_split.windows,
    )
    write_training_history(run_dir / "training_history.csv", history)
    write_resolved_config(run_dir / "resolved_config.yaml", config)
    write_run_metadata(
        run_dir / "run_metadata.json",
        {
            "track": "research-extension",
            "mode": "supervised_classification",
            "view_mode": config.view_mode,
            "reference_artifacts": {
                "category_map_path": str(Path(config.category_map_path).resolve()),
                "class_vocab_manifest_path": str(Path(config.class_vocab_manifest_path).resolve()),
                "exact_upstream_regression_path": str(
                    Path(config.exact_upstream_regression_path).resolve()
                ),
            },
            "window_counts": {
                "train": prepared.train_window_count,
                "test": prepared.test_window_count,
            },
            "feature_names": list(prepared.feature_names),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "shuffle_train_labels": config.shuffle_train_labels,
        },
    )
    write_base_class_vocab_manifest(run_dir / "base_class_vocab.json", vocab_manifest)
    write_research_view_manifest(run_dir / "research_view_manifest.json", prepared.view_manifest)
    return run_dir, metrics, checkpoint_path, prepared


def load_research_run_config(config_path: Path) -> ResearchRunConfig:
    try:
        raw_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ResearchRunError(f"unable to read config file: {config_path}") from exc
    if not isinstance(raw_payload, dict):
        raise ResearchRunError("research run config must deserialize to a mapping")

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
        "view_mode",
        "model",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        raise ResearchRunError(f"research run config is missing required keys: {missing_keys}")
    if payload["track"] != "research-extension":
        raise ResearchRunError(f"unsupported track for this runner: {payload['track']!r}")
    if payload["view_mode"] not in RESEARCH_VIEW_MODES:
        raise ResearchRunError(
            "unsupported research view for this runner: "
            f"{payload['view_mode']!r}; expected one of {RESEARCH_VIEW_MODES}"
        )
    if payload["view_mode"] == FUSED_RAW_DIFF_VIEW and int(payload["diff_period"]) <= 0:
        raise ResearchRunError("fused_raw_diff requires diff_period > 0")
    model_payload = payload["model"]
    if not isinstance(model_payload, dict):
        raise ResearchRunError("research model config must be a mapping")
    return ResearchRunConfig(
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
        view_mode=str(payload["view_mode"]),
        shuffle_train_labels=bool(payload.get("shuffle_train_labels", False)),
        model=ResearchInceptionModelConfig(
            stem_channels=int(model_payload["stem_channels"]),
            branch_channels=int(model_payload["branch_channels"]),
            bottleneck_channels=int(model_payload["bottleneck_channels"]),
            num_blocks=int(model_payload["num_blocks"]),
            residual_interval=int(model_payload["residual_interval"]),
            activation_name=str(model_payload["activation_name"]),
            dropout=float(model_payload["dropout"]),
            head_hidden_dim=int(model_payload["head_hidden_dim"]),
        ),
    )


def prepare_research_window_tensors(
    dataset: Any,
    config: ResearchRunConfig,
) -> PreparedResearchWindowTensors:
    class_names = build_base_class_vocab_manifest(dataset).class_vocab
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    train_records = preprocess_split_records_for_view(
        dataset.train_records,
        view_mode=config.view_mode,
        diff_period=config.diff_period,
    )
    test_records = preprocess_split_records_for_view(
        dataset.test_records,
        view_mode=config.view_mode,
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
        raise ResearchRunError("research train split produced zero windows")
    if test_windows.window_count == 0:
        raise ResearchRunError("research test split produced zero windows")

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
    view_manifest = {
        "resolved_data_root": dataset.resolved_data_root,
        "track": "research-extension",
        "view_mode": config.view_mode,
        "differencing_period": config.diff_period,
        "window_size": config.window_size,
        "stride": train_windows.stride,
        "retained_columns": list(train_records[0].column_names[:6]),
        "feature_names": list(train_records[0].column_names),
        "feature_count": len(train_records[0].column_names),
        "train_record_count": len(train_records),
        "test_record_count": len(test_records),
        "train_window_count": standardized_train.window_count,
        "test_window_count": standardized_test.window_count,
        "standardization_shape": [standardizer.sample_count, standardizer.feature_count],
    }
    return PreparedResearchWindowTensors(
        class_names=class_names,
        feature_names=train_records[0].column_names,
        retained_columns=tuple(
            name.removeprefix("raw_") for name in train_records[0].column_names[:6]
        ),
        train_windows=train_values,
        train_labels=train_labels,
        test_windows=test_values,
        test_labels=test_labels,
        train_window_count=standardized_train.window_count,
        test_window_count=standardized_test.window_count,
        train_standardization_shape=(standardizer.sample_count, standardizer.feature_count),
        view_manifest=view_manifest,
        standardized_train_split=standardized_train,
        standardized_test_split=standardized_test,
    )


def write_research_view_manifest(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
