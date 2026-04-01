"""research-extension gc-ms pretrain and fine-tune runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from smelt.datasets import (
    DIFF_VIEW,
    build_base_class_vocab_manifest,
    load_base_sensor_dataset,
    load_research_gcms_anchor_set,
    write_base_class_vocab_manifest,
    write_research_gcms_anchor_usage,
)
from smelt.evaluation import export_classification_report, load_category_mapping
from smelt.models import (
    ExactResearchInceptionClassifier,
    ResearchGcmsPretrainModel,
    extract_inception_encoder_state_dict,
    load_inception_encoder_state_dict,
)

from .run import (
    build_dataloader,
    build_run_dir,
    collect_evaluation_outputs,
    expand_env_values,
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
from .run_research import (
    PreparedResearchWindowTensors,
    ResearchInceptionModelConfig,
    ResearchRunError,
    prepare_research_window_tensors,
    write_research_view_manifest,
)

GCMS_PRETRAIN_MODE = "gcms_pretrain"
GCMS_FINETUNE_MODE = "gcms_finetune"


@dataclass(slots=True)
class ResearchGcmsRunConfig:
    track: str
    mode: str
    experiment_name: str
    config_path: str
    data_root: str
    output_root: str
    category_map_path: str
    class_vocab_manifest_path: str
    exact_upstream_regression_path: str
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
    view_mode: str
    model: ResearchInceptionModelConfig
    projection_dim: int | None = None
    gcms_hidden_dim: int | None = None
    temperature: float | None = None
    pretrained_checkpoint_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = asdict(self.model)
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_gcms_run_config(args.config)
    if config.mode == GCMS_PRETRAIN_MODE:
        run_dir, metrics, checkpoint_path, prepared = run_gcms_pretrain(args.config)
    elif config.mode == GCMS_FINETUNE_MODE:
        run_dir, metrics, checkpoint_path, prepared = run_gcms_finetune(args.config)
    else:
        raise ResearchRunError(f"unsupported gc-ms research mode: {config.mode!r}")
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


def run_gcms_pretrain(
    config_path: Path,
) -> tuple[Path, Any, Path, PreparedResearchWindowTensors]:
    config = load_gcms_run_config(config_path)
    if config.mode != GCMS_PRETRAIN_MODE:
        raise ResearchRunError(f"expected {GCMS_PRETRAIN_MODE}, found {config.mode!r}")
    if (
        config.projection_dim is None
        or config.gcms_hidden_dim is None
        or config.temperature is None
    ):
        raise ResearchRunError("gc-ms pretrain config is missing projection/gcms encoder fields")

    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = build_run_dir(Path(config.output_root), config.experiment_name, config.to_dict())
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_base_sensor_dataset(Path(config.data_root))
    vocab_manifest = build_base_class_vocab_manifest(dataset)
    validate_reference_manifest(config.class_vocab_manifest_path, vocab_manifest.to_dict())
    validate_required_reference(config.exact_upstream_regression_path)
    validate_required_reference(config.gcms_class_map_manifest_path)
    category_mapping = load_category_mapping(Path(config.category_map_path))
    prepared = prepare_research_window_tensors(dataset, config)
    anchor_set = load_research_gcms_anchor_set(
        Path(config.gcms_class_map_manifest_path),
        class_names=prepared.class_names,
    )

    sensor_backbone = build_research_classifier(prepared, config)
    model = ResearchGcmsPretrainModel(
        sensor_backbone=sensor_backbone,
        gcms_feature_count=anchor_set.feature_count,
        projection_dim=config.projection_dim,
        gcms_hidden_dim=config.gcms_hidden_dim,
        activation_name=config.model.activation_name,
        temperature=config.temperature,
    ).to(device)
    model.set_anchor_features(torch.from_numpy(anchor_set.feature_matrix).to(device))

    train_loader = build_dataloader(
        prepared.train_windows,
        prepared.train_labels,
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
        {
            "mode": config.mode,
            "model_state_dict": model.state_dict(),
            "sensor_encoder_state_dict": extract_inception_encoder_state_dict(
                model.sensor_backbone
            ),
            "config": config.to_dict(),
            "anchor_usage": anchor_set.to_artifact_dict(),
        },
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
            "mode": config.mode,
            "view_mode": config.view_mode,
            "objective": "gcms_anchor_alignment_cross_entropy",
            "gcms_class_map_manifest_path": str(
                Path(config.gcms_class_map_manifest_path).resolve()
            ),
            "exact_upstream_regression_path": str(
                Path(config.exact_upstream_regression_path).resolve()
            ),
            "train_window_count": prepared.train_window_count,
            "test_window_count": prepared.test_window_count,
            "train_standardization_shape": list(prepared.train_standardization_shape),
            "feature_names": list(prepared.feature_names),
            "retained_columns": list(prepared.retained_columns),
            "gcms_anchor_count": anchor_set.anchor_count,
            "gcms_feature_count": anchor_set.feature_count,
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
            "mode": config.mode,
            "view_mode": config.view_mode,
            "reference_artifacts": {
                "category_map_path": str(Path(config.category_map_path).resolve()),
                "class_vocab_manifest_path": str(Path(config.class_vocab_manifest_path).resolve()),
                "exact_upstream_regression_path": str(
                    Path(config.exact_upstream_regression_path).resolve()
                ),
                "gcms_class_map_manifest_path": str(
                    Path(config.gcms_class_map_manifest_path).resolve()
                ),
            },
            "window_counts": {
                "train": prepared.train_window_count,
                "test": prepared.test_window_count,
            },
            "feature_names": list(prepared.feature_names),
            "checkpoint_path": str(checkpoint_path.resolve()),
            "pretraining_objective": "gcms_anchor_alignment_cross_entropy",
            "gcms_anchor_count": anchor_set.anchor_count,
        },
    )
    write_base_class_vocab_manifest(run_dir / "base_class_vocab.json", vocab_manifest)
    write_research_view_manifest(run_dir / "research_view_manifest.json", prepared.view_manifest)
    write_research_gcms_anchor_usage(run_dir / "gcms_anchor_usage.json", anchor_set)
    return run_dir, metrics, checkpoint_path, prepared


def run_gcms_finetune(
    config_path: Path,
) -> tuple[Path, Any, Path, PreparedResearchWindowTensors]:
    config = load_gcms_run_config(config_path)
    if config.mode != GCMS_FINETUNE_MODE:
        raise ResearchRunError(f"expected {GCMS_FINETUNE_MODE}, found {config.mode!r}")
    if not config.pretrained_checkpoint_path:
        raise ResearchRunError("gc-ms fine-tune config must include pretrained_checkpoint_path")

    set_seed(config.seed)
    device = resolve_device(config.device)
    run_dir = build_run_dir(Path(config.output_root), config.experiment_name, config.to_dict())
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_base_sensor_dataset(Path(config.data_root))
    vocab_manifest = build_base_class_vocab_manifest(dataset)
    validate_reference_manifest(config.class_vocab_manifest_path, vocab_manifest.to_dict())
    validate_required_reference(config.exact_upstream_regression_path)
    validate_required_reference(config.gcms_class_map_manifest_path)
    validate_required_reference(config.pretrained_checkpoint_path)
    category_mapping = load_category_mapping(Path(config.category_map_path))
    prepared = prepare_research_window_tensors(dataset, config)
    anchor_set = load_research_gcms_anchor_set(
        Path(config.gcms_class_map_manifest_path),
        class_names=prepared.class_names,
    )
    model = build_research_classifier(prepared, config).to(device)
    load_pretrained_sensor_encoder(model, Path(config.pretrained_checkpoint_path))

    train_loader = build_dataloader(
        prepared.train_windows,
        prepared.train_labels,
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
        {
            "mode": config.mode,
            "model_state_dict": model.state_dict(),
            "sensor_encoder_state_dict": extract_inception_encoder_state_dict(model),
            "config": config.to_dict(),
            "anchor_usage": anchor_set.to_artifact_dict(),
        },
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
            "mode": config.mode,
            "view_mode": config.view_mode,
            "pretrained_checkpoint_path": str(Path(config.pretrained_checkpoint_path).resolve()),
            "gcms_class_map_manifest_path": str(
                Path(config.gcms_class_map_manifest_path).resolve()
            ),
            "exact_upstream_regression_path": str(
                Path(config.exact_upstream_regression_path).resolve()
            ),
            "train_window_count": prepared.train_window_count,
            "test_window_count": prepared.test_window_count,
            "train_standardization_shape": list(prepared.train_standardization_shape),
            "feature_names": list(prepared.feature_names),
            "retained_columns": list(prepared.retained_columns),
            "gcms_anchor_count": anchor_set.anchor_count,
            "gcms_feature_count": anchor_set.feature_count,
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
            "mode": config.mode,
            "view_mode": config.view_mode,
            "reference_artifacts": {
                "category_map_path": str(Path(config.category_map_path).resolve()),
                "class_vocab_manifest_path": str(Path(config.class_vocab_manifest_path).resolve()),
                "exact_upstream_regression_path": str(
                    Path(config.exact_upstream_regression_path).resolve()
                ),
                "gcms_class_map_manifest_path": str(
                    Path(config.gcms_class_map_manifest_path).resolve()
                ),
                "pretrained_checkpoint_path": str(
                    Path(config.pretrained_checkpoint_path).resolve()
                ),
            },
            "window_counts": {
                "train": prepared.train_window_count,
                "test": prepared.test_window_count,
            },
            "feature_names": list(prepared.feature_names),
            "checkpoint_path": str(checkpoint_path.resolve()),
        },
    )
    write_base_class_vocab_manifest(run_dir / "base_class_vocab.json", vocab_manifest)
    write_research_view_manifest(run_dir / "research_view_manifest.json", prepared.view_manifest)
    write_research_gcms_anchor_usage(run_dir / "gcms_anchor_usage.json", anchor_set)
    return run_dir, metrics, checkpoint_path, prepared


def load_gcms_run_config(config_path: Path) -> ResearchGcmsRunConfig:
    try:
        raw_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ResearchRunError(f"unable to read config file: {config_path}") from exc
    if not isinstance(raw_payload, dict):
        raise ResearchRunError("gc-ms research run config must deserialize to a mapping")
    payload = expand_env_values(raw_payload)
    required_keys = {
        "track",
        "mode",
        "experiment_name",
        "data_root",
        "output_root",
        "category_map_path",
        "class_vocab_manifest_path",
        "exact_upstream_regression_path",
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
        "view_mode",
        "model",
    }
    missing_keys = sorted(required_keys - set(payload))
    if missing_keys:
        raise ResearchRunError(f"gc-ms research config is missing required keys: {missing_keys}")
    if payload["track"] != "research-extension":
        raise ResearchRunError(f"unsupported track for gc-ms runner: {payload['track']!r}")
    if payload["view_mode"] != DIFF_VIEW:
        raise ResearchRunError(
            f"t11 only supports the diff research view, found {payload['view_mode']!r}"
        )
    if int(payload["diff_period"]) <= 0:
        raise ResearchRunError("t11 diff-only gc-ms runs require diff_period > 0")
    model_payload = payload["model"]
    if not isinstance(model_payload, dict):
        raise ResearchRunError("research model config must be a mapping")
    return ResearchGcmsRunConfig(
        track=str(payload["track"]),
        mode=str(payload["mode"]),
        experiment_name=str(payload["experiment_name"]),
        config_path=str(config_path.resolve()),
        data_root=str(payload["data_root"]),
        output_root=str(payload["output_root"]),
        category_map_path=str(payload["category_map_path"]),
        class_vocab_manifest_path=str(payload["class_vocab_manifest_path"]),
        exact_upstream_regression_path=str(payload["exact_upstream_regression_path"]),
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
        view_mode=str(payload["view_mode"]),
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
        projection_dim=(
            None if payload.get("projection_dim") is None else int(payload["projection_dim"])
        ),
        gcms_hidden_dim=(
            None if payload.get("gcms_hidden_dim") is None else int(payload["gcms_hidden_dim"])
        ),
        temperature=(None if payload.get("temperature") is None else float(payload["temperature"])),
        pretrained_checkpoint_path=(
            None
            if payload.get("pretrained_checkpoint_path") is None
            else str(payload["pretrained_checkpoint_path"])
        ),
    )


def build_research_classifier(
    prepared: PreparedResearchWindowTensors,
    config: ResearchGcmsRunConfig,
) -> ExactResearchInceptionClassifier:
    return ExactResearchInceptionClassifier(
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
    )


def load_pretrained_sensor_encoder(
    model: ExactResearchInceptionClassifier,
    checkpoint_path: Path,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ResearchRunError(f"pretrained checkpoint must be a mapping: {checkpoint_path}")
    encoder_state_dict = checkpoint.get("sensor_encoder_state_dict")
    if not isinstance(encoder_state_dict, dict):
        raise ResearchRunError(
            f"pretrained checkpoint is missing sensor_encoder_state_dict: {checkpoint_path}"
        )
    load_inception_encoder_state_dict(model, encoder_state_dict)


if __name__ == "__main__":
    raise SystemExit(main())
