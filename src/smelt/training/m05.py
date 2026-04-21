"""helpers for the m05 grouped-cv moonshot protocol."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from smelt.datasets import (
    BaseSensorDataset,
    SensorFileRecord,
    build_base_class_vocab_manifest,
    build_grouped_cv_fold_manifest,
    load_base_sensor_dataset,
    write_base_class_vocab_manifest,
)
from smelt.evaluation import (
    FILE_LEVEL_AGGREGATORS,
    LOCKED_AGGREGATOR_TIEBREAK_ORDER,
    FileScoreBundle,
    aggregate_file_level_metrics,
    build_file_score_bundle,
    export_file_level_report,
    load_category_mapping,
    write_dict_rows_csv,
    write_file_score_bundle,
    write_json,
)
from smelt.training.m04 import (
    M04_ENSEMBLE_METHODS,
    M04_ENSEMBLE_PREFERENCE,
    compute_file_result_from_score_bundle,
    evaluate_cv_ensemble_candidates,
    evaluate_score_ensemble,
)

from .run import (
    build_dataloader,
    build_run_dir,
    maybe_shuffle_train_labels,
    resolve_device,
    set_seed,
    validate_reference_manifest,
    validate_required_reference,
    write_run_metadata,
)
from .run_moonshot import (
    MoonshotRunConfig,
    build_moonshot_model,
    build_scheduler,
    evaluate_moonshot_checkpoint,
    load_moonshot_run_config,
    prepare_moonshot_tensors_from_records,
    require_split_array,
    require_standardized_split,
    train_moonshot_classifier,
)

M05_DEFAULT_BANK_CONFIGS = (
    "configs/moonshot-enhanced/m01c_cnn_all12_diff_locked_seed13.yaml",
    "configs/moonshot-enhanced/m01c_cnn_all12_diff_locked_seed42.yaml",
    "configs/moonshot-enhanced/m01c_cnn_all12_diff_locked_seed7.yaml",
    "configs/moonshot-enhanced/m02_deep_temporal_resnet_all12_diff_locked.yaml",
    "configs/moonshot-enhanced/m04_cnn_all12_diff_locked_seed101.yaml",
    "configs/moonshot-enhanced/m04_cnn_all12_diff_locked_seed202.yaml",
    "configs/moonshot-enhanced/m04_deep_temporal_resnet_all12_diff_seed7.yaml",
    "configs/moonshot-enhanced/m04_hinception_all12_diff_seed17.yaml",
    "configs/moonshot-enhanced/m04_hinception_all12_diff_seed29.yaml",
    "configs/moonshot-enhanced/m04_patch_transformer_all12_diff_seed19.yaml",
)
M05_FOLD_COUNT = 5
M05_ENSEMBLE_SELECTION_SOURCE = "cv_oof_only"


class M05Error(Exception):
    """raised when the m05 protocol cannot proceed safely."""


@dataclass(slots=True)
class M05BankEntry:
    member_id: str
    config_path: Path
    config: MoonshotRunConfig
    stable_order: int


@dataclass(slots=True)
class FoldAggregatorResult:
    aggregator: str
    epoch: int
    checkpoint_path: Path
    validation_window_metrics: Any
    validation_file_result: Any
    validation_file_score_bundle: FileScoreBundle
    validation_prediction_bundle_path: Path
    validation_file_score_bundle_path: Path
    validation_report_paths: dict[str, str]


@dataclass(slots=True)
class FoldModelResult:
    member_id: str
    fold_index: int
    run_dir: Path
    selected_aggregator: str
    selected_epoch: int
    candidate_results: dict[str, FoldAggregatorResult]
    history_path: Path


@dataclass(slots=True)
class AggregatorCvSummary:
    aggregator: str
    oof_result: Any
    oof_score_bundle: FileScoreBundle
    oof_score_bundle_path: Path
    oof_report_paths: dict[str, str]
    fold_mean_file_acc_at_1: float
    fold_std_file_acc_at_1: float
    fold_mean_file_macro_f1: float
    fold_std_file_macro_f1: float
    epoch_mean: float
    epoch_std: float
    epoch_budget: int
    fold_epochs: tuple[int, ...]


@dataclass(slots=True)
class ModelCvSummary:
    entry: M05BankEntry
    fold_results: tuple[FoldModelResult, ...]
    aggregator_summaries: dict[str, AggregatorCvSummary]
    selected_aggregator: str
    selection_report_path: Path

    @property
    def selected_summary(self) -> AggregatorCvSummary:
        return self.aggregator_summaries[self.selected_aggregator]


@dataclass(slots=True)
class FrozenRefitMember:
    member_id: str
    config_path: Path
    config: MoonshotRunConfig
    stable_order: int
    selected_aggregator: str
    epoch_budget: int
    selected_weight: float
    oof_file_acc_at_1: float
    oof_file_macro_f1: float


@dataclass(slots=True)
class RefitMemberResult:
    plan: FrozenRefitMember
    run_dir: Path
    test_file_score_bundle: FileScoreBundle


def build_default_m05_bank_config_paths() -> tuple[Path, ...]:
    return tuple(Path(value).resolve() for value in M05_DEFAULT_BANK_CONFIGS)


def load_m05_bank_entries(
    config_paths: tuple[Path, ...],
) -> tuple[M05BankEntry, ...]:
    if not config_paths:
        raise M05Error("m05 requires at least one bank config path")
    entries: list[M05BankEntry] = []
    seen_member_ids: set[str] = set()
    shared_data_root: str | None = None
    shared_category_map_path: str | None = None
    shared_class_vocab_manifest_path: str | None = None
    for stable_order, config_path in enumerate(config_paths):
        config = load_moonshot_run_config(config_path)
        if config.channel_set != "all12":
            raise M05Error(f"{config_path} is not an all12 config")
        if config.diff_period != 25:
            raise M05Error(f"{config_path} does not preserve g=25")
        if config.window_size != 100:
            raise M05Error(f"{config_path} does not preserve window_size=100")
        if config.stride != 50:
            raise M05Error(f"{config_path} does not preserve stride=50")
        if not config.locked_protocol:
            raise M05Error(f"{config_path} must keep locked_protocol enabled for m05")
        if not set(config.candidate_file_aggregators) <= set(FILE_LEVEL_AGGREGATORS):
            raise M05Error(
                f"{config_path} has unsupported candidate aggregators "
                f"{config.candidate_file_aggregators!r}"
            )
        if config.shuffle_train_labels:
            raise M05Error(f"{config_path} enables shuffled labels; m05 requires real labels")
        if shared_data_root is None:
            shared_data_root = config.data_root
            shared_category_map_path = config.category_map_path
            shared_class_vocab_manifest_path = config.class_vocab_manifest_path
        else:
            if config.data_root != shared_data_root:
                raise M05Error("all m05 bank configs must share the same data_root")
            if config.category_map_path != shared_category_map_path:
                raise M05Error("all m05 bank configs must share the same category_map_path")
            if config.class_vocab_manifest_path != shared_class_vocab_manifest_path:
                raise M05Error("all m05 bank configs must share the same class_vocab_manifest_path")
        member_id = config.experiment_name
        if member_id in seen_member_ids:
            raise M05Error(f"duplicate m05 member_id {member_id!r}")
        seen_member_ids.add(member_id)
        entries.append(
            M05BankEntry(
                member_id=member_id,
                config_path=config_path.resolve(),
                config=config,
                stable_order=stable_order,
            )
        )
    return tuple(entries)


def stable_sensor_content_hash(record: SensorFileRecord) -> str:
    values = np.asarray(record.rows, dtype=np.float64)
    digest = hashlib.sha1()
    digest.update(f"{values.shape[0]}x{values.shape[1]}|".encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def build_m05_duplicate_audit(
    *,
    dataset: BaseSensorDataset,
    fold_manifest: Any,
) -> dict[str, Any]:
    roles_by_path: dict[str, set[str]] = {}
    records_by_path: dict[str, SensorFileRecord] = {}

    def register(record: SensorFileRecord, role: str) -> None:
        roles_by_path.setdefault(record.relative_path, set()).add(role)
        records_by_path.setdefault(record.relative_path, record)

    for fold in fold_manifest.folds:
        for record in fold.train_records:
            register(record, f"fold_{fold.fold_index}:train")
        for record in fold.validation_records:
            register(record, f"fold_{fold.fold_index}:validation")
    for record in dataset.test_records:
        register(record, "official_test")

    hash_entries: dict[str, list[dict[str, Any]]] = {}
    for relative_path in sorted(records_by_path):
        record = records_by_path[relative_path]
        boundary_categories = set()
        for role in roles_by_path[relative_path]:
            if role.endswith(":train"):
                boundary_categories.add("cv_train")
            elif role.endswith(":validation"):
                boundary_categories.add("cv_validation")
            else:
                boundary_categories.add("official_test")
        hash_entries.setdefault(stable_sensor_content_hash(record), []).append(
            {
                "relative_path": record.relative_path,
                "split": record.split,
                "class_name": record.class_name,
                "boundary_categories": sorted(boundary_categories),
                "roles": sorted(roles_by_path[relative_path]),
            }
        )

    collisions: list[dict[str, Any]] = []
    for content_hash, entries in sorted(hash_entries.items()):
        boundary_categories = sorted(
            {category for entry in entries for category in entry["boundary_categories"]}
        )
        unique_paths = {entry["relative_path"] for entry in entries}
        if len(unique_paths) <= 1:
            continue
        if len(boundary_categories) <= 1:
            continue
        collisions.append(
            {
                "content_hash": content_hash,
                "boundary_categories": boundary_categories,
                "entries": entries,
            }
        )
    return {
        "passed": not collisions,
        "fold_count": fold_manifest.fold_count,
        "files_per_class": fold_manifest.files_per_class,
        "unique_file_count": len(records_by_path),
        "collision_count": len(collisions),
        "collisions": collisions,
    }


def concatenate_file_score_bundles(
    bundles: tuple[FileScoreBundle, ...],
    *,
    aggregator: str,
) -> FileScoreBundle:
    if not bundles:
        raise M05Error("at least one file score bundle is required")
    reference = bundles[0]
    for bundle in bundles[1:]:
        if bundle.class_names != reference.class_names:
            raise M05Error("oof file score bundles disagree on class names")
    scores = np.concatenate([bundle.scores for bundle in bundles], axis=0)
    true_labels = np.concatenate([bundle.true_labels for bundle in bundles], axis=0)
    predicted_labels = np.concatenate([bundle.predicted_labels for bundle in bundles], axis=0)
    topk_indices = np.concatenate([bundle.topk_indices for bundle in bundles], axis=0)
    split_names = tuple(value for bundle in bundles for value in bundle.split_names)
    relative_paths = tuple(value for bundle in bundles for value in bundle.relative_paths)
    absolute_paths = tuple(value for bundle in bundles for value in bundle.absolute_paths)
    num_windows = np.concatenate([bundle.num_windows for bundle in bundles], axis=0)
    if len(set(relative_paths)) != len(relative_paths):
        raise M05Error("oof file score bundles overlap on relative_path")
    order = np.argsort(np.asarray(relative_paths), kind="mergesort")
    return FileScoreBundle(
        aggregator=aggregator,
        class_names=reference.class_names,
        scores=np.asarray(scores[order], dtype=np.float64),
        true_labels=np.asarray(true_labels[order], dtype=np.int64),
        predicted_labels=np.asarray(predicted_labels[order], dtype=np.int64),
        topk_indices=np.asarray(topk_indices[order], dtype=np.int64),
        split_names=tuple(split_names[index] for index in order),
        relative_paths=tuple(relative_paths[index] for index in order),
        absolute_paths=tuple(absolute_paths[index] for index in order),
        num_windows=np.asarray(num_windows[order], dtype=np.int64),
    )


def round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def build_epoch_budget(epochs: tuple[int, ...], *, max_epochs: int) -> int:
    if not epochs:
        raise M05Error("epoch budget selection requires at least one fold epoch")
    return max(1, min(max_epochs, round_half_up(float(np.mean(np.asarray(epochs, dtype=float))))))


def serialize_metrics(metrics: Any) -> dict[str, float]:
    return {
        "acc@1": float(metrics.acc_at_1),
        "acc@5": float(metrics.acc_at_5),
        "precision_macro": float(metrics.precision_macro),
        "recall_macro": float(metrics.recall_macro),
        "f1_macro": float(metrics.f1_macro),
    }


def build_training_rows_for_fixed_refit(history: list[dict[str, float]]) -> list[dict[str, str]]:
    return [
        {
            "epoch": str(int(row["epoch"])),
            "learning_rate": str(float(row["learning_rate"])),
            "train_loss": str(float(row["train_loss"])),
            "train_acc@1": str(float(row["train_acc_at_1"])),
        }
        for row in history
    ]


def run_fold_model_cv(
    *,
    entry: M05BankEntry,
    dataset: BaseSensorDataset,
    fold: Any,
    category_mapping: dict[str, str],
    output_root: Path,
) -> FoldModelResult:
    config = entry.config
    set_seed(config.seed)
    device = resolve_device(config.device)
    prepared = prepare_moonshot_tensors_from_records(
        class_names=tuple(sorted(dataset.class_vocab)),
        resolved_data_root=dataset.resolved_data_root,
        train_records=fold.train_records,
        validation_records=fold.validation_records,
        test_records=(),
        config=config,
        validation_files_per_class=1,
        view_manifest_updates={
            "protocol": "m05",
            "split_strategy": "grouped_5_fold_cv",
            "fold_index": fold.fold_index,
            "fold_count": fold.fold_count,
        },
    )
    run_payload = {
        **config.to_dict(),
        "protocol": "m05",
        "fold_index": fold.fold_index,
        "fold_count": fold.fold_count,
        "test_evaluation_enabled": False,
    }
    fold_output_root = output_root / f"fold_{fold.fold_index}" / "runs"
    run_dir = build_run_dir(
        fold_output_root,
        f"{entry.member_id}_m05_fold{fold.fold_index}",
        run_payload,
    )
    run_dir.mkdir(parents=True, exist_ok=False)

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
    training_result = train_moonshot_classifier(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        validation_windows=require_standardized_split(
            prepared.standardized_validation_split,
            split_name="validation",
        ).windows,
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
        checkpoint_dir=run_dir / "checkpoint_candidates",
        config_payload=run_payload,
    )
    candidate_results: dict[str, FoldAggregatorResult] = {}
    for candidate in sorted(
        training_result["best_candidates"],
        key=lambda item: item["aggregator"],
    ):
        aggregator = str(candidate["aggregator"])
        checkpoint_path = Path(candidate["checkpoint_path"])
        _checkpoint, evaluation, bundle = evaluate_moonshot_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            data_loader=validation_loader,
            windows=require_standardized_split(
                prepared.standardized_validation_split,
                split_name="validation",
            ).windows,
            class_names=prepared.class_names,
            category_mapping=category_mapping,
            device=device,
        )
        file_result = aggregate_file_level_metrics(
            bundle=bundle,
            category_mapping=category_mapping,
            aggregator=aggregator,
        )
        file_score_bundle = build_file_score_bundle(bundle=bundle, aggregator=aggregator)
        candidate_dir = run_dir / "validation_candidates" / aggregator
        candidate_dir.mkdir(parents=True, exist_ok=True)
        prediction_bundle_path = candidate_dir / "validation_predictions.npz"
        file_score_bundle_path = candidate_dir / "validation_file_scores.npz"
        from smelt.evaluation import write_window_prediction_bundle

        write_window_prediction_bundle(prediction_bundle_path, bundle)
        write_file_score_bundle(file_score_bundle_path, file_score_bundle)
        report_paths = export_file_level_report(
            output_root=run_dir / "validation_file_level",
            run_name=aggregator,
            result=file_result,
            methods_summary={
                "track": "moonshot-enhanced-setting",
                "protocol": "m05",
                "split": "cv_validation",
                "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
                "candidate_method": aggregator,
                "member_id": entry.member_id,
                "fold_index": fold.fold_index,
                "view_mode": "diff_all12",
                "channel_set": "all12",
            },
        )
        candidate_results[aggregator] = FoldAggregatorResult(
            aggregator=aggregator,
            epoch=int(candidate["epoch"]),
            checkpoint_path=checkpoint_path,
            validation_window_metrics=evaluation.metrics,
            validation_file_result=file_result,
            validation_file_score_bundle=file_score_bundle,
            validation_prediction_bundle_path=prediction_bundle_path,
            validation_file_score_bundle_path=file_score_bundle_path,
            validation_report_paths=report_paths.to_dict(),
        )

    fold_selection_path = run_dir / "m05_fold_selection.json"
    selection_payload = {
        "protocol": "m05",
        "fold_index": fold.fold_index,
        "fold_count": fold.fold_count,
        "member_id": entry.member_id,
        "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
        "selected_aggregator": training_result["locked_primary_aggregator"],
        "selected_epoch": int(training_result["best_selection"].epoch),
        "candidates": [
            {
                "aggregator": aggregator,
                "epoch": result.epoch,
                "validation_window_metrics": serialize_metrics(result.validation_window_metrics),
                "validation_file_metrics": serialize_metrics(result.validation_file_result.metrics),
                "validation_prediction_bundle_path": str(
                    result.validation_prediction_bundle_path.resolve()
                ),
                "validation_file_score_bundle_path": str(
                    result.validation_file_score_bundle_path.resolve()
                ),
                "validation_report_paths": result.validation_report_paths,
            }
            for aggregator, result in sorted(candidate_results.items())
        ],
    }
    write_json(fold_selection_path, selection_payload)
    history_path = run_dir / "training_history.json"
    write_json(
        history_path,
        {
            "rows": [
                {
                    "epoch": row.epoch,
                    "learning_rate": row.learning_rate,
                    "train_loss": row.train_loss,
                    "train_acc@1": row.train_acc_at_1,
                    "validation_window_acc@1": row.validation_window_acc_at_1,
                    "validation_window_f1": row.validation_window_f1,
                    "validation_primary_aggregator": row.validation_primary_aggregator,
                    "validation_file_acc@1": row.validation_file_acc_at_1,
                    "validation_file_f1": row.validation_file_f1,
                    "is_best": row.is_best,
                }
                for row in training_result["history"]
            ]
        },
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(run_payload, sort_keys=True),
        encoding="utf-8",
    )
    write_run_metadata(
        run_dir / "run_metadata.json",
        {
            "track": "moonshot-enhanced-setting",
            "protocol": "m05",
            "mode": "grouped_cv_fold_model",
            "member_id": entry.member_id,
            "model_family": config.model_name,
            "device": str(device),
            "parameter_count": int(architecture_summary["parameter_count"]),
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
            "locked_primary_aggregator": training_result["locked_primary_aggregator"],
            "fold_index": fold.fold_index,
            "fold_count": fold.fold_count,
            "official_test_evaluated": False,
            "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
            "fold_selection_path": str(fold_selection_path.resolve()),
        },
    )
    write_base_class_vocab_manifest(
        run_dir / "base_class_vocab.json",
        build_base_class_vocab_manifest(dataset),
    )
    from smelt.datasets import write_moonshot_view_manifest

    write_moonshot_view_manifest(run_dir / "moonshot_view_manifest.json", prepared.view_manifest)
    return FoldModelResult(
        member_id=entry.member_id,
        fold_index=fold.fold_index,
        run_dir=run_dir,
        selected_aggregator=str(training_result["locked_primary_aggregator"]),
        selected_epoch=int(training_result["best_selection"].epoch),
        candidate_results=candidate_results,
        history_path=history_path,
    )


def select_model_aggregator(
    summaries: dict[str, AggregatorCvSummary],
) -> str:
    tie_order = {name: index for index, name in enumerate(LOCKED_AGGREGATOR_TIEBREAK_ORDER)}
    return min(
        summaries.values(),
        key=lambda summary: (
            -summary.oof_result.metrics.acc_at_1,
            -summary.oof_result.metrics.f1_macro,
            tie_order.get(summary.aggregator, len(LOCKED_AGGREGATOR_TIEBREAK_ORDER)),
        ),
    ).aggregator


def build_model_cv_summary(
    *,
    entry: M05BankEntry,
    fold_results: tuple[FoldModelResult, ...],
    category_mapping: dict[str, str],
    output_root: Path,
) -> ModelCvSummary:
    aggregator_summaries: dict[str, AggregatorCvSummary] = {}
    aggregator_order = tuple(entry.config.candidate_file_aggregators)
    member_root = output_root / entry.member_id
    member_root.mkdir(parents=True, exist_ok=True)
    for aggregator in aggregator_order:
        fold_candidates = tuple(
            fold_result.candidate_results[aggregator] for fold_result in fold_results
        )
        oof_bundle = concatenate_file_score_bundles(
            tuple(candidate.validation_file_score_bundle for candidate in fold_candidates),
            aggregator=aggregator,
        )
        oof_result = compute_file_result_from_score_bundle(
            oof_bundle,
            category_mapping=category_mapping,
            aggregator_name=aggregator,
        )
        score_bundle_path = member_root / f"oof_file_scores_{aggregator}.npz"
        write_file_score_bundle(score_bundle_path, oof_bundle)
        report_paths = export_file_level_report(
            output_root=member_root / "oof_file_level",
            run_name=aggregator,
            result=oof_result,
            methods_summary={
                "track": "moonshot-enhanced-setting",
                "protocol": "m05",
                "split": "cv_oof",
                "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
                "member_id": entry.member_id,
                "candidate_method": aggregator,
                "view_mode": "diff_all12",
                "channel_set": "all12",
            },
        )
        acc_values = np.asarray(
            [candidate.validation_file_result.metrics.acc_at_1 for candidate in fold_candidates],
            dtype=np.float64,
        )
        f1_values = np.asarray(
            [candidate.validation_file_result.metrics.f1_macro for candidate in fold_candidates],
            dtype=np.float64,
        )
        epochs = tuple(candidate.epoch for candidate in fold_candidates)
        aggregator_summaries[aggregator] = AggregatorCvSummary(
            aggregator=aggregator,
            oof_result=oof_result,
            oof_score_bundle=oof_bundle,
            oof_score_bundle_path=score_bundle_path,
            oof_report_paths=report_paths.to_dict(),
            fold_mean_file_acc_at_1=float(acc_values.mean()),
            fold_std_file_acc_at_1=float(acc_values.std(ddof=0)),
            fold_mean_file_macro_f1=float(f1_values.mean()),
            fold_std_file_macro_f1=float(f1_values.std(ddof=0)),
            epoch_mean=float(np.mean(np.asarray(epochs, dtype=np.float64))),
            epoch_std=float(np.std(np.asarray(epochs, dtype=np.float64), ddof=0)),
            epoch_budget=build_epoch_budget(epochs, max_epochs=entry.config.epochs),
            fold_epochs=epochs,
        )
    selected_aggregator = select_model_aggregator(aggregator_summaries)
    selection_report_path = member_root / "aggregator_selection.json"
    write_json(
        selection_report_path,
        {
            "member_id": entry.member_id,
            "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
            "selection_rule": {
                "primary": "cv_oof_file_acc@1",
                "tie_break": "cv_oof_file_macro_f1",
                "final_tie_break_order": list(LOCKED_AGGREGATOR_TIEBREAK_ORDER),
            },
            "selected_aggregator": selected_aggregator,
            "candidates": [
                {
                    "aggregator": aggregator,
                    "cv_oof_file_metrics": serialize_metrics(summary.oof_result.metrics),
                    "fold_mean_file_acc@1": summary.fold_mean_file_acc_at_1,
                    "fold_std_file_acc@1": summary.fold_std_file_acc_at_1,
                    "fold_mean_file_macro_f1": summary.fold_mean_file_macro_f1,
                    "fold_std_file_macro_f1": summary.fold_std_file_macro_f1,
                    "fold_epochs": list(summary.fold_epochs),
                    "epoch_mean": summary.epoch_mean,
                    "epoch_std": summary.epoch_std,
                    "frozen_epoch_budget": summary.epoch_budget,
                    "oof_score_bundle_path": str(summary.oof_score_bundle_path.resolve()),
                    "oof_report_paths": summary.oof_report_paths,
                }
                for aggregator, summary in sorted(aggregator_summaries.items())
            ],
        },
    )
    return ModelCvSummary(
        entry=entry,
        fold_results=fold_results,
        aggregator_summaries=aggregator_summaries,
        selected_aggregator=selected_aggregator,
        selection_report_path=selection_report_path,
    )


def model_bank_rank_key(summary: ModelCvSummary) -> tuple[float, float, int]:
    selected = summary.selected_summary
    return (
        -selected.oof_result.metrics.acc_at_1,
        -selected.oof_result.metrics.f1_macro,
        summary.entry.stable_order,
    )


def build_model_bank_rows(
    *,
    summaries: tuple[ModelCvSummary, ...],
    selected_member_ids: tuple[str, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rank, summary in enumerate(sorted(summaries, key=model_bank_rank_key), start=1):
        selected = summary.selected_summary
        rows.append(
            {
                "rank": str(rank),
                "member_id": summary.entry.member_id,
                "config_path": str(summary.entry.config_path),
                "model_family": summary.entry.config.model_name,
                "selected_aggregator": summary.selected_aggregator,
                "selected_epoch_budget": str(selected.epoch_budget),
                "cv_oof_file_acc@1": str(selected.oof_result.metrics.acc_at_1),
                "cv_oof_file_acc@5": str(selected.oof_result.metrics.acc_at_5),
                "cv_oof_file_macro_f1": str(selected.oof_result.metrics.f1_macro),
                "fold_mean_file_acc@1": str(selected.fold_mean_file_acc_at_1),
                "fold_std_file_acc@1": str(selected.fold_std_file_acc_at_1),
                "fold_mean_file_macro_f1": str(selected.fold_mean_file_macro_f1),
                "fold_std_file_macro_f1": str(selected.fold_std_file_macro_f1),
                "aggregator_selection_json": str(summary.selection_report_path.resolve()),
                "oof_score_bundle_path": str(selected.oof_score_bundle_path.resolve()),
                "oof_summary_json": selected.oof_report_paths.get("summary_json", ""),
                "selected_for_final_bank": str(
                    summary.entry.member_id in selected_member_ids
                ).lower(),
            }
        )
    return rows


def build_m05_ensemble_rows(
    *,
    candidates: dict[str, Any],
    selected_method: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method_name in M04_ENSEMBLE_METHODS:
        candidate = candidates[method_name]
        rows.append(
            {
                "method_name": method_name,
                "member_ids": json.dumps(list(candidate.member_run_ids)),
                "weights": json.dumps([round(weight, 8) for weight in candidate.weights]),
                "cv_file_acc@1": str(candidate.validation_result.metrics.acc_at_1),
                "cv_file_acc@5": str(candidate.validation_result.metrics.acc_at_5),
                "cv_file_macro_f1": str(candidate.validation_result.metrics.f1_macro),
                "avg_pairwise_agreement": str(candidate.avg_pairwise_agreement),
                "avg_pairwise_correlation": str(candidate.avg_pairwise_correlation),
                "selected": str(method_name == selected_method).lower(),
            }
        )
    return rows


def build_m05_ensemble_selection_payload(
    *,
    candidates: dict[str, Any],
    selected_method: str,
) -> dict[str, Any]:
    selected = candidates[selected_method]
    return {
        "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
        "selection_rule": {
            "primary": "cv_oof_file_acc@1",
            "tie_break": "cv_oof_file_macro_f1",
            "secondary_tie_break": (
                "lower_average_pairwise_agreement_then_lower_probability_correlation"
            ),
            "final_tie_break_order": list(M04_ENSEMBLE_PREFERENCE),
        },
        "selected_method": selected_method,
        "selected_member_ids": list(selected.member_run_ids),
        "selected_weights": list(selected.weights),
        "selected_cv_file_metrics": serialize_metrics(selected.validation_result.metrics),
        "selected_avg_pairwise_agreement": selected.avg_pairwise_agreement,
        "selected_avg_pairwise_correlation": selected.avg_pairwise_correlation,
        "candidates": [
            {
                "method_name": method_name,
                "member_ids": list(candidates[method_name].member_run_ids),
                "weights": list(candidates[method_name].weights),
                "cv_file_metrics": serialize_metrics(
                    candidates[method_name].validation_result.metrics
                ),
                "avg_pairwise_agreement": candidates[method_name].avg_pairwise_agreement,
                "avg_pairwise_correlation": candidates[method_name].avg_pairwise_correlation,
            }
            for method_name in M04_ENSEMBLE_METHODS
        ],
    }


def build_m05_refit_plan(
    *,
    summaries: dict[str, ModelCvSummary],
    selected_method: str,
    selected_member_ids: tuple[str, ...],
    selected_weights: tuple[float, ...],
) -> tuple[FrozenRefitMember, ...]:
    if len(selected_member_ids) != len(selected_weights):
        raise M05Error("selected member ids and weights must be aligned")
    plan: list[FrozenRefitMember] = []
    for member_id, selected_weight in zip(selected_member_ids, selected_weights, strict=True):
        summary = summaries[member_id]
        selected = summary.selected_summary
        plan.append(
            FrozenRefitMember(
                member_id=member_id,
                config_path=summary.entry.config_path,
                config=summary.entry.config,
                stable_order=summary.entry.stable_order,
                selected_aggregator=summary.selected_aggregator,
                epoch_budget=selected.epoch_budget,
                selected_weight=float(selected_weight),
                oof_file_acc_at_1=float(selected.oof_result.metrics.acc_at_1),
                oof_file_macro_f1=float(selected.oof_result.metrics.f1_macro),
            )
        )
    return tuple(plan)


def train_moonshot_fixed_epochs(
    *,
    model: nn.Module,
    train_loader: Any,
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
) -> list[dict[str, float]]:
    if gradient_accumulation_steps <= 0:
        raise M05Error("gradient_accumulation_steps must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        scheduler_t_max=scheduler_t_max,
        scheduler_eta_min=scheduler_eta_min,
    )
    history: list[dict[str, float]] = []
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
        history.append(
            {
                "epoch": float(epoch),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(total_loss / max(total_examples, 1)),
                "train_acc_at_1": float(100.0 * total_correct / max(total_examples, 1)),
            }
        )
        if scheduler is not None:
            scheduler.step()
    return history


def resolve_ensemble_mode(method_name: str) -> str:
    if method_name == "mean_logits_all":
        return "logits"
    return "probabilities"


def run_full_refit_member(
    *,
    plan: FrozenRefitMember,
    dataset: BaseSensorDataset,
    category_mapping: dict[str, str],
    output_root: Path,
) -> RefitMemberResult:
    config = plan.config
    set_seed(config.seed)
    device = resolve_device(config.device)
    prepared = prepare_moonshot_tensors_from_records(
        class_names=tuple(sorted(dataset.class_vocab)),
        resolved_data_root=dataset.resolved_data_root,
        train_records=dataset.train_records,
        validation_records=(),
        test_records=dataset.test_records,
        config=config,
        validation_files_per_class=0,
        view_manifest_updates={
            "protocol": "m05",
            "split_strategy": "full_train_refit",
            "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
            "frozen_epoch_budget": plan.epoch_budget,
            "frozen_aggregator": plan.selected_aggregator,
        },
    )
    refit_payload = {
        **config.to_dict(),
        "protocol": "m05",
        "split_strategy": "full_train_refit",
        "frozen_epoch_budget": plan.epoch_budget,
        "frozen_aggregator": plan.selected_aggregator,
    }
    run_dir = build_run_dir(output_root, f"{plan.member_id}_m05_full_refit", refit_payload)
    run_dir.mkdir(parents=True, exist_ok=False)
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
    history = train_moonshot_fixed_epochs(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=plan.epoch_budget,
        lr=config.lr,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        label_smoothing=config.label_smoothing,
        scheduler_name=config.scheduler_name,
        scheduler_t_max=max(plan.epoch_budget, 1),
        scheduler_eta_min=config.scheduler_eta_min,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    checkpoint_path = run_dir / "checkpoint_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": plan.epoch_budget,
            "config": refit_payload,
            "locked_primary_aggregator": plan.selected_aggregator,
            "frozen_epoch_budget": plan.epoch_budget,
        },
        checkpoint_path,
    )
    test_loader = build_dataloader(
        require_split_array(prepared.test_windows, split_name="test"),
        require_split_array(prepared.test_labels, split_name="test"),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    _checkpoint, evaluation, bundle = evaluate_moonshot_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        data_loader=test_loader,
        windows=require_standardized_split(
            prepared.standardized_test_split,
            split_name="test",
        ).windows,
        class_names=prepared.class_names,
        category_mapping=category_mapping,
        device=device,
    )
    file_score_bundle = build_file_score_bundle(bundle=bundle, aggregator=plan.selected_aggregator)
    write_json(
        run_dir / "training_history.json",
        {"rows": build_training_rows_for_fixed_refit(history)},
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(refit_payload, sort_keys=True),
        encoding="utf-8",
    )
    write_run_metadata(
        run_dir / "run_metadata.json",
        {
            "track": "moonshot-enhanced-setting",
            "protocol": "m05",
            "mode": "full_train_refit_member",
            "member_id": plan.member_id,
            "model_family": config.model_name,
            "device": str(device),
            "parameter_count": int(architecture_summary["parameter_count"]),
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
            "official_test_evaluated": True,
            "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
            "locked_primary_aggregator": plan.selected_aggregator,
            "frozen_epoch_budget": plan.epoch_budget,
            "official_test_metrics_persisted": False,
            "official_test_scope": "used only to build the final frozen ensemble output",
        },
    )
    from smelt.datasets import write_moonshot_view_manifest

    write_moonshot_view_manifest(run_dir / "moonshot_view_manifest.json", prepared.view_manifest)
    return RefitMemberResult(
        plan=plan,
        run_dir=run_dir,
        test_file_score_bundle=file_score_bundle,
    )


def load_current_tracked_rows(table_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    m01c_path = table_root / "m01c_seed_summary.json"
    if m01c_path.is_file():
        payload = json.loads(m01c_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "protocol_id": "m01c",
                "selection_protocol": (
                    "single grouped holdout, locked per-run aggregator/checkpoint"
                ),
                "official_test_hygiene": "single-run test only; no cv",
                "file_acc@1": str(payload.get("file_acc@1_locked", {}).get("mean", "")),
                "file_macro_f1": str(payload.get("file_macro_f1_locked", {}).get("mean", "")),
            }
        )
    m03_path = table_root / "m03_ensemble_summary.json"
    if m03_path.is_file():
        payload = json.loads(m03_path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        rows.append(
            {
                "protocol_id": "m03",
                "selection_protocol": "validation-selected cross-seed ensemble",
                "official_test_hygiene": (
                    "search still evaluates all methods on official test after selection"
                ),
                "file_acc@1": row.get("file_acc@1", ""),
                "file_macro_f1": row.get("file_macro_f1", ""),
            }
        )
    m04_path = table_root / "m04_final_comparison.json"
    if m04_path.is_file():
        payload = json.loads(m04_path.read_text(encoding="utf-8"))
        row = payload["rows"][0]
        rows.append(
            {
                "protocol_id": "m04",
                "selection_protocol": "validation-selected heterogeneous ensemble bank",
                "official_test_hygiene": (
                    "candidate-level official-test metrics existed during search/output"
                ),
                "file_acc@1": row.get("ensemble_file_acc@1", ""),
                "file_macro_f1": row.get("ensemble_file_macro_f1", ""),
            }
        )
    return rows


def run_m05_cv_search(
    *,
    entries: tuple[M05BankEntry, ...],
    dataset: BaseSensorDataset,
    fold_manifest: Any,
    category_mapping: dict[str, str],
    output_root: Path,
) -> tuple[dict[str, ModelCvSummary], dict[str, Any], dict[str, Any]]:
    summaries: dict[str, ModelCvSummary] = {}
    cv_root = output_root / "cv"
    for entry in entries:
        fold_results = tuple(
            run_fold_model_cv(
                entry=entry,
                dataset=dataset,
                fold=fold,
                category_mapping=category_mapping,
                output_root=cv_root / "fold_runs" / entry.member_id,
            )
            for fold in fold_manifest.folds
        )
        summaries[entry.member_id] = build_model_cv_summary(
            entry=entry,
            fold_results=fold_results,
            category_mapping=category_mapping,
            output_root=cv_root / "model_bank",
        )
    validation_bundles = {
        member_id: summary.selected_summary.oof_score_bundle
        for member_id, summary in summaries.items()
    }
    helper_selection_payload, candidates = evaluate_cv_ensemble_candidates(
        validation_bundles=validation_bundles,
        category_mapping=category_mapping,
    )
    selected_method = str(helper_selection_payload["selected_method"])
    selection_payload = build_m05_ensemble_selection_payload(
        candidates=candidates,
        selected_method=selected_method,
    )
    return summaries, selection_payload, candidates


def run_m05_protocol(
    *,
    config_paths: tuple[Path, ...],
    output_root: Path,
    table_root: Path,
) -> dict[str, Any]:
    entries = load_m05_bank_entries(config_paths)
    shared_category_map = Path(entries[0].config.category_map_path).resolve()
    shared_class_vocab_manifest = Path(entries[0].config.class_vocab_manifest_path).resolve()
    validate_required_reference(shared_category_map)
    validate_required_reference(shared_class_vocab_manifest)
    for entry in entries:
        validate_required_reference(Path(entry.config.exact_upstream_regression_path).resolve())

    dataset = load_base_sensor_dataset(Path(entries[0].config.data_root))
    vocab_manifest = build_base_class_vocab_manifest(dataset)
    validate_reference_manifest(str(shared_class_vocab_manifest), vocab_manifest.to_dict())
    category_mapping = load_category_mapping(shared_category_map)
    fold_manifest = build_grouped_cv_fold_manifest(dataset.train_records, fold_count=M05_FOLD_COUNT)
    duplicate_audit = build_m05_duplicate_audit(dataset=dataset, fold_manifest=fold_manifest)

    run_payload = {
        "protocol": "m05",
        "fold_count": M05_FOLD_COUNT,
        "bank_config_paths": [str(path) for path in config_paths],
        "channel_set": "all12",
        "diff_period": 25,
        "window_size": 100,
        "stride": 50,
    }
    run_dir = build_run_dir(output_root, "m05_grouped_cv_refit", run_payload)
    run_dir.mkdir(parents=True, exist_ok=False)
    table_root.mkdir(parents=True, exist_ok=True)

    fold_manifest_path = table_root / "m05_fold_manifest.json"
    duplicate_audit_path = table_root / "m05_duplicate_audit.json"
    write_json(fold_manifest_path, fold_manifest.to_dict())
    write_json(duplicate_audit_path, duplicate_audit)
    write_json(run_dir / "m05_fold_manifest.json", fold_manifest.to_dict())
    write_json(run_dir / "m05_duplicate_audit.json", duplicate_audit)
    if not duplicate_audit["passed"]:
        raise M05Error(
            "m05 duplicate-content audit failed with "
            f"{duplicate_audit['collision_count']} collisions"
        )

    summaries, selection_payload, candidates = run_m05_cv_search(
        entries=entries,
        dataset=dataset,
        fold_manifest=fold_manifest,
        category_mapping=category_mapping,
        output_root=run_dir,
    )
    selected_method = str(selection_payload["selected_method"])
    selected_member_ids = tuple(str(value) for value in selection_payload["selected_member_ids"])
    selected_weights = tuple(float(value) for value in selection_payload["selected_weights"])

    model_bank_rows = build_model_bank_rows(
        summaries=tuple(summaries.values()),
        selected_member_ids=selected_member_ids,
    )
    ensemble_rows = build_m05_ensemble_rows(
        candidates=candidates,
        selected_method=selected_method,
    )

    model_bank_csv = table_root / "m05_cv_model_bank.csv"
    model_bank_json = table_root / "m05_cv_model_bank.json"
    ensemble_selection_csv = table_root / "m05_cv_ensemble_selection.csv"
    ensemble_selection_json = table_root / "m05_cv_ensemble_selection.json"
    search_summary_csv = table_root / "m05_cv_search_summary.csv"
    search_summary_json = table_root / "m05_cv_search_summary.json"

    write_dict_rows_csv(model_bank_csv, model_bank_rows)
    write_json(model_bank_json, {"rows": model_bank_rows})
    write_dict_rows_csv(run_dir / "m05_cv_model_bank.csv", model_bank_rows)
    write_json(run_dir / "m05_cv_model_bank.json", {"rows": model_bank_rows})
    selected_candidate = candidates[selected_method]
    write_dict_rows_csv(
        ensemble_selection_csv,
        [
            {
                "selected_method": selected_method,
                "selected_member_ids": json.dumps(list(selected_member_ids)),
                "selected_weights": json.dumps(list(selected_weights)),
                "cv_file_acc@1": str(selected_candidate.validation_result.metrics.acc_at_1),
                "cv_file_acc@5": str(selected_candidate.validation_result.metrics.acc_at_5),
                "cv_file_macro_f1": str(selected_candidate.validation_result.metrics.f1_macro),
                "avg_pairwise_agreement": str(selected_candidate.avg_pairwise_agreement),
                "avg_pairwise_correlation": str(selected_candidate.avg_pairwise_correlation),
            }
        ],
    )
    write_json(ensemble_selection_json, selection_payload)
    write_dict_rows_csv(search_summary_csv, ensemble_rows)
    write_json(search_summary_json, {"rows": ensemble_rows})
    write_dict_rows_csv(
        run_dir / "m05_cv_ensemble_selection.csv",
        [
            {
                "selected_method": selected_method,
                "selected_member_ids": json.dumps(list(selected_member_ids)),
                "selected_weights": json.dumps(list(selected_weights)),
                "cv_file_acc@1": str(selected_candidate.validation_result.metrics.acc_at_1),
                "cv_file_acc@5": str(selected_candidate.validation_result.metrics.acc_at_5),
                "cv_file_macro_f1": str(selected_candidate.validation_result.metrics.f1_macro),
                "avg_pairwise_agreement": str(selected_candidate.avg_pairwise_agreement),
                "avg_pairwise_correlation": str(selected_candidate.avg_pairwise_correlation),
            }
        ],
    )
    write_json(run_dir / "m05_cv_ensemble_selection.json", selection_payload)
    write_dict_rows_csv(run_dir / "m05_cv_search_summary.csv", ensemble_rows)
    write_json(run_dir / "m05_cv_search_summary.json", {"rows": ensemble_rows})

    refit_plan = build_m05_refit_plan(
        summaries=summaries,
        selected_method=selected_method,
        selected_member_ids=selected_member_ids,
        selected_weights=selected_weights,
    )
    refit_selection_path = run_dir / "m05_refit_plan.json"
    write_json(
        refit_selection_path,
        {
            "selected_method": selected_method,
            "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
            "members": [
                {
                    "member_id": plan.member_id,
                    "config_path": str(plan.config_path),
                    "selected_aggregator": plan.selected_aggregator,
                    "frozen_epoch_budget": plan.epoch_budget,
                    "selected_weight": plan.selected_weight,
                    "cv_oof_file_acc@1": plan.oof_file_acc_at_1,
                    "cv_oof_file_macro_f1": plan.oof_file_macro_f1,
                }
                for plan in refit_plan
            ],
        },
    )
    refit_results = tuple(
        run_full_refit_member(
            plan=plan,
            dataset=dataset,
            category_mapping=category_mapping,
            output_root=run_dir / "full_train_refit",
        )
        for plan in refit_plan
    )
    refit_bundles = {
        result.plan.member_id: result.test_file_score_bundle for result in refit_results
    }
    final_result = evaluate_score_ensemble(
        bundles=tuple(refit_bundles[member_id] for member_id in selected_member_ids),
        category_mapping=category_mapping,
        mode=resolve_ensemble_mode(selected_method),
        method_name=selected_method,
        weights=selected_weights,
    )
    final_report_paths = export_file_level_report(
        output_root=run_dir / "final_test",
        run_name=selected_method,
        result=final_result,
        methods_summary={
            "track": "moonshot-enhanced-setting",
            "protocol": "m05",
            "split": "official_test",
            "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
            "selected_method": selected_method,
            "selected_member_ids": list(selected_member_ids),
        },
    )

    final_row = {
        "protocol_id": "m05",
        "selected_method": selected_method,
        "selected_member_ids": json.dumps(list(selected_member_ids)),
        "selected_weights": json.dumps(list(selected_weights)),
        "file_acc@1": str(final_result.metrics.acc_at_1),
        "file_acc@5": str(final_result.metrics.acc_at_5),
        "file_macro_precision": str(final_result.metrics.precision_macro),
        "file_macro_recall": str(final_result.metrics.recall_macro),
        "file_macro_f1": str(final_result.metrics.f1_macro),
        "summary_json": final_report_paths.summary_json,
        "confusion_matrix_csv": final_report_paths.confusion_matrix_csv,
        "per_category_accuracy_csv": final_report_paths.per_category_accuracy_csv,
        "per_file_predictions_csv": final_report_paths.per_file_predictions_csv,
    }
    final_csv = table_root / "m05_final_test.csv"
    final_json = table_root / "m05_final_test.json"
    write_dict_rows_csv(final_csv, [final_row])
    write_json(final_json, {"rows": [final_row]})
    write_dict_rows_csv(run_dir / "m05_final_test.csv", [final_row])
    write_json(run_dir / "m05_final_test.json", {"rows": [final_row]})

    scorecard_rows = load_current_tracked_rows(table_root)
    scorecard_rows.append(
        {
            "protocol_id": "m05",
            "selection_protocol": (
                "grouped 5-fold cv/oof search, frozen aggregator + epoch budget, single final test"
            ),
            "official_test_hygiene": "no candidate-level official-test metrics before finalization",
            "file_acc@1": str(final_result.metrics.acc_at_1),
            "file_macro_f1": str(final_result.metrics.f1_macro),
        }
    )
    scorecard_csv = table_root / "m05_scorecard.csv"
    scorecard_json = table_root / "m05_scorecard.json"
    write_dict_rows_csv(scorecard_csv, scorecard_rows)
    write_json(scorecard_json, {"rows": scorecard_rows})
    write_dict_rows_csv(run_dir / "m05_scorecard.csv", scorecard_rows)
    write_json(run_dir / "m05_scorecard.json", {"rows": scorecard_rows})

    write_run_metadata(
        run_dir / "run_metadata.json",
        {
            "track": "moonshot-enhanced-setting",
            "protocol": "m05",
            "mode": "grouped_cv_search_then_full_train_refit",
            "selection_source": M05_ENSEMBLE_SELECTION_SOURCE,
            "selected_method": selected_method,
            "selected_member_ids": list(selected_member_ids),
            "selected_weights": list(selected_weights),
            "fold_manifest_path": str(fold_manifest_path.resolve()),
            "duplicate_audit_path": str(duplicate_audit_path.resolve()),
            "model_bank_path": str(model_bank_json.resolve()),
            "ensemble_selection_path": str(ensemble_selection_json.resolve()),
            "search_summary_path": str(search_summary_json.resolve()),
            "refit_plan_path": str(refit_selection_path.resolve()),
            "final_test_path": str(final_json.resolve()),
            "scorecard_path": str(scorecard_json.resolve()),
        },
    )
    write_base_class_vocab_manifest(run_dir / "base_class_vocab.json", vocab_manifest)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(run_payload, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "run_dir": run_dir,
        "selection_payload": selection_payload,
        "refit_plan": refit_plan,
        "refit_results": refit_results,
        "final_result": final_result,
        "model_bank_rows": model_bank_rows,
        "ensemble_rows": ensemble_rows,
        "scorecard_rows": scorecard_rows,
        "duplicate_audit": duplicate_audit,
    }
