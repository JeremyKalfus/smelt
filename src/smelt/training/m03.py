"""shared helpers for m03 frozen moonshot file-level work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from smelt.datasets import build_base_class_vocab_manifest, load_base_sensor_dataset
from smelt.evaluation import (
    MAJORITY_VOTE_AGGREGATOR,
    MEAN_LOGITS_AGGREGATOR,
    MEAN_PROBABILITIES_AGGREGATOR,
    WindowPredictionBundle,
    aggregate_file_level_metrics,
    build_file_level_result_from_predictions,
    select_validation_locked_aggregator,
    write_json,
)
from smelt.evaluation.file_level import AggregatorSelectionCandidate, group_indices_by_file

from .run import build_dataloader, resolve_device
from .run_moonshot import build_moonshot_model, load_moonshot_run_config, prepare_moonshot_tensors

M03_ENSEMBLE_METHODS = (
    MEAN_LOGITS_AGGREGATOR,
    MEAN_PROBABILITIES_AGGREGATOR,
    "vote",
)
M03_ENSEMBLE_PREFERENCE = (
    MEAN_PROBABILITIES_AGGREGATOR,
    MEAN_LOGITS_AGGREGATOR,
    "vote",
)
WINDOW_FEATURE_BUNDLE_NAME = "window_features.npz"


class M03Error(Exception):
    """raised when the m03 pipeline cannot proceed safely."""


@dataclass(slots=True)
class WindowFeatureBundle:
    class_names: tuple[str, ...]
    embeddings: NDArray[np.float32]
    logits: NDArray[np.float32]
    true_labels: NDArray[np.int64]
    predicted_labels: NDArray[np.int64]
    topk_indices: NDArray[np.int64]
    splits: tuple[str, ...]
    relative_paths: tuple[str, ...]
    absolute_paths: tuple[str, ...]
    window_indices: NDArray[np.int64]
    start_rows: NDArray[np.int64]
    stop_rows: NDArray[np.int64]

    def __post_init__(self) -> None:
        sample_count = int(self.true_labels.shape[0])
        if sample_count == 0:
            raise M03Error("window feature bundle must be non-empty")
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != sample_count:
            raise M03Error("embeddings shape does not match sample count")
        if self.logits.ndim != 2 or self.logits.shape[0] != sample_count:
            raise M03Error("logits shape does not match sample count")
        if self.logits.shape[1] != len(self.class_names):
            raise M03Error("logit class count does not match class_names")
        if self.predicted_labels.shape != (sample_count,):
            raise M03Error("predicted_labels shape does not match sample count")
        if self.topk_indices.ndim != 2 or self.topk_indices.shape[0] != sample_count:
            raise M03Error("topk_indices shape does not match sample count")
        if len(self.splits) != sample_count:
            raise M03Error("split metadata length does not match sample count")
        if len(self.relative_paths) != sample_count:
            raise M03Error("relative_paths length does not match sample count")
        if len(self.absolute_paths) != sample_count:
            raise M03Error("absolute_paths length does not match sample count")
        if self.window_indices.shape != (sample_count,):
            raise M03Error("window_indices shape does not match sample count")
        if self.start_rows.shape != (sample_count,):
            raise M03Error("start_rows shape does not match sample count")
        if self.stop_rows.shape != (sample_count,):
            raise M03Error("stop_rows shape does not match sample count")

    @property
    def sample_count(self) -> int:
        return int(self.true_labels.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def filter_split(self, split_name: str) -> WindowFeatureBundle:
        mask = np.asarray([value == split_name for value in self.splits], dtype=bool)
        if not mask.any():
            raise M03Error(f"bundle does not contain split {split_name!r}")
        indices = np.flatnonzero(mask)
        return WindowFeatureBundle(
            class_names=self.class_names,
            embeddings=np.asarray(self.embeddings[indices], dtype=np.float32),
            logits=np.asarray(self.logits[indices], dtype=np.float32),
            true_labels=np.asarray(self.true_labels[indices], dtype=np.int64),
            predicted_labels=np.asarray(self.predicted_labels[indices], dtype=np.int64),
            topk_indices=np.asarray(self.topk_indices[indices], dtype=np.int64),
            splits=tuple(self.splits[index] for index in indices),
            relative_paths=tuple(self.relative_paths[index] for index in indices),
            absolute_paths=tuple(self.absolute_paths[index] for index in indices),
            window_indices=np.asarray(self.window_indices[indices], dtype=np.int64),
            start_rows=np.asarray(self.start_rows[indices], dtype=np.int64),
            stop_rows=np.asarray(self.stop_rows[indices], dtype=np.int64),
        )

    def to_prediction_bundle(
        self,
        *,
        logits: NDArray[np.float64] | None = None,
        predicted_labels: NDArray[np.int64] | None = None,
        topk_indices: NDArray[np.int64] | None = None,
    ) -> WindowPredictionBundle:
        resolved_logits = (
            np.asarray(logits, dtype=np.float64)
            if logits is not None
            else np.asarray(self.logits, dtype=np.float64)
        )
        resolved_predicted = (
            np.asarray(predicted_labels, dtype=np.int64)
            if predicted_labels is not None
            else np.asarray(self.predicted_labels, dtype=np.int64)
        )
        resolved_topk = (
            np.asarray(topk_indices, dtype=np.int64)
            if topk_indices is not None
            else np.asarray(self.topk_indices, dtype=np.int64)
        )
        return WindowPredictionBundle(
            class_names=self.class_names,
            true_labels=np.asarray(self.true_labels, dtype=np.int64),
            predicted_labels=resolved_predicted,
            topk_indices=resolved_topk,
            logits=resolved_logits,
            splits=self.splits,
            relative_paths=self.relative_paths,
            absolute_paths=self.absolute_paths,
            window_indices=np.asarray(self.window_indices, dtype=np.int64),
            start_rows=np.asarray(self.start_rows, dtype=np.int64),
            stop_rows=np.asarray(self.stop_rows, dtype=np.int64),
        )


@dataclass(slots=True)
class FrozenEncoderSelection:
    selected_run_id: str
    selected_run_dir: str
    locked_primary_aggregator: str
    validation_file_acc_at_1: float
    validation_file_macro_f1: float
    candidates: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_run_id": self.selected_run_id,
            "selected_run_dir": self.selected_run_dir,
            "locked_primary_aggregator": self.locked_primary_aggregator,
            "validation_file_acc@1": self.validation_file_acc_at_1,
            "validation_file_macro_f1": self.validation_file_macro_f1,
            "selection_rule": {
                "primary": "validation_file_acc@1_under_locked_aggregator",
                "tie_break": "validation_file_macro_f1_under_locked_aggregator",
                "final_tie_break": "run_id_lexical_ascending",
                "source": "validation_only",
            },
            "candidates": list(self.candidates),
        }


def export_locked_moonshot_run_embeddings(
    *,
    run_dir: Path,
    output_root: Path,
) -> Path:
    config = load_moonshot_run_config(run_dir / "resolved_config.yaml")
    metadata = load_json_file(run_dir / "run_metadata.json")
    dataset = load_base_sensor_dataset(Path(config.data_root))
    vocab_manifest = build_base_class_vocab_manifest(dataset)
    prepared = prepare_moonshot_tensors(dataset, config)
    model, _ = build_moonshot_model(
        config=config,
        input_dim=prepared.train_windows.shape[2],
        num_classes=len(vocab_manifest.class_vocab),
    )
    device = resolve_device(config.device)
    model = model.to(device)
    load_checkpoint_weights(model, resolve_best_checkpoint_path(run_dir, metadata))

    train_bundle = collect_window_features(
        model=model,
        values=prepared.train_windows,
        labels=prepared.train_labels,
        windows=prepared.standardized_train_split.windows,
        split_name="train",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        class_names=prepared.class_names,
    )
    validation_bundle = collect_window_features(
        model=model,
        values=prepared.validation_windows,
        labels=prepared.validation_labels,
        windows=prepared.standardized_validation_split.windows,
        split_name="validation",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        class_names=prepared.class_names,
    )
    test_bundle = collect_window_features(
        model=model,
        values=prepared.test_windows,
        labels=prepared.test_labels,
        windows=prepared.standardized_test_split.windows,
        split_name="test",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        class_names=prepared.class_names,
    )
    combined_bundle = concatenate_feature_bundles(
        train_bundle,
        validation_bundle,
        test_bundle,
    )

    export_dir = output_root / run_dir.name
    export_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        export_dir / WINDOW_FEATURE_BUNDLE_NAME,
        class_names=np.asarray(combined_bundle.class_names),
        embeddings=combined_bundle.embeddings,
        logits=combined_bundle.logits,
        true_labels=combined_bundle.true_labels,
        predicted_labels=combined_bundle.predicted_labels,
        topk_indices=combined_bundle.topk_indices,
        splits=np.asarray(combined_bundle.splits),
        relative_paths=np.asarray(combined_bundle.relative_paths),
        absolute_paths=np.asarray(combined_bundle.absolute_paths),
        window_indices=combined_bundle.window_indices,
        start_rows=combined_bundle.start_rows,
        stop_rows=combined_bundle.stop_rows,
    )
    source_checkpoint_path = resolve_best_checkpoint_path(run_dir, metadata)
    write_json(
        export_dir / "export_metadata.json",
        {
            "source_run_id": run_dir.name,
            "source_run_dir": str(run_dir.resolve()),
            "source_checkpoint_path": str(source_checkpoint_path.resolve()),
            "track": metadata.get("track", ""),
            "view_mode": metadata.get("view_mode", ""),
            "channel_set": metadata.get("channel_set", ""),
            "locked_primary_aggregator": metadata.get("locked_primary_aggregator", ""),
            "embedding_dim": combined_bundle.embedding_dim,
            "class_count": len(combined_bundle.class_names),
            "sample_count": combined_bundle.sample_count,
            "split_counts": {
                "train": train_bundle.sample_count,
                "validation": validation_bundle.sample_count,
                "test": test_bundle.sample_count,
            },
            "device": str(device),
        },
    )
    return export_dir


def load_window_feature_bundle(path: Path) -> WindowFeatureBundle:
    payload = np.load(path, allow_pickle=False)
    required_fields = {
        "class_names",
        "embeddings",
        "logits",
        "true_labels",
        "predicted_labels",
        "topk_indices",
        "splits",
        "relative_paths",
        "absolute_paths",
        "window_indices",
        "start_rows",
        "stop_rows",
    }
    missing = sorted(required_fields - set(payload.files))
    if missing:
        raise M03Error(f"window feature bundle is missing fields: {missing}")
    return WindowFeatureBundle(
        class_names=tuple(str(value) for value in payload["class_names"].tolist()),
        embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
        logits=np.asarray(payload["logits"], dtype=np.float32),
        true_labels=np.asarray(payload["true_labels"], dtype=np.int64),
        predicted_labels=np.asarray(payload["predicted_labels"], dtype=np.int64),
        topk_indices=np.asarray(payload["topk_indices"], dtype=np.int64),
        splits=tuple(str(value) for value in payload["splits"].tolist()),
        relative_paths=tuple(str(value) for value in payload["relative_paths"].tolist()),
        absolute_paths=tuple(str(value) for value in payload["absolute_paths"].tolist()),
        window_indices=np.asarray(payload["window_indices"], dtype=np.int64),
        start_rows=np.asarray(payload["start_rows"], dtype=np.int64),
        stop_rows=np.asarray(payload["stop_rows"], dtype=np.int64),
    )


def select_primary_encoder_run(run_dirs: tuple[Path, ...]) -> FrozenEncoderSelection:
    candidates: list[dict[str, Any]] = []
    for run_dir in sorted(run_dirs, key=lambda path: path.name):
        metadata = load_json_file(run_dir / "run_metadata.json")
        summary = metadata.get("best_validation_summary", {})
        candidate = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir.resolve()),
            "locked_primary_aggregator": str(metadata.get("locked_primary_aggregator", "")),
            "validation_file_acc@1": float(summary.get("file_acc@1", 0.0)),
            "validation_file_macro_f1": float(summary.get("file_macro_f1", 0.0)),
        }
        candidates.append(candidate)
    if not candidates:
        raise M03Error("at least one locked moonshot run is required for primary selection")
    selected = min(
        candidates,
        key=lambda item: (
            -item["validation_file_acc@1"],
            -item["validation_file_macro_f1"],
            item["run_id"],
        ),
    )
    return FrozenEncoderSelection(
        selected_run_id=str(selected["run_id"]),
        selected_run_dir=str(selected["run_dir"]),
        locked_primary_aggregator=str(selected["locked_primary_aggregator"]),
        validation_file_acc_at_1=float(selected["validation_file_acc@1"]),
        validation_file_macro_f1=float(selected["validation_file_macro_f1"]),
        candidates=tuple(candidates),
    )


def select_ensemble_method(
    *,
    validation_results: dict[str, Any],
) -> str:
    candidates = [
        AggregatorSelectionCandidate(
            aggregator=method_name,
            acc_at_1=float(result.metrics.acc_at_1),
            f1_macro=float(result.metrics.f1_macro),
        )
        for method_name, result in validation_results.items()
    ]
    preferred = select_validation_locked_aggregator(
        tuple(
            AggregatorSelectionCandidate(
                aggregator=normalize_ensemble_method_name(candidate.aggregator),
                acc_at_1=candidate.acc_at_1,
                f1_macro=candidate.f1_macro,
            )
            for candidate in candidates
        )
    )
    if preferred == MAJORITY_VOTE_AGGREGATOR:
        return "vote"
    return preferred


def evaluate_locked_seed_ensemble(
    *,
    export_dirs: tuple[Path, ...],
    run_dirs: tuple[Path, ...],
    category_mapping: dict[str, str],
) -> dict[str, Any]:
    if len(export_dirs) != len(run_dirs):
        raise M03Error("export_dirs and run_dirs must be the same length")
    bundles = [
        load_window_feature_bundle(path / WINDOW_FEATURE_BUNDLE_NAME) for path in export_dirs
    ]
    locked_aggregators = [load_locked_primary_aggregator(run_dir) for run_dir in run_dirs]
    validation_results = {
        method: evaluate_ensemble_method(
            bundles=bundles,
            locked_aggregators=locked_aggregators,
            split_name="validation",
            category_mapping=category_mapping,
            method=method,
        )
        for method in M03_ENSEMBLE_METHODS
    }
    selected_method = select_ensemble_method(validation_results=validation_results)
    test_results = {
        method: evaluate_ensemble_method(
            bundles=bundles,
            locked_aggregators=locked_aggregators,
            split_name="test",
            category_mapping=category_mapping,
            method=method,
        )
        for method in M03_ENSEMBLE_METHODS
    }
    return {
        "selected_method": selected_method,
        "validation_results": validation_results,
        "test_results": test_results,
    }


def evaluate_ensemble_method(
    *,
    bundles: list[WindowFeatureBundle],
    locked_aggregators: list[str],
    split_name: str,
    category_mapping: dict[str, str],
    method: str,
) -> Any:
    split_bundles = [bundle.filter_split(split_name) for bundle in bundles]
    validate_aligned_feature_bundles(split_bundles)
    if method == "vote":
        return evaluate_seed_vote_ensemble(
            bundles=split_bundles,
            locked_aggregators=locked_aggregators,
            category_mapping=category_mapping,
        )
    stacked_logits = np.stack(
        [np.asarray(bundle.logits, dtype=np.float64) for bundle in split_bundles],
        axis=0,
    )
    if method == MEAN_LOGITS_AGGREGATOR:
        window_scores = stacked_logits.mean(axis=0)
        topk = stable_topk_per_row(window_scores)
        predicted = topk[:, 0]
        bundle = split_bundles[0].to_prediction_bundle(
            logits=window_scores,
            predicted_labels=predicted,
            topk_indices=topk,
        )
        return aggregate_file_level_metrics(
            bundle=bundle,
            category_mapping=category_mapping,
            aggregator=MEAN_LOGITS_AGGREGATOR,
        )
    if method == MEAN_PROBABILITIES_AGGREGATOR:
        probabilities = softmax_np(stacked_logits)
        averaged_probabilities = probabilities.mean(axis=0)
        pseudo_logits = np.log(np.clip(averaged_probabilities, 1e-12, 1.0))
        topk = stable_topk_per_row(averaged_probabilities)
        predicted = topk[:, 0]
        bundle = split_bundles[0].to_prediction_bundle(
            logits=pseudo_logits,
            predicted_labels=predicted,
            topk_indices=topk,
        )
        return aggregate_file_level_metrics(
            bundle=bundle,
            category_mapping=category_mapping,
            aggregator=MEAN_PROBABILITIES_AGGREGATOR,
        )
    raise M03Error(f"unsupported ensemble method {method!r}")


def evaluate_seed_vote_ensemble(
    *,
    bundles: list[WindowFeatureBundle],
    locked_aggregators: list[str],
    category_mapping: dict[str, str],
) -> Any:
    if len(bundles) != len(locked_aggregators):
        raise M03Error("bundles and locked_aggregators must be the same length")
    seed_results = [
        aggregate_file_level_metrics(
            bundle=bundle.to_prediction_bundle(),
            category_mapping=category_mapping,
            aggregator=aggregator,
        )
        for bundle, aggregator in zip(bundles, locked_aggregators, strict=True)
    ]
    reference_rows = {row.relative_path: row for row in seed_results[0].rows}
    class_names = bundles[0].class_names
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    topk_indices: list[NDArray[np.int64]] = []
    split_names: list[str] = []
    relative_paths: list[str] = []
    absolute_paths: list[str] = []
    num_windows: list[int] = []
    for relative_path in sorted(reference_rows):
        vote_scores = np.zeros(len(class_names), dtype=np.float64)
        for seed_result in seed_results:
            row_by_path = {row.relative_path: row for row in seed_result.rows}
            if relative_path not in row_by_path:
                raise M03Error(f"vote ensemble missing file {relative_path!r} in one seed result")
            vote_row = row_by_path[relative_path]
            vote_scores[class_to_index[vote_row.predicted_class]] += 1.0
        topk = stable_descending_topk(vote_scores, min(5, len(class_names)))
        reference_row = reference_rows[relative_path]
        true_labels.append(class_to_index[reference_row.true_class])
        predicted_labels.append(int(topk[0]))
        topk_indices.append(topk)
        split_names.append(reference_row.split)
        relative_paths.append(reference_row.relative_path)
        absolute_paths.append(reference_row.absolute_path)
        num_windows.append(reference_row.num_windows)
    return build_file_level_result_from_predictions(
        aggregator="vote",
        class_names=class_names,
        true_labels=np.asarray(true_labels, dtype=np.int64),
        predicted_labels=np.asarray(predicted_labels, dtype=np.int64),
        topk_indices=np.stack(topk_indices, axis=0),
        split_names=tuple(split_names),
        relative_paths=tuple(relative_paths),
        absolute_paths=tuple(absolute_paths),
        num_windows=np.asarray(num_windows, dtype=np.int64),
        category_mapping=category_mapping,
    )


def validate_aligned_feature_bundles(bundles: list[WindowFeatureBundle]) -> None:
    if not bundles:
        raise M03Error("at least one feature bundle is required")
    reference = bundles[0]
    for bundle in bundles[1:]:
        if bundle.class_names != reference.class_names:
            raise M03Error("feature bundles disagree on class_names")
        comparisons = {
            "true_labels": np.array_equal(bundle.true_labels, reference.true_labels),
            "splits": bundle.splits == reference.splits,
            "relative_paths": bundle.relative_paths == reference.relative_paths,
            "absolute_paths": bundle.absolute_paths == reference.absolute_paths,
            "window_indices": np.array_equal(bundle.window_indices, reference.window_indices),
            "start_rows": np.array_equal(bundle.start_rows, reference.start_rows),
            "stop_rows": np.array_equal(bundle.stop_rows, reference.stop_rows),
        }
        mismatches = sorted(name for name, matched in comparisons.items() if not matched)
        if mismatches:
            raise M03Error(f"feature bundles are not aligned: {mismatches}")


def collect_window_features(
    *,
    model: nn.Module,
    values: NDArray[np.float32],
    labels: NDArray[np.int64],
    windows: tuple[Any, ...],
    split_name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    class_names: tuple[str, ...],
) -> WindowFeatureBundle:
    if len(windows) != int(values.shape[0]):
        raise M03Error("window metadata does not match value tensor rows")
    if not hasattr(model, "forward_features"):
        raise M03Error("model does not expose forward_features for frozen export")
    data_loader = build_dataloader(
        values,
        labels,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    embeddings_batches: list[NDArray[np.float32]] = []
    logits_batches: list[NDArray[np.float32]] = []
    labels_batches: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.long)
            features = model.forward_features(batch_x)
            logits = model(batch_x)
            embeddings_batches.append(
                features.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            logits_batches.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
            labels_batches.append(batch_y.detach().cpu().numpy().astype(np.int64, copy=False))
    embeddings = np.concatenate(embeddings_batches, axis=0)
    logits = np.concatenate(logits_batches, axis=0)
    true_labels = np.concatenate(labels_batches, axis=0)
    topk = stable_topk_per_row(logits.astype(np.float64, copy=False))
    predicted = np.asarray(topk[:, 0], dtype=np.int64)
    return WindowFeatureBundle(
        class_names=class_names,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        logits=np.asarray(logits, dtype=np.float32),
        true_labels=np.asarray(true_labels, dtype=np.int64),
        predicted_labels=predicted,
        topk_indices=topk,
        splits=tuple(split_name for _ in windows),
        relative_paths=tuple(window.relative_path for window in windows),
        absolute_paths=tuple(window.absolute_path for window in windows),
        window_indices=np.asarray([window.window_index for window in windows], dtype=np.int64),
        start_rows=np.asarray([window.start_row for window in windows], dtype=np.int64),
        stop_rows=np.asarray([window.stop_row for window in windows], dtype=np.int64),
    )


def concatenate_feature_bundles(*bundles: WindowFeatureBundle) -> WindowFeatureBundle:
    if not bundles:
        raise M03Error("at least one feature bundle is required")
    class_names = bundles[0].class_names
    if any(bundle.class_names != class_names for bundle in bundles[1:]):
        raise M03Error("cannot concatenate feature bundles with different class_names")
    return WindowFeatureBundle(
        class_names=class_names,
        embeddings=np.concatenate([bundle.embeddings for bundle in bundles], axis=0),
        logits=np.concatenate([bundle.logits for bundle in bundles], axis=0),
        true_labels=np.concatenate([bundle.true_labels for bundle in bundles], axis=0),
        predicted_labels=np.concatenate([bundle.predicted_labels for bundle in bundles], axis=0),
        topk_indices=np.concatenate([bundle.topk_indices for bundle in bundles], axis=0),
        splits=tuple(value for bundle in bundles for value in bundle.splits),
        relative_paths=tuple(value for bundle in bundles for value in bundle.relative_paths),
        absolute_paths=tuple(value for bundle in bundles for value in bundle.absolute_paths),
        window_indices=np.concatenate([bundle.window_indices for bundle in bundles], axis=0),
        start_rows=np.concatenate([bundle.start_rows for bundle in bundles], axis=0),
        stop_rows=np.concatenate([bundle.stop_rows for bundle in bundles], axis=0),
    )


def resolve_best_checkpoint_path(run_dir: Path, metadata: dict[str, Any]) -> Path:
    best_path = metadata.get("best_checkpoint_path", "")
    if best_path:
        candidate = Path(str(best_path))
        if candidate.is_file():
            return candidate
    for candidate in (run_dir / "checkpoint_best.pt", run_dir / "checkpoint_final.pt"):
        if candidate.is_file():
            return candidate
    raise M03Error(f"unable to locate best checkpoint for {run_dir}")


def load_locked_primary_aggregator(run_dir: Path) -> str:
    metadata = load_json_file(run_dir / "run_metadata.json")
    value = str(metadata.get("locked_primary_aggregator", ""))
    if not value:
        raise M03Error(f"run is missing locked_primary_aggregator: {run_dir}")
    return value


def load_checkpoint_weights(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise M03Error(f"checkpoint missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(state_dict)


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise M03Error(f"unable to read json file: {path}") from exc
    if not isinstance(payload, dict):
        raise M03Error(f"json file must contain a mapping: {path}")
    return payload


def normalize_ensemble_method_name(method: str) -> str:
    if method == "vote":
        return MAJORITY_VOTE_AGGREGATOR
    return method


def stable_topk_per_row(scores: NDArray[np.float64], k: int | None = None) -> NDArray[np.int64]:
    if scores.ndim != 2:
        raise M03Error(f"scores must be 2d, found shape {scores.shape}")
    topk = min(k or 5, scores.shape[1])
    return np.stack(
        [stable_descending_topk(row, topk) for row in scores],
        axis=0,
    )


def softmax_np(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def stable_descending_topk(scores: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    order = np.argsort(-scores, kind="mergesort")
    return order[:k].astype(np.int64, copy=False)


def build_file_groups(
    bundle: WindowFeatureBundle,
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    split_bundle = bundle.filter_split(split_name)
    grouped = group_indices_by_file(split_bundle.relative_paths)
    groups: list[dict[str, Any]] = []
    for relative_path in sorted(grouped):
        indices = np.asarray(grouped[relative_path], dtype=np.int64)
        label_set = {int(split_bundle.true_labels[index]) for index in indices.tolist()}
        if len(label_set) != 1:
            raise M03Error(f"file labels changed within grouped file {relative_path!r}")
        groups.append(
            {
                "split": split_name,
                "relative_path": relative_path,
                "absolute_path": split_bundle.absolute_paths[int(indices[0])],
                "true_label": int(split_bundle.true_labels[int(indices[0])]),
                "num_windows": int(indices.size),
                "features": np.asarray(split_bundle.embeddings[indices], dtype=np.float32),
                "logits": np.asarray(split_bundle.logits[indices], dtype=np.float32),
            }
        )
    return groups
