"""file-aware window prediction bundles and file-level aggregation helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from smelt.preprocessing.windows import SensorWindow

from .metrics import ClassificationMetrics, compute_classification_metrics
from .reports import export_classification_report

MEAN_LOGITS_AGGREGATOR = "mean_logits"
MEAN_PROBABILITIES_AGGREGATOR = "mean_probabilities"
MAJORITY_VOTE_AGGREGATOR = "majority_vote"
MAJORITY_VOTE_LEGACY_ALIAS = "majority_vote"
FILE_LEVEL_AGGREGATORS = (
    MEAN_LOGITS_AGGREGATOR,
    MEAN_PROBABILITIES_AGGREGATOR,
    MAJORITY_VOTE_AGGREGATOR,
)
LOCKED_AGGREGATOR_TIEBREAK_ORDER = (
    MEAN_PROBABILITIES_AGGREGATOR,
    MEAN_LOGITS_AGGREGATOR,
    MAJORITY_VOTE_AGGREGATOR,
)


class FileLevelAggregationError(Exception):
    """raised when file-level aggregation cannot be computed safely."""


@dataclass(slots=True)
class WindowPredictionBundle:
    class_names: tuple[str, ...]
    true_labels: NDArray[np.int64]
    predicted_labels: NDArray[np.int64]
    topk_indices: NDArray[np.int64]
    logits: NDArray[np.float64]
    splits: tuple[str, ...]
    relative_paths: tuple[str, ...]
    absolute_paths: tuple[str, ...]
    window_indices: NDArray[np.int64]
    start_rows: NDArray[np.int64]
    stop_rows: NDArray[np.int64]

    def __post_init__(self) -> None:
        sample_count = self.true_labels.shape[0]
        if sample_count == 0:
            raise FileLevelAggregationError("window prediction bundle must be non-empty")
        if self.predicted_labels.shape != (sample_count,):
            raise FileLevelAggregationError("predicted_labels shape does not match true_labels")
        if self.topk_indices.ndim != 2 or self.topk_indices.shape[0] != sample_count:
            raise FileLevelAggregationError("topk_indices shape does not match sample count")
        if self.logits.ndim != 2 or self.logits.shape[0] != sample_count:
            raise FileLevelAggregationError("logits shape does not match sample count")
        if self.logits.shape[1] != len(self.class_names):
            raise FileLevelAggregationError("logit class count does not match class_names")
        if len(self.splits) != sample_count:
            raise FileLevelAggregationError("split metadata does not match sample count")
        if len(self.relative_paths) != sample_count:
            raise FileLevelAggregationError("relative path metadata does not match sample count")
        if len(self.absolute_paths) != sample_count:
            raise FileLevelAggregationError("absolute path metadata does not match sample count")
        if self.window_indices.shape != (sample_count,):
            raise FileLevelAggregationError("window_indices shape does not match sample count")
        if self.start_rows.shape != (sample_count,):
            raise FileLevelAggregationError("start_rows shape does not match sample count")
        if self.stop_rows.shape != (sample_count,):
            raise FileLevelAggregationError("stop_rows shape does not match sample count")

    @property
    def sample_count(self) -> int:
        return int(self.true_labels.shape[0])


@dataclass(slots=True)
class FilePredictionRow:
    split: str
    relative_path: str
    absolute_path: str
    true_class: str
    predicted_class: str
    num_windows: int
    top5_classes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "relative_path": self.relative_path,
            "absolute_path": self.absolute_path,
            "true_class": self.true_class,
            "predicted_class": self.predicted_class,
            "num_windows": self.num_windows,
            "top5_classes": list(self.top5_classes),
        }


@dataclass(slots=True)
class FileLevelAggregationResult:
    aggregator: str
    metrics: ClassificationMetrics
    rows: tuple[FilePredictionRow, ...]


@dataclass(slots=True)
class FileScoreBundle:
    aggregator: str
    class_names: tuple[str, ...]
    scores: NDArray[np.float64]
    true_labels: NDArray[np.int64]
    predicted_labels: NDArray[np.int64]
    topk_indices: NDArray[np.int64]
    split_names: tuple[str, ...]
    relative_paths: tuple[str, ...]
    absolute_paths: tuple[str, ...]
    num_windows: NDArray[np.int64]

    def __post_init__(self) -> None:
        sample_count = int(self.true_labels.shape[0])
        if sample_count == 0:
            raise FileLevelAggregationError("file score bundle must be non-empty")
        if self.scores.shape != (sample_count, len(self.class_names)):
            raise FileLevelAggregationError("scores shape does not match sample and class count")
        if self.predicted_labels.shape != (sample_count,):
            raise FileLevelAggregationError("predicted_labels shape does not match sample count")
        if self.topk_indices.ndim != 2 or self.topk_indices.shape[0] != sample_count:
            raise FileLevelAggregationError("topk_indices shape does not match sample count")
        if len(self.split_names) != sample_count:
            raise FileLevelAggregationError("split_names length does not match sample count")
        if len(self.relative_paths) != sample_count:
            raise FileLevelAggregationError("relative_paths length does not match sample count")
        if len(self.absolute_paths) != sample_count:
            raise FileLevelAggregationError("absolute_paths length does not match sample count")
        if self.num_windows.shape != (sample_count,):
            raise FileLevelAggregationError("num_windows shape does not match sample count")


@dataclass(slots=True)
class FileLevelReportPaths:
    report_dir: str
    summary_json: str
    confusion_matrix_csv: str
    per_category_accuracy_csv: str
    per_file_predictions_csv: str

    def to_dict(self) -> dict[str, str]:
        return {
            "report_dir": self.report_dir,
            "summary_json": self.summary_json,
            "confusion_matrix_csv": self.confusion_matrix_csv,
            "per_category_accuracy_csv": self.per_category_accuracy_csv,
            "per_file_predictions_csv": self.per_file_predictions_csv,
        }


PredictionBundle = WindowPredictionBundle
AggregatedPredictionBundle = FileLevelAggregationResult
FileLevelEvaluationError = FileLevelAggregationError
FileLevelArtifactPaths = FileLevelReportPaths


@dataclass(slots=True)
class AggregatorSelectionCandidate:
    aggregator: str
    acc_at_1: float
    f1_macro: float


def normalize_aggregator_candidates(aggregators: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(normalize_aggregator_name(value) for value in aggregators)
    if not normalized:
        raise FileLevelAggregationError("at least one file-level aggregator is required")
    invalid = sorted(set(normalized) - set(FILE_LEVEL_AGGREGATORS))
    if invalid:
        raise FileLevelAggregationError(
            f"unsupported file-level aggregators {invalid!r}; expected {FILE_LEVEL_AGGREGATORS}"
        )
    return tuple(dict.fromkeys(normalized))


def select_validation_locked_aggregator(
    candidates: tuple[AggregatorSelectionCandidate, ...] | list[AggregatorSelectionCandidate],
) -> str:
    if not candidates:
        raise FileLevelAggregationError("at least one aggregator candidate is required")
    tie_order = {name: index for index, name in enumerate(LOCKED_AGGREGATOR_TIEBREAK_ORDER)}
    return min(
        candidates,
        key=lambda candidate: (
            -candidate.acc_at_1,
            -candidate.f1_macro,
            tie_order.get(candidate.aggregator, len(LOCKED_AGGREGATOR_TIEBREAK_ORDER)),
        ),
    ).aggregator


def build_window_prediction_bundle(
    *,
    class_names: tuple[str, ...],
    true_labels: NDArray[np.int64],
    predicted_labels: NDArray[np.int64],
    topk_indices: NDArray[np.int64],
    logits: NDArray[np.float64],
    windows: tuple[SensorWindow, ...],
) -> WindowPredictionBundle:
    if len(windows) != int(true_labels.shape[0]):
        raise FileLevelAggregationError(
            "window metadata count does not match prediction count: "
            f"{len(windows)} vs {true_labels.shape[0]}"
        )
    return WindowPredictionBundle(
        class_names=class_names,
        true_labels=np.asarray(true_labels, dtype=np.int64),
        predicted_labels=np.asarray(predicted_labels, dtype=np.int64),
        topk_indices=np.asarray(topk_indices, dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float64),
        splits=tuple(window.split for window in windows),
        relative_paths=tuple(window.relative_path for window in windows),
        absolute_paths=tuple(window.absolute_path for window in windows),
        window_indices=np.asarray([window.window_index for window in windows], dtype=np.int64),
        start_rows=np.asarray([window.start_row for window in windows], dtype=np.int64),
        stop_rows=np.asarray([window.stop_row for window in windows], dtype=np.int64),
    )


def write_window_prediction_bundle(
    output_path: Path,
    bundle: WindowPredictionBundle,
) -> None:
    np.savez_compressed(output_path, **build_prediction_bundle_payload_from_bundle(bundle))


def load_window_prediction_bundle(input_path: Path) -> WindowPredictionBundle:
    try:
        payload = np.load(input_path, allow_pickle=False)
    except OSError as exc:
        raise FileLevelAggregationError(f"unable to read prediction bundle: {input_path}") from exc
    required_fields = {
        "class_names",
        "true_labels",
        "predicted_labels",
        "topk_indices",
        "logits",
        "splits",
        "relative_paths",
        "absolute_paths",
        "window_indices",
        "start_rows",
        "stop_rows",
    }
    missing_fields = sorted(required_fields - set(payload.files))
    if missing_fields:
        raise FileLevelAggregationError(
            "prediction bundle is missing file-aware fields: "
            f"{missing_fields}; re-export from checkpoint is required"
        )
    return WindowPredictionBundle(
        class_names=tuple(str(value) for value in payload["class_names"].tolist()),
        true_labels=np.asarray(payload["true_labels"], dtype=np.int64),
        predicted_labels=np.asarray(payload["predicted_labels"], dtype=np.int64),
        topk_indices=np.asarray(payload["topk_indices"], dtype=np.int64),
        logits=np.asarray(payload["logits"], dtype=np.float64),
        splits=tuple(str(value) for value in payload["splits"].tolist()),
        relative_paths=tuple(str(value) for value in payload["relative_paths"].tolist()),
        absolute_paths=tuple(str(value) for value in payload["absolute_paths"].tolist()),
        window_indices=np.asarray(payload["window_indices"], dtype=np.int64),
        start_rows=np.asarray(payload["start_rows"], dtype=np.int64),
        stop_rows=np.asarray(payload["stop_rows"], dtype=np.int64),
    )


def build_prediction_bundle_payload(
    *,
    class_names: tuple[str, ...],
    true_labels: NDArray[np.int64],
    predicted_labels: NDArray[np.int64],
    topk_indices: NDArray[np.int64],
    logits: NDArray[np.float64],
    windows: tuple[SensorWindow, ...] | None = None,
) -> dict[str, NDArray[np.generic[Any]] | NDArray[np.int64] | NDArray[np.float64]]:
    payload: dict[str, NDArray[np.generic[Any]] | NDArray[np.int64] | NDArray[np.float64]] = {
        "class_names": np.asarray(class_names),
        "true_labels": np.asarray(true_labels, dtype=np.int64),
        "predicted_labels": np.asarray(predicted_labels, dtype=np.int64),
        "topk_indices": np.asarray(topk_indices, dtype=np.int64),
        "logits": np.asarray(logits, dtype=np.float64),
    }
    if windows is None:
        return payload
    bundle = build_window_prediction_bundle(
        class_names=class_names,
        true_labels=np.asarray(true_labels, dtype=np.int64),
        predicted_labels=np.asarray(predicted_labels, dtype=np.int64),
        topk_indices=np.asarray(topk_indices, dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float64),
        windows=windows,
    )
    return build_prediction_bundle_payload_from_bundle(bundle)


def build_prediction_bundle_payload_from_bundle(
    bundle: WindowPredictionBundle,
) -> dict[str, NDArray[np.generic[Any]] | NDArray[np.int64] | NDArray[np.float64]]:
    return {
        "class_names": np.asarray(bundle.class_names),
        "true_labels": bundle.true_labels,
        "predicted_labels": bundle.predicted_labels,
        "topk_indices": bundle.topk_indices,
        "logits": bundle.logits,
        "splits": np.asarray(bundle.splits),
        "relative_paths": np.asarray(bundle.relative_paths),
        "absolute_paths": np.asarray(bundle.absolute_paths),
        "window_indices": bundle.window_indices,
        "start_rows": bundle.start_rows,
        "stop_rows": bundle.stop_rows,
    }


def aggregate_file_level_metrics(
    *,
    bundle: WindowPredictionBundle,
    category_mapping: dict[str, str],
    aggregator: str,
) -> FileLevelAggregationResult:
    aggregator = normalize_aggregator_name(aggregator)
    if aggregator not in FILE_LEVEL_AGGREGATORS:
        raise FileLevelAggregationError(
            f"unsupported file-level aggregator {aggregator!r}; expected {FILE_LEVEL_AGGREGATORS}"
        )
    grouped_indices = group_indices_by_file(bundle.relative_paths)
    aggregated_topk: list[NDArray[np.int64]] = []
    aggregated_true: list[int] = []
    aggregated_predicted: list[int] = []
    rows: list[FilePredictionRow] = []

    for relative_path in sorted(grouped_indices):
        indices = grouped_indices[relative_path]
        split_name = require_unique_string(bundle.splits, indices, "split", relative_path)
        absolute_path = require_unique_string(
            bundle.absolute_paths,
            indices,
            "absolute_path",
            relative_path,
        )
        true_label = require_unique_int(bundle.true_labels, indices, "true_label", relative_path)
        aggregated_scores, topk = aggregate_group_scores(bundle, indices, aggregator)
        predicted_label = int(topk[0])
        aggregated_topk.append(topk)
        aggregated_true.append(true_label)
        aggregated_predicted.append(predicted_label)
        rows.append(
            FilePredictionRow(
                split=split_name,
                relative_path=relative_path,
                absolute_path=absolute_path,
                true_class=bundle.class_names[true_label],
                predicted_class=bundle.class_names[predicted_label],
                num_windows=len(indices),
                top5_classes=tuple(bundle.class_names[index] for index in topk[:5]),
            )
        )

    metrics = compute_classification_metrics(
        class_names=bundle.class_names,
        true_labels=np.asarray(aggregated_true, dtype=np.int64),
        predicted_labels=np.asarray(aggregated_predicted, dtype=np.int64),
        topk_indices=np.stack(aggregated_topk, axis=0),
        category_mapping=category_mapping,
    )
    return FileLevelAggregationResult(
        aggregator=aggregator,
        metrics=metrics,
        rows=tuple(rows),
    )


def build_file_score_bundle(
    *,
    bundle: WindowPredictionBundle,
    aggregator: str,
) -> FileScoreBundle:
    aggregator = normalize_aggregator_name(aggregator)
    grouped_indices = group_indices_by_file(bundle.relative_paths)
    score_rows: list[NDArray[np.float64]] = []
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    topk_rows: list[NDArray[np.int64]] = []
    split_names: list[str] = []
    relative_paths: list[str] = []
    absolute_paths: list[str] = []
    num_windows: list[int] = []

    for relative_path in sorted(grouped_indices):
        indices = grouped_indices[relative_path]
        split_names.append(require_unique_string(bundle.splits, indices, "split", relative_path))
        absolute_paths.append(
            require_unique_string(bundle.absolute_paths, indices, "absolute_path", relative_path)
        )
        true_label = require_unique_int(bundle.true_labels, indices, "true_label", relative_path)
        scores, topk = aggregate_group_scores(bundle, indices, aggregator)
        score_rows.append(scores.astype(np.float64, copy=False))
        true_labels.append(true_label)
        predicted_labels.append(int(topk[0]))
        topk_rows.append(topk.astype(np.int64, copy=False))
        relative_paths.append(relative_path)
        num_windows.append(len(indices))

    return FileScoreBundle(
        aggregator=aggregator,
        class_names=bundle.class_names,
        scores=np.stack(score_rows, axis=0),
        true_labels=np.asarray(true_labels, dtype=np.int64),
        predicted_labels=np.asarray(predicted_labels, dtype=np.int64),
        topk_indices=np.stack(topk_rows, axis=0),
        split_names=tuple(split_names),
        relative_paths=tuple(relative_paths),
        absolute_paths=tuple(absolute_paths),
        num_windows=np.asarray(num_windows, dtype=np.int64),
    )


def write_file_score_bundle(
    output_path: Path,
    bundle: FileScoreBundle,
) -> None:
    np.savez_compressed(
        output_path,
        aggregator=np.asarray(bundle.aggregator),
        class_names=np.asarray(bundle.class_names),
        scores=np.asarray(bundle.scores, dtype=np.float64),
        true_labels=np.asarray(bundle.true_labels, dtype=np.int64),
        predicted_labels=np.asarray(bundle.predicted_labels, dtype=np.int64),
        topk_indices=np.asarray(bundle.topk_indices, dtype=np.int64),
        split_names=np.asarray(bundle.split_names),
        relative_paths=np.asarray(bundle.relative_paths),
        absolute_paths=np.asarray(bundle.absolute_paths),
        num_windows=np.asarray(bundle.num_windows, dtype=np.int64),
    )


def load_file_score_bundle(
    input_path: Path,
) -> FileScoreBundle:
    try:
        payload = np.load(input_path, allow_pickle=False)
    except OSError as exc:
        raise FileLevelAggregationError(f"unable to read file score bundle: {input_path}") from exc
    required_fields = {
        "aggregator",
        "class_names",
        "scores",
        "true_labels",
        "predicted_labels",
        "topk_indices",
        "split_names",
        "relative_paths",
        "absolute_paths",
        "num_windows",
    }
    missing_fields = sorted(required_fields - set(payload.files))
    if missing_fields:
        raise FileLevelAggregationError(
            f"file score bundle is missing required fields: {missing_fields}"
        )
    return FileScoreBundle(
        aggregator=str(payload["aggregator"].tolist()),
        class_names=tuple(str(value) for value in payload["class_names"].tolist()),
        scores=np.asarray(payload["scores"], dtype=np.float64),
        true_labels=np.asarray(payload["true_labels"], dtype=np.int64),
        predicted_labels=np.asarray(payload["predicted_labels"], dtype=np.int64),
        topk_indices=np.asarray(payload["topk_indices"], dtype=np.int64),
        split_names=tuple(str(value) for value in payload["split_names"].tolist()),
        relative_paths=tuple(str(value) for value in payload["relative_paths"].tolist()),
        absolute_paths=tuple(str(value) for value in payload["absolute_paths"].tolist()),
        num_windows=np.asarray(payload["num_windows"], dtype=np.int64),
    )


def evaluate_file_level_aggregation(
    bundle: WindowPredictionBundle,
    *,
    aggregator: str,
    category_mapping: dict[str, str],
) -> tuple[ClassificationMetrics, FileLevelAggregationResult]:
    result = aggregate_file_level_metrics(
        bundle=bundle,
        category_mapping=category_mapping,
        aggregator=aggregator,
    )
    return result.metrics, result


def build_file_level_result_from_predictions(
    *,
    aggregator: str,
    class_names: tuple[str, ...],
    true_labels: NDArray[np.int64],
    predicted_labels: NDArray[np.int64],
    topk_indices: NDArray[np.int64],
    split_names: tuple[str, ...],
    relative_paths: tuple[str, ...],
    absolute_paths: tuple[str, ...],
    num_windows: NDArray[np.int64],
    category_mapping: dict[str, str],
) -> FileLevelAggregationResult:
    sample_count = int(true_labels.shape[0])
    if sample_count == 0:
        raise FileLevelAggregationError("file-level predictions must be non-empty")
    if predicted_labels.shape != (sample_count,):
        raise FileLevelAggregationError("predicted_labels shape does not match true_labels")
    if topk_indices.ndim != 2 or topk_indices.shape[0] != sample_count:
        raise FileLevelAggregationError("topk_indices shape does not match sample count")
    if len(split_names) != sample_count:
        raise FileLevelAggregationError("split_names length does not match sample count")
    if len(relative_paths) != sample_count:
        raise FileLevelAggregationError("relative_paths length does not match sample count")
    if len(absolute_paths) != sample_count:
        raise FileLevelAggregationError("absolute_paths length does not match sample count")
    if num_windows.shape != (sample_count,):
        raise FileLevelAggregationError("num_windows shape does not match sample count")

    rows = tuple(
        FilePredictionRow(
            split=split_names[index],
            relative_path=relative_paths[index],
            absolute_path=absolute_paths[index],
            true_class=class_names[int(true_labels[index])],
            predicted_class=class_names[int(predicted_labels[index])],
            num_windows=int(num_windows[index]),
            top5_classes=tuple(
                class_names[int(class_index)] for class_index in topk_indices[index][:5]
            ),
        )
        for index in range(sample_count)
    )
    metrics = compute_classification_metrics(
        class_names=class_names,
        true_labels=np.asarray(true_labels, dtype=np.int64),
        predicted_labels=np.asarray(predicted_labels, dtype=np.int64),
        topk_indices=np.asarray(topk_indices, dtype=np.int64),
        category_mapping=category_mapping,
    )
    return FileLevelAggregationResult(
        aggregator=aggregator,
        metrics=metrics,
        rows=rows,
    )


def export_file_level_report(
    *,
    output_root: Path,
    run_name: str,
    result: FileLevelAggregationResult,
    methods_summary: dict[str, Any] | None = None,
) -> FileLevelReportPaths:
    bundle_paths = export_classification_report(
        output_root=output_root,
        run_name=run_name,
        metrics=result.metrics,
        methods_summary=methods_summary,
    )
    per_file_predictions_csv = Path(bundle_paths.report_dir) / "per_file_predictions.csv"
    write_per_file_predictions_csv(per_file_predictions_csv, result.rows)
    return FileLevelReportPaths(
        report_dir=bundle_paths.report_dir,
        summary_json=bundle_paths.summary_json,
        confusion_matrix_csv=bundle_paths.confusion_matrix_csv,
        per_category_accuracy_csv=bundle_paths.per_category_accuracy_csv,
        per_file_predictions_csv=str(per_file_predictions_csv.resolve()),
    )


def load_prediction_bundle(input_path: Path) -> WindowPredictionBundle:
    return load_window_prediction_bundle(input_path)


def prediction_bundle_has_file_metadata(input_path: Path) -> bool:
    try:
        payload = np.load(input_path, allow_pickle=False)
    except OSError:
        return False
    return {
        "splits",
        "relative_paths",
        "absolute_paths",
        "window_indices",
        "start_rows",
        "stop_rows",
    } <= set(payload.files)


def build_file_level_summary_row(
    *,
    run_id: str,
    track: str,
    model_family: str,
    view_mode: str,
    diff_period: str,
    window_size: str,
    stride: str,
    aggregator: str,
    window_metrics: ClassificationMetrics,
    file_metrics: ClassificationMetrics,
    report_paths: FileLevelReportPaths,
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "track": track,
        "model_family": model_family,
        "view_mode": view_mode,
        "g": diff_period,
        "window_size": window_size,
        "stride": stride,
        "aggregator": aggregator,
        "window_acc@1": str(window_metrics.acc_at_1),
        "window_acc@5": str(window_metrics.acc_at_5),
        "window_macro_f1": str(window_metrics.f1_macro),
        "file_acc@1": str(file_metrics.acc_at_1),
        "file_acc@5": str(file_metrics.acc_at_5),
        "file_macro_f1": str(file_metrics.f1_macro),
        "summary_json": report_paths.summary_json,
        "confusion_matrix_csv": report_paths.confusion_matrix_csv,
        "per_category_accuracy_csv": report_paths.per_category_accuracy_csv,
        "per_file_predictions_csv": report_paths.per_file_predictions_csv,
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
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate_group_scores(
    bundle: WindowPredictionBundle,
    indices: tuple[int, ...],
    aggregator: str,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    aggregator = normalize_aggregator_name(aggregator)
    logits = bundle.logits[list(indices)]
    predicted = bundle.predicted_labels[list(indices)]
    if aggregator == MEAN_LOGITS_AGGREGATOR:
        scores = logits.mean(axis=0)
    elif aggregator == MEAN_PROBABILITIES_AGGREGATOR:
        scores = softmax(logits).mean(axis=0)
    else:
        counts = np.bincount(predicted, minlength=len(bundle.class_names)).astype(np.float64)
        scores = counts
    topk = stable_descending_topk(scores, min(5, len(bundle.class_names)))
    return scores, topk


def write_per_file_predictions_csv(
    output_path: Path,
    rows: tuple[FilePredictionRow, ...],
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "relative_path",
                "absolute_path",
                "true_class",
                "predicted_class",
                "num_windows",
                "top5_classes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.split,
                    row.relative_path,
                    row.absolute_path,
                    row.true_class,
                    row.predicted_class,
                    row.num_windows,
                    json.dumps(list(row.top5_classes)),
                ]
            )


def group_indices_by_file(relative_paths: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for index, relative_path in enumerate(relative_paths):
        grouped.setdefault(relative_path, []).append(index)
    return {path: tuple(indices) for path, indices in grouped.items()}


def normalize_aggregator_name(aggregator: str) -> str:
    if aggregator == MAJORITY_VOTE_LEGACY_ALIAS:
        return MAJORITY_VOTE_AGGREGATOR
    return aggregator


def require_unique_string(
    values: tuple[str, ...],
    indices: tuple[int, ...],
    field_name: str,
    relative_path: str,
) -> str:
    unique_values = {values[index] for index in indices}
    if len(unique_values) != 1:
        raise FileLevelAggregationError(
            f"{field_name} changed within file group {relative_path}: {sorted(unique_values)}"
        )
    return next(iter(unique_values))


def require_unique_int(
    values: NDArray[np.int64],
    indices: tuple[int, ...],
    field_name: str,
    relative_path: str,
) -> int:
    unique_values = {int(values[index]) for index in indices}
    if len(unique_values) != 1:
        raise FileLevelAggregationError(
            f"{field_name} changed within file group {relative_path}: {sorted(unique_values)}"
        )
    return next(iter(unique_values))


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def stable_descending_topk(scores: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    if scores.ndim != 1:
        raise FileLevelAggregationError(f"scores must be 1d, found shape {scores.shape}")
    if k <= 0:
        raise FileLevelAggregationError(f"k must be positive, found {k}")
    order = np.argsort(-scores, kind="mergesort")
    return order[:k].astype(np.int64, copy=False)
