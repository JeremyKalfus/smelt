"""exact-upstream metric helpers for base classification runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

EXACT_UPSTREAM_CATEGORY_ORDER = (
    "nuts",
    "spices",
    "herbs",
    "fruits",
    "vegetables",
)


class EvaluationMetricError(Exception):
    """raised when evaluation metrics cannot be computed safely."""


@dataclass(slots=True)
class ClassificationInputs:
    class_names: tuple[str, ...]
    true_labels: NDArray[np.int64]
    predicted_labels: NDArray[np.int64]
    topk_indices: NDArray[np.int64]

    @property
    def class_count(self) -> int:
        return len(self.class_names)

    @property
    def sample_count(self) -> int:
        return int(self.true_labels.shape[0])


@dataclass(slots=True)
class CategoryAccuracyRow:
    category: str
    sample_count: int
    acc_at_1: float
    acc_at_5: float

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "n": self.sample_count,
            "acc@1": self.acc_at_1,
            "acc@5": self.acc_at_5,
        }


@dataclass(slots=True)
class ClassificationMetrics:
    acc_at_1: float
    acc_at_5: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: NDArray[np.int64]
    class_names: tuple[str, ...]
    per_category: tuple[CategoryAccuracyRow, ...]

    def summary_dict(self) -> dict[str, object]:
        return {
            "acc@1": self.acc_at_1,
            "acc@5": self.acc_at_5,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "class_names": list(self.class_names),
        }


def load_category_mapping(mapping_path: Path) -> dict[str, str]:
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvaluationMetricError(f"unable to read category mapping: {mapping_path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationMetricError(f"invalid category mapping json: {mapping_path}") from exc
    if not isinstance(payload, dict):
        raise EvaluationMetricError("category mapping json must be an object of class->category")

    mapping: dict[str, str] = {}
    for class_name, category_name in payload.items():
        if not isinstance(class_name, str) or not isinstance(category_name, str):
            raise EvaluationMetricError("category mapping entries must be string->string")
        normalized_category = category_name.strip().lower()
        if normalized_category not in EXACT_UPSTREAM_CATEGORY_ORDER:
            raise EvaluationMetricError(
                f"unsupported category {category_name!r}; expected one of "
                f"{list(EXACT_UPSTREAM_CATEGORY_ORDER)}"
            )
        if class_name in mapping:
            raise EvaluationMetricError(f"duplicate class in category mapping: {class_name}")
        mapping[class_name] = normalized_category
    return mapping


def prepare_classification_inputs(
    *,
    class_names: Sequence[str],
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    topk_indices: Sequence[Sequence[int]] | NDArray[np.int64],
) -> ClassificationInputs:
    class_names_tuple = tuple(class_names)
    if not class_names_tuple:
        raise EvaluationMetricError("class_names must be non-empty")
    if len(set(class_names_tuple)) != len(class_names_tuple):
        raise EvaluationMetricError(f"class_names contain duplicates: {list(class_names_tuple)}")

    class_count = len(class_names_tuple)
    true_array = np.asarray(true_labels, dtype=np.int64)
    predicted_array = np.asarray(predicted_labels, dtype=np.int64)
    topk_array = np.asarray(topk_indices, dtype=np.int64)

    if true_array.ndim != 1:
        raise EvaluationMetricError(f"true_labels must be 1d, found shape {true_array.shape}")
    if predicted_array.ndim != 1:
        raise EvaluationMetricError(
            f"predicted_labels must be 1d, found shape {predicted_array.shape}"
        )
    if topk_array.ndim != 2:
        raise EvaluationMetricError(f"topk_indices must be 2d, found shape {topk_array.shape}")
    if true_array.shape[0] != predicted_array.shape[0]:
        raise EvaluationMetricError(
            "true_labels and predicted_labels must have the same length: "
            f"{true_array.shape[0]} vs {predicted_array.shape[0]}"
        )
    if true_array.shape[0] != topk_array.shape[0]:
        raise EvaluationMetricError(
            "topk_indices must align with label count: "
            f"{topk_array.shape[0]} vs {true_array.shape[0]}"
        )
    if topk_array.shape[1] == 0:
        raise EvaluationMetricError("topk_indices must contain at least one rank")

    validate_label_bounds(true_array, class_count, name="true_labels")
    validate_label_bounds(predicted_array, class_count, name="predicted_labels")
    validate_label_bounds(topk_array.reshape(-1), class_count, name="topk_indices")

    if not np.array_equal(predicted_array, topk_array[:, 0]):
        raise EvaluationMetricError("predicted_labels must match topk_indices[:, 0]")

    return ClassificationInputs(
        class_names=class_names_tuple,
        true_labels=true_array,
        predicted_labels=predicted_array,
        topk_indices=topk_array,
    )


def compute_classification_metrics(
    *,
    class_names: Sequence[str],
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    topk_indices: Sequence[Sequence[int]] | NDArray[np.int64],
    category_mapping: Mapping[str, str],
) -> ClassificationMetrics:
    inputs = prepare_classification_inputs(
        class_names=class_names,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        topk_indices=topk_indices,
    )
    validated_mapping = validate_category_mapping(inputs.class_names, category_mapping)

    confusion = compute_confusion_matrix(
        true_labels=inputs.true_labels,
        predicted_labels=inputs.predicted_labels,
        class_count=inputs.class_count,
    )
    acc_at_1 = compute_topk_accuracy(inputs.true_labels, inputs.topk_indices, k=1)
    acc_at_5 = compute_topk_accuracy(inputs.true_labels, inputs.topk_indices, k=5)
    precision_macro, recall_macro, f1_macro = compute_macro_precision_recall_f1(confusion)
    per_category = compute_per_category_accuracy(inputs, validated_mapping)

    return ClassificationMetrics(
        acc_at_1=acc_at_1,
        acc_at_5=acc_at_5,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        confusion_matrix=confusion,
        class_names=inputs.class_names,
        per_category=per_category,
    )


def validate_category_mapping(
    class_names: Sequence[str],
    category_mapping: Mapping[str, str],
) -> dict[str, str]:
    class_names_tuple = tuple(class_names)
    missing_classes = [
        class_name for class_name in class_names_tuple if class_name not in category_mapping
    ]
    extra_classes = sorted(set(category_mapping) - set(class_names_tuple))
    if missing_classes:
        raise EvaluationMetricError(
            f"category mapping is missing classes: {sorted(missing_classes)}"
        )
    if extra_classes:
        raise EvaluationMetricError(
            f"category mapping contains classes outside this run: {extra_classes}"
        )

    normalized_mapping: dict[str, str] = {}
    for class_name in class_names_tuple:
        category_name = category_mapping[class_name].strip().lower()
        if category_name not in EXACT_UPSTREAM_CATEGORY_ORDER:
            raise EvaluationMetricError(
                f"class {class_name} maps to unsupported category {category_name!r}"
            )
        normalized_mapping[class_name] = category_name
    return normalized_mapping


def compute_topk_accuracy(
    true_labels: NDArray[np.int64],
    topk_indices: NDArray[np.int64],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise EvaluationMetricError(f"k must be positive, found {k}")
    effective_k = min(k, topk_indices.shape[1])
    hits = (topk_indices[:, :effective_k] == true_labels[:, None]).any(axis=1)
    return float(hits.mean() * 100.0) if hits.size else 0.0


def compute_confusion_matrix(
    *,
    true_labels: NDArray[np.int64],
    predicted_labels: NDArray[np.int64],
    class_count: int,
) -> NDArray[np.int64]:
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    for true_label, predicted_label in zip(true_labels, predicted_labels, strict=True):
        confusion[int(true_label), int(predicted_label)] += 1
    return confusion


def compute_macro_precision_recall_f1(
    confusion_matrix: NDArray[np.int64],
) -> tuple[float, float, float]:
    if confusion_matrix.ndim != 2 or confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise EvaluationMetricError(
            f"confusion_matrix must be square, found shape {confusion_matrix.shape}"
        )
    true_positive = np.diag(confusion_matrix).astype(np.float64)
    predicted_total = confusion_matrix.sum(axis=0).astype(np.float64)
    actual_total = confusion_matrix.sum(axis=1).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted_total,
        out=np.zeros_like(true_positive),
        where=predicted_total > 0,
    )
    recall = np.divide(
        true_positive,
        actual_total,
        out=np.zeros_like(true_positive),
        where=actual_total > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return (
        float(precision.mean() * 100.0),
        float(recall.mean() * 100.0),
        float(f1.mean() * 100.0),
    )


def compute_per_category_accuracy(
    inputs: ClassificationInputs,
    category_mapping: Mapping[str, str],
) -> tuple[CategoryAccuracyRow, ...]:
    label_to_category = {
        class_index: category_mapping[class_name]
        for class_index, class_name in enumerate(inputs.class_names)
    }
    hits_at_1 = (inputs.predicted_labels == inputs.true_labels).astype(np.float64)
    hits_at_5 = (
        (
            inputs.topk_indices[:, : min(5, inputs.topk_indices.shape[1])]
            == inputs.true_labels[:, None]
        )
        .any(axis=1)
        .astype(np.float64)
    )

    rows: list[CategoryAccuracyRow] = []
    for category_name in EXACT_UPSTREAM_CATEGORY_ORDER:
        mask = np.asarray(
            [label_to_category[int(label)] == category_name for label in inputs.true_labels],
            dtype=bool,
        )
        sample_count = int(mask.sum())
        acc_at_1 = float(hits_at_1[mask].mean() * 100.0) if sample_count else 0.0
        acc_at_5 = float(hits_at_5[mask].mean() * 100.0) if sample_count else 0.0
        rows.append(
            CategoryAccuracyRow(
                category=category_name,
                sample_count=sample_count,
                acc_at_1=acc_at_1,
                acc_at_5=acc_at_5,
            )
        )
    return tuple(rows)


def validate_label_bounds(labels: NDArray[np.int64], class_count: int, *, name: str) -> None:
    if labels.size == 0:
        return
    if int(labels.min()) < 0:
        raise EvaluationMetricError(f"{name} contains negative class ids")
    if int(labels.max()) >= class_count:
        raise EvaluationMetricError(f"{name} contains class ids outside [0, {class_count - 1}]")
