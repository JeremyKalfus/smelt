"""evaluation utilities for exact-upstream reporting."""

from .metrics import (
    EXACT_UPSTREAM_CATEGORY_ORDER,
    CategoryAccuracyRow,
    ClassificationInputs,
    ClassificationMetrics,
    EvaluationMetricError,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_macro_precision_recall_f1,
    compute_per_category_accuracy,
    compute_topk_accuracy,
    load_category_mapping,
    prepare_classification_inputs,
    validate_category_mapping,
)
from .reports import (
    ReportBundlePaths,
    ReportExportError,
    export_classification_report,
)

__all__ = [
    "CategoryAccuracyRow",
    "ClassificationInputs",
    "ClassificationMetrics",
    "EXACT_UPSTREAM_CATEGORY_ORDER",
    "EvaluationMetricError",
    "ReportBundlePaths",
    "ReportExportError",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_macro_precision_recall_f1",
    "compute_per_category_accuracy",
    "compute_topk_accuracy",
    "export_classification_report",
    "load_category_mapping",
    "prepare_classification_inputs",
    "validate_category_mapping",
]
