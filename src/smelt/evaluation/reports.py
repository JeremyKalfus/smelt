"""artifact-friendly report exports for exact-upstream runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metrics import ClassificationMetrics


class ReportExportError(Exception):
    """raised when evaluation reports cannot be written safely."""


@dataclass(slots=True)
class ReportBundlePaths:
    report_dir: str
    summary_json: str
    confusion_matrix_csv: str
    per_category_accuracy_csv: str

    def to_dict(self) -> dict[str, str]:
        return {
            "report_dir": self.report_dir,
            "summary_json": self.summary_json,
            "confusion_matrix_csv": self.confusion_matrix_csv,
            "per_category_accuracy_csv": self.per_category_accuracy_csv,
        }


def export_classification_report(
    *,
    output_root: Path,
    run_name: str,
    metrics: ClassificationMetrics,
    methods_summary: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> ReportBundlePaths:
    report_dir = output_root / run_name
    if report_dir.exists() and not overwrite:
        raise ReportExportError(f"report directory already exists: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=overwrite)

    summary_json = report_dir / "summary_metrics.json"
    confusion_csv = report_dir / "confusion_matrix.csv"
    per_category_csv = report_dir / "per_category_accuracy.csv"

    write_summary_json(summary_json, metrics, methods_summary)
    write_confusion_matrix_csv(confusion_csv, metrics)
    write_per_category_accuracy_csv(per_category_csv, metrics)

    return ReportBundlePaths(
        report_dir=str(report_dir.resolve()),
        summary_json=str(summary_json.resolve()),
        confusion_matrix_csv=str(confusion_csv.resolve()),
        per_category_accuracy_csv=str(per_category_csv.resolve()),
    )


def write_summary_json(
    output_path: Path,
    metrics: ClassificationMetrics,
    methods_summary: dict[str, Any] | None,
) -> None:
    payload: dict[str, Any] = metrics.summary_dict()
    payload["per_category"] = [row.to_dict() for row in metrics.per_category]
    if methods_summary is not None:
        payload["methods"] = methods_summary
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_confusion_matrix_csv(output_path: Path, metrics: ClassificationMetrics) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", *metrics.class_names])
        for class_name, row in zip(metrics.class_names, metrics.confusion_matrix, strict=True):
            writer.writerow([class_name, *[int(value) for value in row]])


def write_per_category_accuracy_csv(output_path: Path, metrics: ClassificationMetrics) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "n", "acc@1", "acc@5"])
        for row in metrics.per_category:
            writer.writerow([row.category, row.sample_count, row.acc_at_1, row.acc_at_5])
