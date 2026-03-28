"""anti-cheat verification for exact-upstream classifier runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from smelt.datasets import load_base_sensor_dataset
from smelt.evaluation import (
    ClassificationMetrics,
    compute_classification_metrics,
    export_classification_report,
    load_category_mapping,
)
from smelt.preprocessing import (
    EXACT_UPSTREAM_DROPPED_COLUMNS,
    fit_window_standardizer,
    generate_split_windows,
    preprocess_split_records,
)

from .run import (
    ExactUpstreamRunConfig,
    build_classifier_model,
    build_dataloader,
    build_run_dir,
    collect_evaluation_outputs,
    load_run_config,
    resolve_device,
    write_prediction_bundle,
)

FLOAT_TOLERANCE = 1e-9


class VerificationError(Exception):
    """raised when anti-cheat verification fails."""


@dataclass(slots=True)
class VerificationResult:
    verification_dir: Path
    summary_metrics_path: Path
    confusion_matrix_path: Path
    per_category_accuracy_path: Path
    predictions_path: Path
    metric_comparison_path: Path
    leakage_audit_path: Path
    a1_pass: bool
    a2_pass: bool
    a3_pass: bool
    acc_at_1: float
    acc_at_5: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    test_window_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = verify_exact_upstream_run(args.run_dir)
    print(f"verification_dir: {result.verification_dir}")
    print(f"summary_metrics_path: {result.summary_metrics_path}")
    print(f"confusion_matrix_path: {result.confusion_matrix_path}")
    print(f"per_category_accuracy_path: {result.per_category_accuracy_path}")
    print(f"predictions_path: {result.predictions_path}")
    print(f"metric_comparison_path: {result.metric_comparison_path}")
    print(f"leakage_audit_path: {result.leakage_audit_path}")
    print(f"a1_pass: {result.a1_pass}")
    print(f"a2_pass: {result.a2_pass}")
    print(f"a3_pass: {result.a3_pass}")
    print(f"acc@1: {result.acc_at_1}")
    print(f"acc@5: {result.acc_at_5}")
    print(f"precision_macro: {result.precision_macro}")
    print(f"recall_macro: {result.recall_macro}")
    print(f"f1_macro: {result.f1_macro}")
    print(f"test_window_count: {result.test_window_count}")
    return 0


def verify_exact_upstream_run(run_dir: Path) -> VerificationResult:
    resolved_run_dir = run_dir.expanduser().resolve()
    config = load_run_config(resolved_run_dir / "resolved_config.yaml")
    checkpoint_path = resolved_run_dir / "checkpoint_final.pt"
    if not checkpoint_path.is_file():
        raise VerificationError(f"checkpoint is missing: {checkpoint_path}")

    saved_summary = load_saved_summary_metrics(resolved_run_dir / "summary_metrics.json")
    saved_confusion = load_confusion_matrix_csv(resolved_run_dir / "confusion_matrix.csv")
    saved_per_category = load_per_category_rows(resolved_run_dir / "per_category_accuracy.csv")

    dataset = load_base_sensor_dataset(Path(config.data_root))
    category_mapping = load_category_mapping(Path(config.category_map_path))
    prepared = prepare_verification_inputs(dataset, config)

    device = resolve_device(config.device)
    model = build_classifier_model(
        config=config,
        input_dim=prepared["input_dim"],
        num_classes=prepared["class_count"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    evaluation = collect_evaluation_outputs(
        model=model,
        data_loader=prepared["test_loader"],
        device=device,
        class_names=prepared["class_names"],
        category_mapping=category_mapping,
    )

    verification_name = build_run_dir(
        resolved_run_dir / "verification",
        "anti_cheat_eval_only",
        {"run_dir": str(resolved_run_dir), "checkpoint_path": str(checkpoint_path.resolve())},
    ).name
    report_paths = export_classification_report(
        output_root=resolved_run_dir / "verification",
        run_name=verification_name,
        metrics=evaluation.metrics,
        methods_summary={
            "original_run_dir": str(resolved_run_dir),
            "checkpoint_path": str(checkpoint_path.resolve()),
        },
    )
    verification_dir = Path(report_paths.report_dir)
    predictions_path = verification_dir / "predictions.npz"
    write_prediction_bundle(predictions_path, evaluation)

    compare_saved_metrics(saved_summary, evaluation.metrics)
    compare_saved_confusion(saved_confusion, evaluation.metrics)
    compare_saved_per_category(saved_per_category, evaluation.metrics)
    validate_confusion_total(evaluation.metrics, evaluation.true_labels.shape[0])

    independent_metrics = recompute_metrics_from_saved_predictions(
        predictions_path=predictions_path,
        category_mapping=category_mapping,
    )
    compare_metrics_pair(independent_metrics, evaluation.metrics, context="independent_recompute")

    metric_comparison_path = verification_dir / "metric_comparison.json"
    metric_comparison_path.write_text(
        json.dumps(
            {
                "a1_pass": True,
                "a2_pass": True,
                "saved_summary_match": True,
                "saved_confusion_match": True,
                "saved_per_category_match": True,
                "independent_recompute_match": True,
                "confusion_total": int(evaluation.metrics.confusion_matrix.sum()),
                "test_example_count": int(evaluation.true_labels.shape[0]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    leakage_audit_path = verification_dir / "leakage_audit.json"
    leakage_audit = build_leakage_audit(dataset, config)
    leakage_audit_path.write_text(
        json.dumps(leakage_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return VerificationResult(
        verification_dir=verification_dir,
        summary_metrics_path=Path(report_paths.summary_json),
        confusion_matrix_path=Path(report_paths.confusion_matrix_csv),
        per_category_accuracy_path=Path(report_paths.per_category_accuracy_csv),
        predictions_path=predictions_path,
        metric_comparison_path=metric_comparison_path,
        leakage_audit_path=leakage_audit_path,
        a1_pass=True,
        a2_pass=True,
        a3_pass=bool(leakage_audit["passed"]),
        acc_at_1=evaluation.metrics.acc_at_1,
        acc_at_5=evaluation.metrics.acc_at_5,
        precision_macro=evaluation.metrics.precision_macro,
        recall_macro=evaluation.metrics.recall_macro,
        f1_macro=evaluation.metrics.f1_macro,
        test_window_count=int(evaluation.true_labels.shape[0]),
    )


def prepare_verification_inputs(
    dataset: Any,
    config: ExactUpstreamRunConfig,
) -> dict[str, Any]:
    class_names = tuple(sorted({record.class_name for record in dataset.train_records}))
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
    standardizer = fit_window_standardizer(train_windows)
    from smelt.preprocessing import apply_window_standardizer, stack_window_values

    standardized_test = apply_window_standardizer(test_windows, standardizer)
    test_loader = build_dataloader(
        stack_window_values(standardized_test.windows).astype(np.float32, copy=False),
        np.asarray(
            [class_to_index[window.class_name] for window in standardized_test.windows],
            dtype=np.int64,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return {
        "class_names": class_names,
        "class_count": len(class_names),
        "input_dim": len(standardized_test.column_names),
        "test_loader": test_loader,
    }


def load_saved_summary_metrics(summary_path: Path) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "acc@1": float(payload["acc@1"]),
        "acc@5": float(payload["acc@5"]),
        "precision_macro": float(payload["precision_macro"]),
        "recall_macro": float(payload["recall_macro"]),
        "f1_macro": float(payload["f1_macro"]),
    }


def load_confusion_matrix_csv(confusion_path: Path) -> np.ndarray:
    with confusion_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [tuple(int(value) for value in row[1:]) for row in reader]
    if len(rows) != len(header) - 1:
        raise VerificationError("saved confusion matrix is not square")
    return np.asarray(rows, dtype=np.int64)


def load_per_category_rows(per_category_path: Path) -> tuple[dict[str, Any], ...]:
    with per_category_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return tuple(
            {
                "category": row["category"],
                "n": int(row["n"]),
                "acc@1": float(row["acc@1"]),
                "acc@5": float(row["acc@5"]),
            }
            for row in reader
        )


def compare_saved_metrics(
    saved_summary: dict[str, float],
    metrics: ClassificationMetrics,
) -> None:
    observed = {
        "acc@1": metrics.acc_at_1,
        "acc@5": metrics.acc_at_5,
        "precision_macro": metrics.precision_macro,
        "recall_macro": metrics.recall_macro,
        "f1_macro": metrics.f1_macro,
    }
    for key, saved_value in saved_summary.items():
        observed_value = observed[key]
        if not np.isclose(saved_value, observed_value, atol=FLOAT_TOLERANCE, rtol=0.0):
            raise VerificationError(
                f"saved metric mismatch for {key}: {saved_value} vs {observed_value}"
            )


def compare_saved_confusion(saved_confusion: np.ndarray, metrics: ClassificationMetrics) -> None:
    if not np.array_equal(saved_confusion, metrics.confusion_matrix):
        raise VerificationError("saved confusion matrix does not match eval-only recomputation")


def compare_saved_per_category(
    saved_rows: tuple[dict[str, Any], ...],
    metrics: ClassificationMetrics,
) -> None:
    observed_rows = tuple(row.to_dict() for row in metrics.per_category)
    if saved_rows != observed_rows:
        raise VerificationError(
            "saved per-category accuracy does not match eval-only recomputation"
        )


def validate_confusion_total(metrics: ClassificationMetrics, test_example_count: int) -> None:
    confusion_total = int(metrics.confusion_matrix.sum())
    if confusion_total != test_example_count:
        raise VerificationError(
            f"confusion matrix total {confusion_total} does not match {test_example_count}"
        )


def recompute_metrics_from_saved_predictions(
    *,
    predictions_path: Path,
    category_mapping: dict[str, str],
) -> ClassificationMetrics:
    with np.load(predictions_path, allow_pickle=False) as payload:
        class_names = tuple(str(value) for value in payload["class_names"].tolist())
        true_labels = np.asarray(payload["true_labels"], dtype=np.int64)
        predicted_labels = np.asarray(payload["predicted_labels"], dtype=np.int64)
        topk_indices = np.asarray(payload["topk_indices"], dtype=np.int64)
    return compute_classification_metrics(
        class_names=class_names,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        topk_indices=topk_indices,
        category_mapping=category_mapping,
    )


def compare_metrics_pair(
    left: ClassificationMetrics,
    right: ClassificationMetrics,
    *,
    context: str,
) -> None:
    scalar_pairs = (
        ("acc@1", left.acc_at_1, right.acc_at_1),
        ("acc@5", left.acc_at_5, right.acc_at_5),
        ("precision_macro", left.precision_macro, right.precision_macro),
        ("recall_macro", left.recall_macro, right.recall_macro),
        ("f1_macro", left.f1_macro, right.f1_macro),
    )
    for name, left_value, right_value in scalar_pairs:
        if not np.isclose(left_value, right_value, atol=FLOAT_TOLERANCE, rtol=0.0):
            raise VerificationError(f"{context} mismatch for {name}: {left_value} vs {right_value}")
    if not np.array_equal(left.confusion_matrix, right.confusion_matrix):
        raise VerificationError(f"{context} confusion matrix mismatch")
    if tuple(row.to_dict() for row in left.per_category) != tuple(
        row.to_dict() for row in right.per_category
    ):
        raise VerificationError(f"{context} per-category mismatch")


def build_leakage_audit(
    dataset: Any,
    config: ExactUpstreamRunConfig,
) -> dict[str, Any]:
    train_files = sorted(record.relative_path for record in dataset.train_records)
    test_files = sorted(record.relative_path for record in dataset.test_records)
    overlap_paths = sorted(set(train_files) & set(test_files))
    if overlap_paths:
        raise VerificationError(f"train/test file overlap detected: {overlap_paths}")

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
    standardizer = fit_window_standardizer(train_windows)
    if standardizer.fitted_split != "offline_training":
        raise VerificationError(
            "standardizer fit split drifted away from offline_training: "
            f"{standardizer.fitted_split}"
        )

    boundary_violations = []
    train_row_counts = {record.relative_path: record.row_count for record in train_records}
    test_row_counts = {record.relative_path: record.row_count for record in test_records}
    for window in train_windows.windows:
        row_count = train_row_counts[window.relative_path]
        if window.stop_row > row_count or window.start_row < 0:
            boundary_violations.append(window.relative_path)
    for window in test_windows.windows:
        row_count = test_row_counts[window.relative_path]
        if window.stop_row > row_count or window.start_row < 0:
            boundary_violations.append(window.relative_path)
    if boundary_violations:
        raise VerificationError(
            f"window boundary violation detected: {sorted(set(boundary_violations))}"
        )

    warnings: list[str] = []
    if train_windows.skipped_records:
        warnings.append(f"skipped_train_records={len(train_windows.skipped_records)}")
    if test_windows.skipped_records:
        warnings.append(f"skipped_test_records={len(test_windows.skipped_records)}")

    return {
        "passed": True,
        "train_files": train_files,
        "test_files": test_files,
        "overlap_count": len(overlap_paths),
        "overlap_paths": overlap_paths,
        "standardizer_fit_source_split": standardizer.fitted_split,
        "train_window_count": train_windows.window_count,
        "test_window_count": test_windows.window_count,
        "boundary_violation_count": 0,
        "test_derived_fit_detected": False,
        "calibration_steps_detected": [],
        "warnings": warnings,
    }


if __name__ == "__main__":
    raise SystemExit(main())
