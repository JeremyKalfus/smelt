from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from smelt.evaluation import (
    EvaluationMetricError,
    ReportExportError,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_topk_accuracy,
    export_classification_report,
    load_category_mapping,
)


def test_acc_at_1_is_correct() -> None:
    topk = np.asarray(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 1, 0, 3, 4],
            [1, 2, 0, 3, 4],
        ],
        dtype=np.int64,
    )
    true_labels = np.asarray([0, 1, 2, 3], dtype=np.int64)

    assert compute_topk_accuracy(true_labels, topk, k=1) == 75.0


def test_acc_at_5_is_correct() -> None:
    topk = np.asarray(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 1, 0, 3, 4],
            [1, 2, 0, 3, 4],
        ],
        dtype=np.int64,
    )
    true_labels = np.asarray([0, 1, 2, 3], dtype=np.int64)

    assert compute_topk_accuracy(true_labels, topk, k=5) == 100.0


def test_macro_precision_recall_f1_match_known_example() -> None:
    metrics = compute_classification_metrics(
        class_names=("alpha", "beta", "gamma", "delta", "epsilon"),
        true_labels=[0, 1, 2, 3, 4],
        predicted_labels=[0, 0, 2, 3, 3],
        topk_indices=[
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [2, 1, 0, 3, 4],
            [3, 2, 1, 0, 4],
            [3, 4, 2, 1, 0],
        ],
        category_mapping={
            "alpha": "nuts",
            "beta": "spices",
            "gamma": "herbs",
            "delta": "fruits",
            "epsilon": "vegetables",
        },
    )

    assert metrics.acc_at_1 == 60.0
    assert metrics.acc_at_5 == 100.0
    assert metrics.precision_macro == 40.0
    assert metrics.recall_macro == 60.0
    assert round(metrics.f1_macro, 6) == round(46.666666666666664, 6)


def test_confusion_matrix_uses_deterministic_class_order() -> None:
    confusion = compute_confusion_matrix(
        true_labels=np.asarray([1, 0, 1, 2], dtype=np.int64),
        predicted_labels=np.asarray([1, 2, 1, 0], dtype=np.int64),
        class_count=3,
    )

    np.testing.assert_array_equal(
        confusion,
        np.asarray(
            [
                [0, 0, 1],
                [0, 2, 0],
                [1, 0, 0],
            ],
            dtype=np.int64,
        ),
    )


def test_per_category_accuracy_is_correct() -> None:
    metrics = compute_classification_metrics(
        class_names=("alpha", "beta", "gamma", "delta", "epsilon"),
        true_labels=[0, 1, 2, 3, 4],
        predicted_labels=[0, 0, 2, 3, 3],
        topk_indices=[
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [2, 1, 0, 3, 4],
            [3, 2, 1, 0, 4],
            [3, 4, 2, 1, 0],
        ],
        category_mapping={
            "alpha": "nuts",
            "beta": "spices",
            "gamma": "herbs",
            "delta": "fruits",
            "epsilon": "vegetables",
        },
    )

    by_category = {row.category: row for row in metrics.per_category}
    assert by_category["nuts"].acc_at_1 == 100.0
    assert by_category["spices"].acc_at_1 == 0.0
    assert by_category["vegetables"].acc_at_1 == 0.0
    assert by_category["vegetables"].acc_at_5 == 100.0


def test_metrics_fail_on_incomplete_category_mapping() -> None:
    with pytest.raises(EvaluationMetricError, match="missing classes"):
        compute_classification_metrics(
            class_names=("alpha", "beta"),
            true_labels=[0, 1],
            predicted_labels=[0, 1],
            topk_indices=[[0, 1], [1, 0]],
            category_mapping={"alpha": "nuts"},
        )


def test_metrics_fail_on_mapping_classes_outside_run_scope() -> None:
    with pytest.raises(EvaluationMetricError, match="outside this run"):
        compute_classification_metrics(
            class_names=("alpha", "beta"),
            true_labels=[0, 1],
            predicted_labels=[0, 1],
            topk_indices=[[0, 1], [1, 0]],
            category_mapping={
                "alpha": "nuts",
                "beta": "spices",
                "gamma": "herbs",
            },
        )


def test_report_export_writes_expected_files(tmp_path: Path) -> None:
    metrics = compute_classification_metrics(
        class_names=("alpha", "beta", "gamma", "delta", "epsilon"),
        true_labels=[0, 1, 2, 3, 4],
        predicted_labels=[0, 0, 2, 3, 3],
        topk_indices=[
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [2, 1, 0, 3, 4],
            [3, 2, 1, 0, 4],
            [3, 4, 2, 1, 0],
        ],
        category_mapping={
            "alpha": "nuts",
            "beta": "spices",
            "gamma": "herbs",
            "delta": "fruits",
            "epsilon": "vegetables",
        },
    )

    bundle = export_classification_report(
        output_root=tmp_path,
        run_name="deterministic_smoke",
        metrics=metrics,
        methods_summary={"retained_columns": ["NO2", "C2H5OH"]},
    )

    assert Path(bundle.summary_json).is_file()
    assert Path(bundle.confusion_matrix_csv).is_file()
    assert Path(bundle.per_category_accuracy_csv).is_file()
    with pytest.raises(ReportExportError, match="already exists"):
        export_classification_report(
            output_root=tmp_path,
            run_name="deterministic_smoke",
            metrics=metrics,
        )


def test_load_category_mapping_reads_explicit_json(tmp_path: Path) -> None:
    mapping_path = tmp_path / "category_map.json"
    mapping_path.write_text(json.dumps({"alpha": "nuts", "beta": "spices"}), encoding="utf-8")

    mapping = load_category_mapping(mapping_path)

    assert mapping == {"alpha": "nuts", "beta": "spices"}
