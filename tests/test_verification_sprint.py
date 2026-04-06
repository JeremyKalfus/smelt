from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from smelt.training.verification_sprint import (
    bootstrap_metrics,
    is_m04_selection_validation_locked,
    load_file_prediction_table,
)


def test_load_file_prediction_table_is_file_aware_and_deterministic(tmp_path: Path) -> None:
    prediction_path = tmp_path / "per_file_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
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
        writer.writerow(
            [
                "test",
                "offline_testing/a.csv",
                "/tmp/a.csv",
                "alpha",
                "alpha",
                "10",
                json.dumps(["alpha", "beta", "gamma"]),
            ]
        )
        writer.writerow(
            [
                "test",
                "offline_testing/b.csv",
                "/tmp/b.csv",
                "beta",
                "gamma",
                "11",
                json.dumps(["gamma", "beta", "alpha"]),
            ]
        )

    class_names = ("alpha", "beta", "gamma")
    first = load_file_prediction_table(prediction_path, class_names)
    second = load_file_prediction_table(prediction_path, class_names)

    assert first.relative_paths == ("offline_testing/a.csv", "offline_testing/b.csv")
    assert np.array_equal(first.true_labels, np.asarray([0, 1], dtype=np.int64))
    assert np.array_equal(first.predicted_labels, np.asarray([0, 2], dtype=np.int64))
    assert np.array_equal(first.topk_indices, second.topk_indices)


def test_bootstrap_metrics_is_deterministic() -> None:
    class_names = ("alpha", "beta")
    category_mapping = {"alpha": "nuts", "beta": "spices"}
    table = build_table(
        true_labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
        predicted_labels=np.asarray([0, 1, 1, 1], dtype=np.int64),
        topk_indices=np.asarray([[0, 1], [1, 0], [1, 0], [1, 0]], dtype=np.int64),
    )

    first = bootstrap_metrics(
        tables=[table],
        class_names=class_names,
        category_mapping=category_mapping,
        bootstrap_seed=9,
        n_resamples=32,
    )
    second = bootstrap_metrics(
        tables=[table],
        class_names=class_names,
        category_mapping=category_mapping,
        bootstrap_seed=9,
        n_resamples=32,
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_m04_selection_validation_lock_ignores_test_metric_advantage() -> None:
    selection_payload = {
        "selection_rule": {
            "final_tie_break_order": [
                "diversity_greedy_probabilities",
                "greedy_forward_probabilities",
                "weighted_probabilities_all",
                "mean_probabilities_all",
                "mean_logits_all",
            ]
        },
        "selected_method": "mean_probabilities_all",
        "candidates": [
            {
                "method_name": "mean_probabilities_all",
                "validation_file_acc@1": 88.0,
                "validation_file_macro_f1": 84.0,
                "test_file_acc@1": 80.0,
                "avg_pairwise_agreement": 0.7,
                "avg_pairwise_correlation": 0.8,
            },
            {
                "method_name": "mean_logits_all",
                "validation_file_acc@1": 86.0,
                "validation_file_macro_f1": 82.0,
                "test_file_acc@1": 99.0,
                "avg_pairwise_agreement": 0.6,
                "avg_pairwise_correlation": 0.7,
            },
        ],
    }

    assert is_m04_selection_validation_locked(selection_payload) is True


def build_table(
    *,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    topk_indices: np.ndarray,
):
    from smelt.training.verification_sprint import FilePredictionTable

    return FilePredictionTable(
        class_names=("alpha", "beta"),
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        topk_indices=topk_indices,
        relative_paths=("a.csv", "b.csv", "c.csv", "d.csv"),
    )
