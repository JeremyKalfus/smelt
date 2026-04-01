from __future__ import annotations

import numpy as np

from smelt.evaluation import (
    MEAN_LOGITS_AGGREGATOR,
    build_prediction_bundle_payload,
    evaluate_file_level_aggregation,
    load_prediction_bundle,
)
from smelt.preprocessing import SensorWindow


def test_file_level_aggregation_is_deterministic(tmp_path) -> None:
    bundle_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        bundle_path,
        **build_prediction_bundle_payload(
            class_names=("alpha", "beta", "gamma", "delta", "epsilon"),
            true_labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
            predicted_labels=np.asarray([0, 0, 2, 1], dtype=np.int64),
            topk_indices=np.asarray(
                [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [2, 1, 0, 3, 4],
                    [1, 2, 0, 3, 4],
                ],
                dtype=np.int64,
            ),
            logits=np.asarray(
                [
                    [3.0, 1.0, 0.0, 0.0, 0.0],
                    [2.5, 1.0, 0.0, 0.0, 0.0],
                    [0.1, 1.1, 1.2, 0.0, 0.0],
                    [0.1, 1.6, 0.8, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            windows=make_windows(),
        ),
    )
    bundle = load_prediction_bundle(bundle_path)
    category_mapping = {
        "alpha": "nuts",
        "beta": "spices",
        "gamma": "herbs",
        "delta": "fruits",
        "epsilon": "vegetables",
    }

    first_metrics, first_bundle = evaluate_file_level_aggregation(
        bundle,
        category_mapping=category_mapping,
        aggregator=MEAN_LOGITS_AGGREGATOR,
    )
    second_metrics, second_bundle = evaluate_file_level_aggregation(
        bundle,
        category_mapping=category_mapping,
        aggregator=MEAN_LOGITS_AGGREGATOR,
    )

    assert first_metrics.summary_dict() == second_metrics.summary_dict()
    assert [row.relative_path for row in first_bundle.rows] == [
        "offline_testing/alpha/file_a.csv",
        "offline_testing/beta/file_b.csv",
    ]
    assert [row.predicted_class for row in first_bundle.rows] == [
        row.predicted_class for row in second_bundle.rows
    ]


def test_prediction_bundle_round_trip_preserves_file_metadata(tmp_path) -> None:
    bundle_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        bundle_path,
        **build_prediction_bundle_payload(
            class_names=("alpha", "beta", "gamma", "delta", "epsilon"),
            true_labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
            predicted_labels=np.asarray([0, 0, 2, 1], dtype=np.int64),
            topk_indices=np.asarray(
                [
                    [0, 1, 2, 3, 4],
                    [0, 1, 2, 3, 4],
                    [2, 1, 0, 3, 4],
                    [1, 2, 0, 3, 4],
                ],
                dtype=np.int64,
            ),
            logits=np.asarray(
                [
                    [3.0, 1.0, 0.0, 0.0, 0.0],
                    [2.5, 1.0, 0.0, 0.0, 0.0],
                    [0.1, 1.1, 1.2, 0.0, 0.0],
                    [0.1, 1.6, 0.8, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
            windows=make_windows(),
        ),
    )

    loaded = load_prediction_bundle(bundle_path)

    assert loaded.relative_paths == (
        "offline_testing/alpha/file_a.csv",
        "offline_testing/alpha/file_a.csv",
        "offline_testing/beta/file_b.csv",
        "offline_testing/beta/file_b.csv",
    )
    assert loaded.absolute_paths == (
        "/tmp/file_a.csv",
        "/tmp/file_a.csv",
        "/tmp/file_b.csv",
        "/tmp/file_b.csv",
    )
    assert loaded.splits == (
        "offline_testing",
        "offline_testing",
        "offline_testing",
        "offline_testing",
    )
    np.testing.assert_array_equal(loaded.window_indices, np.asarray([0, 1, 0, 1]))


def make_windows() -> tuple[SensorWindow, ...]:
    return (
        SensorWindow(
            split="offline_testing",
            class_name="alpha",
            relative_path="offline_testing/alpha/file_a.csv",
            absolute_path="/tmp/file_a.csv",
            column_names=("c0", "c1"),
            window_index=0,
            start_row=0,
            stop_row=2,
            values=np.zeros((2, 2), dtype=np.float64),
        ),
        SensorWindow(
            split="offline_testing",
            class_name="alpha",
            relative_path="offline_testing/alpha/file_a.csv",
            absolute_path="/tmp/file_a.csv",
            column_names=("c0", "c1"),
            window_index=1,
            start_row=2,
            stop_row=4,
            values=np.zeros((2, 2), dtype=np.float64),
        ),
        SensorWindow(
            split="offline_testing",
            class_name="beta",
            relative_path="offline_testing/beta/file_b.csv",
            absolute_path="/tmp/file_b.csv",
            column_names=("c0", "c1"),
            window_index=0,
            start_row=0,
            stop_row=2,
            values=np.zeros((2, 2), dtype=np.float64),
        ),
        SensorWindow(
            split="offline_testing",
            class_name="beta",
            relative_path="offline_testing/beta/file_b.csv",
            absolute_path="/tmp/file_b.csv",
            column_names=("c0", "c1"),
            window_index=1,
            start_row=2,
            stop_row=4,
            values=np.zeros((2, 2), dtype=np.float64),
        ),
    )
