from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smelt.datasets import BENCHMARK_SENSOR_COLUMNS, RAW_SENSOR_COLUMNS, SensorFileRecord
from smelt.datasets.contracts import TEST_SPLIT, TRAIN_SPLIT
from smelt.preprocessing import (
    EXACT_UPSTREAM_DROPPED_COLUMNS,
    StandardizationError,
    apply_window_standardizer,
    fit_window_standardizer,
    generate_record_windows,
    generate_split_windows,
    preprocess_sensor_record,
    preprocess_split_records,
    resolve_retained_columns,
    resolve_window_stride,
    subtract_first_row,
)


def make_sensor_record(
    row_values: list[list[float]],
    *,
    split: str = TRAIN_SPLIT,
    class_name: str = "odor_00",
    relative_path: str = "offline_training/odor_00/train_00.csv",
) -> SensorFileRecord:
    return SensorFileRecord(
        split=split,
        class_name=class_name,
        relative_path=relative_path,
        absolute_path=f"/tmp/{Path(relative_path).name}",
        column_names=RAW_SENSOR_COLUMNS,
        rows=tuple(tuple(float(value) for value in row) for row in row_values),
    )


def test_subtract_first_row_matches_upstream_baseline_subtraction() -> None:
    values = np.asarray(
        [
            [10.0, 20.0],
            [11.0, 24.0],
            [13.0, 27.0],
        ]
    )

    centered = subtract_first_row(values)

    np.testing.assert_allclose(
        centered,
        np.asarray(
            [
                [0.0, 0.0],
                [1.0, 4.0],
                [3.0, 7.0],
            ]
        ),
    )


def test_preprocess_sensor_record_drops_exact_upstream_columns_in_order() -> None:
    record = make_sensor_record(
        [
            list(range(1, 13)),
            list(range(2, 14)),
            list(range(4, 16)),
        ]
    )

    processed = preprocess_sensor_record(record)

    assert resolve_retained_columns(RAW_SENSOR_COLUMNS) == BENCHMARK_SENSOR_COLUMNS
    assert processed.column_names == BENCHMARK_SENSOR_COLUMNS
    assert processed.channel_count == 6
    np.testing.assert_allclose(
        processed.values,
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            ]
        ),
    )


def test_temporal_differencing_supports_zero_and_positive_periods() -> None:
    record = make_sensor_record(
        [
            list(range(1, 13)),
            list(range(2, 14)),
            list(range(4, 16)),
            list(range(7, 19)),
        ]
    )

    raw_processed = preprocess_sensor_record(record, dropped_columns=(), diff_period=0)
    diffed_processed = preprocess_sensor_record(record, dropped_columns=(), diff_period=2)

    np.testing.assert_allclose(raw_processed.values[:, 0], np.asarray([0.0, 1.0, 3.0, 6.0]))
    np.testing.assert_allclose(diffed_processed.values[:, 0], np.asarray([3.0, 5.0]))
    assert diffed_processed.row_count == 2


def test_generate_record_windows_handles_exact_fit() -> None:
    record = make_sensor_record([list(range(index, index + 12)) for index in range(1, 9)])
    processed = preprocess_sensor_record(record, dropped_columns=(), diff_period=0)

    windows = generate_record_windows(processed, window_size=8, stride=4)

    assert len(windows) == 1
    assert windows[0].start_row == 0
    assert windows[0].stop_row == 8


def test_generate_split_windows_uses_upstream_default_stride() -> None:
    record = make_sensor_record([list(range(index, index + 12)) for index in range(1, 9)])
    processed = preprocess_sensor_record(record, dropped_columns=(), diff_period=0)

    windowed_split = generate_split_windows((processed,), window_size=4, stride=None)

    assert resolve_window_stride(4, None) == 2
    assert windowed_split.stride == 2
    assert windowed_split.window_count == 3
    assert [window.start_row for window in windowed_split.windows] == [0, 2, 4]


def test_generate_split_windows_reports_too_short_records() -> None:
    record = make_sensor_record([list(range(index, index + 12)) for index in range(1, 4)])
    processed = preprocess_sensor_record(record, dropped_columns=(), diff_period=0)

    windowed_split = generate_split_windows((processed,), window_size=4, stride=None)

    assert windowed_split.window_count == 0
    assert len(windowed_split.skipped_records) == 1
    assert windowed_split.skipped_records[0].row_count == 3


def test_windows_never_cross_file_boundaries() -> None:
    first_record = make_sensor_record(
        [list(range(index, index + 12)) for index in range(1, 6)],
        relative_path="offline_training/odor_00/train_00.csv",
    )
    second_record = make_sensor_record(
        [list(range(index + 100, index + 112)) for index in range(1, 6)],
        class_name="odor_01",
        relative_path="offline_training/odor_01/train_00.csv",
    )

    windowed_split = generate_split_windows(
        (
            preprocess_sensor_record(first_record, dropped_columns=(), diff_period=0),
            preprocess_sensor_record(second_record, dropped_columns=(), diff_period=0),
        ),
        window_size=4,
        stride=2,
    )

    assert windowed_split.window_count == 2
    assert {window.relative_path for window in windowed_split.windows} == {
        "offline_training/odor_00/train_00.csv",
        "offline_training/odor_01/train_00.csv",
    }


def test_fit_window_standardizer_uses_train_only_windows() -> None:
    train_record = make_sensor_record(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
            [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36],
            [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
        ],
        split=TRAIN_SPLIT,
    )
    test_record = make_sensor_record(
        [
            [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            [12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122],
            [14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124],
            [16, 26, 36, 46, 56, 66, 76, 86, 96, 106, 116, 126],
        ],
        split=TEST_SPLIT,
        relative_path="offline_testing/odor_00/test_00.csv",
    )

    train_windows = generate_split_windows(
        preprocess_split_records((train_record,), dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS),
        window_size=2,
        stride=2,
    )
    test_windows = generate_split_windows(
        preprocess_split_records((test_record,), dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS),
        window_size=2,
        stride=2,
    )

    stats = fit_window_standardizer(train_windows)
    standardized_test = apply_window_standardizer(test_windows, stats)

    assert stats.fitted_split == TRAIN_SPLIT
    assert stats.window_count == 2
    assert stats.sample_count == 4
    assert stats.feature_count == 6
    assert len(stats.mean) == 6
    assert standardized_test.window_count == test_windows.window_count
    with pytest.raises(StandardizationError, match="training split only"):
        fit_window_standardizer(test_windows)


def test_real_data_preprocessing_smoke_runs_when_snapshot_is_available() -> None:
    data_root = find_local_smellnet_snapshot()
    if data_root is None:
        pytest.skip("local smell-net snapshot is not available")

    from smelt.datasets import load_base_sensor_dataset

    dataset = load_base_sensor_dataset(data_root)
    train_records = preprocess_split_records(
        dataset.train_records,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=25,
    )
    test_records = preprocess_split_records(
        dataset.test_records,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=25,
    )
    train_windows = generate_split_windows(train_records, window_size=100, stride=None)
    test_windows = generate_split_windows(test_records, window_size=100, stride=None)
    stats = fit_window_standardizer(train_windows)

    assert train_records[0].column_names == BENCHMARK_SENSOR_COLUMNS
    assert train_records[0].channel_count == 6
    assert train_windows.stride == 50
    assert train_windows.window_count > 0
    assert test_windows.window_count > 0
    assert stats.feature_count == 6
    assert len(stats.mean) == 6


def find_local_smellnet_snapshot() -> Path | None:
    snapshots_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--DeweiFeng--smell-net"
        / "snapshots"
    )
    if not snapshots_root.is_dir():
        return None
    candidates = sorted(
        path / "data" for path in snapshots_root.iterdir() if (path / "data").is_dir()
    )
    if not candidates:
        return None
    return candidates[-1]
