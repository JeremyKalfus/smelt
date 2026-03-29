from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smelt.datasets import RAW_SENSOR_COLUMNS, SensorFileRecord, preprocess_split_records_for_view
from smelt.datasets.contracts import TEST_SPLIT, TRAIN_SPLIT
from smelt.datasets.research_views import FUSED_RAW_DIFF_VIEW, ResearchViewError
from smelt.preprocessing import (
    EXACT_UPSTREAM_DROPPED_COLUMNS,
    StandardizationError,
    apply_window_standardizer,
    fit_window_standardizer,
    generate_split_windows,
    preprocess_split_records,
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


def make_long_sensor_rows(row_count: int) -> list[list[float]]:
    return [
        [float(index + column_offset) for column_offset in range(12)] for index in range(row_count)
    ]


def test_raw_aligned_length_matches_diff_length_for_positive_g() -> None:
    record = make_sensor_record(make_long_sensor_rows(10))

    raw_aligned = preprocess_split_records_for_view(
        (record,),
        view_mode="raw_aligned",
        diff_period=3,
    )[0]
    diff = preprocess_split_records_for_view(
        (record,),
        view_mode="diff",
        diff_period=3,
    )[0]

    assert raw_aligned.row_count == diff.row_count
    assert raw_aligned.row_count == 7


def test_diff_semantics_match_expected_values() -> None:
    record = make_sensor_record(
        [
            list(range(1, 13)),
            list(range(2, 14)),
            list(range(4, 16)),
            list(range(7, 19)),
        ]
    )

    diff = preprocess_split_records_for_view(
        (record,),
        view_mode="diff",
        dropped_columns=(),
        diff_period=2,
    )[0]

    np.testing.assert_allclose(diff.values[:, 0], np.asarray([3.0, 5.0]))


def test_feature_counts_match_expected_view_contract() -> None:
    record = make_sensor_record(make_long_sensor_rows(40))

    raw_aligned = preprocess_split_records_for_view(
        (record,),
        view_mode="raw_aligned",
        diff_period=5,
    )[0]
    diff = preprocess_split_records_for_view(
        (record,),
        view_mode="diff",
        diff_period=5,
    )[0]
    fused = preprocess_split_records_for_view(
        (record,),
        view_mode="fused_raw_diff",
        diff_period=5,
    )[0]

    assert raw_aligned.channel_count == 6
    assert diff.channel_count == 6
    assert fused.channel_count == 12


def test_research_view_windows_never_cross_file_boundaries() -> None:
    first = make_sensor_record(
        make_long_sensor_rows(140),
        relative_path="offline_training/odor_00/train_00.csv",
    )
    second = make_sensor_record(
        make_long_sensor_rows(140),
        class_name="odor_01",
        relative_path="offline_training/odor_01/train_00.csv",
    )

    windowed = generate_split_windows(
        preprocess_split_records_for_view(
            (first, second),
            view_mode="fused_raw_diff",
            diff_period=25,
        ),
        window_size=100,
        stride=50,
    )

    assert {window.relative_path for window in windowed.windows} == {
        "offline_training/odor_00/train_00.csv",
        "offline_training/odor_01/train_00.csv",
    }


def test_view_window_counts_match_for_same_input() -> None:
    record = make_sensor_record(make_long_sensor_rows(180))

    counts = {}
    for view_mode in ("raw_aligned", "diff", "fused_raw_diff"):
        windowed = generate_split_windows(
            preprocess_split_records_for_view(
                (record,),
                view_mode=view_mode,
                diff_period=25,
            ),
            window_size=100,
            stride=50,
        )
        counts[view_mode] = windowed.window_count

    assert counts["raw_aligned"] == counts["diff"] == counts["fused_raw_diff"]


def test_train_only_standardization_supports_fused_windows() -> None:
    train_record = make_sensor_record(make_long_sensor_rows(160), split=TRAIN_SPLIT)
    test_record = make_sensor_record(
        make_long_sensor_rows(160),
        split=TEST_SPLIT,
        relative_path="offline_testing/odor_00/test_00.csv",
    )

    train_windows = generate_split_windows(
        preprocess_split_records_for_view(
            (train_record,),
            view_mode="fused_raw_diff",
            diff_period=25,
        ),
        window_size=100,
        stride=50,
    )
    test_windows = generate_split_windows(
        preprocess_split_records_for_view(
            (test_record,),
            view_mode="fused_raw_diff",
            diff_period=25,
        ),
        window_size=100,
        stride=50,
    )

    stats = fit_window_standardizer(train_windows)
    standardized_test = apply_window_standardizer(test_windows, stats)

    assert stats.feature_count == 12
    assert standardized_test.window_count == test_windows.window_count
    with pytest.raises(StandardizationError, match="training split only"):
        fit_window_standardizer(test_windows)


def test_fused_view_requires_positive_diff_period() -> None:
    record = make_sensor_record(make_long_sensor_rows(40))

    with pytest.raises(ResearchViewError, match="requires diff_period > 0"):
        preprocess_split_records_for_view(
            (record,),
            view_mode=FUSED_RAW_DIFF_VIEW,
            diff_period=0,
        )


def test_exact_upstream_regression_counts_remain_unchanged_when_snapshot_is_available() -> None:
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

    assert train_records[0].channel_count == 6
    assert train_windows.stride == 50
    assert train_windows.window_count == 2512
    assert test_windows.window_count == 502


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
