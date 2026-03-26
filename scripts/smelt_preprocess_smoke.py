"""run a real-data exact-upstream preprocessing smoke check."""

from __future__ import annotations

import argparse
from pathlib import Path

from smelt.datasets import load_base_sensor_dataset
from smelt.preprocessing import (
    EXACT_UPSTREAM_DROPPED_COLUMNS,
    apply_window_standardizer,
    fit_window_standardizer,
    generate_split_windows,
    preprocess_split_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--diff-period", type=int, default=25)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_base_sensor_dataset(args.data_root)
    train_records = preprocess_split_records(
        dataset.train_records,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=args.diff_period,
    )
    test_records = preprocess_split_records(
        dataset.test_records,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=args.diff_period,
    )
    train_windows = generate_split_windows(
        train_records,
        window_size=args.window_size,
        stride=args.stride,
    )
    test_windows = generate_split_windows(
        test_records,
        window_size=args.window_size,
        stride=args.stride,
    )
    stats = fit_window_standardizer(train_windows)
    standardized_train = apply_window_standardizer(train_windows, stats)
    standardized_test = apply_window_standardizer(test_windows, stats)

    retained_columns = list(train_records[0].column_names)
    print(f"resolved_data_root: {dataset.resolved_data_root}")
    print(f"retained_columns: {retained_columns}")
    print(f"retained_channel_count: {len(retained_columns)}")
    print(f"differencing_period: {args.diff_period}")
    print(f"window_size: {args.window_size}")
    print(f"stride: {train_windows.stride}")
    print(f"train_record_count: {len(train_records)}")
    print(f"test_record_count: {len(test_records)}")
    print(f"train_window_count: {train_windows.window_count}")
    print(f"test_window_count: {test_windows.window_count}")
    print(
        f"standardization_shape: ({stats.window_count * stats.window_size}, {stats.feature_count})"
    )
    print(f"train_standardized_window_count: {standardized_train.window_count}")
    print(f"test_standardized_window_count: {standardized_test.window_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
