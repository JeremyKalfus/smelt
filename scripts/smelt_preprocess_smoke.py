"""run a real-data exact-upstream preprocessing smoke check."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--emit", type=Path, default=None)
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
    summary = {
        "resolved_data_root": dataset.resolved_data_root,
        "retained_columns": retained_columns,
        "retained_channel_count": len(retained_columns),
        "differencing_period": args.diff_period,
        "window_size": args.window_size,
        "stride": train_windows.stride,
        "train_record_count": len(train_records),
        "test_record_count": len(test_records),
        "train_window_count": train_windows.window_count,
        "test_window_count": test_windows.window_count,
        "standardization_shape": [
            stats.window_count * stats.window_size,
            stats.feature_count,
        ],
        "train_standardized_window_count": standardized_train.window_count,
        "test_standardized_window_count": standardized_test.window_count,
    }
    if args.emit is not None:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
