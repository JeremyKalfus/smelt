"""run a real-data smoke check for research-extension sensor views."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from smelt.datasets import (
    BENCHMARK_SENSOR_COLUMNS,
    DIFF_VIEW,
    FUSED_RAW_DIFF_VIEW,
    RAW_ALIGNED_VIEW,
    load_base_sensor_dataset,
    preprocess_split_records_for_view,
)
from smelt.preprocessing import (
    EXACT_UPSTREAM_DROPPED_COLUMNS,
    apply_window_standardizer,
    fit_window_standardizer,
    generate_split_windows,
    preprocess_split_records,
    resolve_window_stride,
    stack_window_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--diff-period", type=int, default=25)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/manifests"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_base_sensor_dataset(args.data_root)
    artifact_dir = build_artifact_dir(args.artifact_root)
    artifact_dir.mkdir(parents=True, exist_ok=False)

    view_summaries = []
    for view_mode in (RAW_ALIGNED_VIEW, DIFF_VIEW, FUSED_RAW_DIFF_VIEW):
        summary = build_view_summary(
            dataset=dataset,
            view_mode=view_mode,
            diff_period=args.diff_period,
            window_size=args.window_size,
            stride=args.stride,
        )
        view_summaries.append(summary)

    regression_summary = build_exact_upstream_regression(
        dataset=dataset,
        diff_period=args.diff_period,
        window_size=args.window_size,
        stride=args.stride,
    )

    research_manifest = {
        "resolved_data_root": dataset.resolved_data_root,
        "retained_columns": list(BENCHMARK_SENSOR_COLUMNS),
        "view_summaries": view_summaries,
        "warnings": [],
    }
    research_manifest_path = artifact_dir / "research_view_manifest.json"
    research_manifest_path.write_text(
        json.dumps(research_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    regression_path = artifact_dir / "exact_upstream_regression.json"
    regression_path.write_text(
        json.dumps(regression_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"artifact_dir: {artifact_dir}")
    print(f"research_manifest_path: {research_manifest_path}")
    print(f"exact_upstream_regression_path: {regression_path}")
    print(f"resolved_data_root: {dataset.resolved_data_root}")
    print(f"retained_columns: {list(BENCHMARK_SENSOR_COLUMNS)}")
    for summary in view_summaries:
        print(f"{summary['view_mode']}_feature_count: {summary['feature_count']}")
        print(f"{summary['view_mode']}_train_window_count: {summary['train_window_count']}")
        print(f"{summary['view_mode']}_test_window_count: {summary['test_window_count']}")
        print(f"{summary['view_mode']}_standardization_shape: {summary['standardization_shape']}")
    print(
        "exact_upstream_regression:"
        f" retained_channel_count={regression_summary['retained_channel_count']},"
        f" train_window_count={regression_summary['train_window_count']},"
        f" test_window_count={regression_summary['test_window_count']},"
        f" stride={regression_summary['stride']}"
    )
    return 0


def build_view_summary(
    *,
    dataset,
    view_mode: str,
    diff_period: int,
    window_size: int,
    stride: int | None,
) -> dict[str, object]:
    train_records = preprocess_split_records_for_view(
        dataset.train_records,
        view_mode=view_mode,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=diff_period,
    )
    test_records = preprocess_split_records_for_view(
        dataset.test_records,
        view_mode=view_mode,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=diff_period,
    )
    train_windows = generate_split_windows(
        train_records,
        window_size=window_size,
        stride=stride,
    )
    test_windows = generate_split_windows(
        test_records,
        window_size=window_size,
        stride=stride,
    )
    stats = fit_window_standardizer(train_windows)
    standardized_train = apply_window_standardizer(train_windows, stats)
    standardized_test = apply_window_standardizer(test_windows, stats)

    feature_names = list(train_records[0].column_names)
    sample_window_shape = list(standardized_train.windows[0].values.shape)
    return {
        "view_mode": view_mode,
        "resolved_data_root": dataset.resolved_data_root,
        "retained_columns": list(BENCHMARK_SENSOR_COLUMNS),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "differencing_period": diff_period,
        "window_size": window_size,
        "stride": train_windows.stride,
        "train_record_count": len(train_records),
        "test_record_count": len(test_records),
        "train_window_count": train_windows.window_count,
        "test_window_count": test_windows.window_count,
        "standardization_shape": [stats.sample_count, stats.feature_count],
        "sample_window_shape": sample_window_shape,
        "sample_train_tensor_shape": list(stack_window_values(standardized_train.windows).shape),
        "sample_test_tensor_shape": list(stack_window_values(standardized_test.windows).shape),
        "warnings": [],
    }


def build_exact_upstream_regression(
    *,
    dataset,
    diff_period: int,
    window_size: int,
    stride: int | None,
) -> dict[str, object]:
    train_records = preprocess_split_records(
        dataset.train_records,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=diff_period,
    )
    test_records = preprocess_split_records(
        dataset.test_records,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=diff_period,
    )
    train_windows = generate_split_windows(train_records, window_size=window_size, stride=stride)
    test_windows = generate_split_windows(test_records, window_size=window_size, stride=stride)
    summary = {
        "resolved_data_root": dataset.resolved_data_root,
        "retained_columns": list(train_records[0].column_names),
        "retained_channel_count": len(train_records[0].column_names),
        "differencing_period": diff_period,
        "window_size": window_size,
        "stride": train_windows.stride,
        "expected_default_stride": resolve_window_stride(window_size, stride),
        "train_window_count": train_windows.window_count,
        "test_window_count": test_windows.window_count,
    }
    if (
        summary["retained_channel_count"] != 6
        or summary["stride"] != 50
        or summary["train_window_count"] != 2512
        or summary["test_window_count"] != 502
    ):
        raise SystemExit(
            f"exact-upstream regression drift detected: {json.dumps(summary, sort_keys=True)}"
        )
    return summary


def build_artifact_dir(artifact_root: Path) -> Path:
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d-%H%M%S")
    return artifact_root / f"research_view_smoke-{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())
