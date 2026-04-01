"""moonshot-enhanced-setting dataset helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from smelt.preprocessing.base import (
    EXACT_UPSTREAM_DROPPED_COLUMNS,
    PreprocessedSensorRecord,
    apply_temporal_differencing,
    project_columns,
    sensor_record_to_array,
    subtract_first_row,
)

from .contracts import BaseSensorDataset, SensorFileRecord

if TYPE_CHECKING:
    from smelt.preprocessing.standardize import StandardizationStats
    from smelt.preprocessing.windows import WindowedSplit

MOONSHOT_TRACK = "moonshot-enhanced-setting"
MOONSHOT_ALL12_CHANNEL_SET = "all12"
MOONSHOT_BENCHMARK6_CHANNEL_SET = "benchmark6"
MOONSHOT_EXTRA6_CHANNEL_SET = "extra6"
MOONSHOT_CHANNEL_SETS = (
    MOONSHOT_ALL12_CHANNEL_SET,
    MOONSHOT_BENCHMARK6_CHANNEL_SET,
    MOONSHOT_EXTRA6_CHANNEL_SET,
)


class MoonshotDataError(Exception):
    """raised when moonshot data preparation cannot proceed safely."""


@dataclass(slots=True)
class GroupedValidationSplit:
    train_records: tuple[SensorFileRecord, ...]
    validation_records: tuple[SensorFileRecord, ...]


@dataclass(slots=True)
class MoonshotPreparedSplits:
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    retained_columns: tuple[str, ...]
    channel_set: str
    train_records: tuple[SensorFileRecord, ...]
    validation_records: tuple[SensorFileRecord, ...]
    test_records: tuple[SensorFileRecord, ...]
    standardized_train_split: WindowedSplit
    standardized_validation_split: WindowedSplit
    standardized_test_split: WindowedSplit
    standardizer: StandardizationStats
    view_manifest: dict[str, Any]


def deterministic_grouped_validation_split(
    train_records: tuple[SensorFileRecord, ...],
    *,
    validation_files_per_class: int,
) -> GroupedValidationSplit:
    if validation_files_per_class <= 0:
        raise MoonshotDataError(
            "validation_files_per_class must be positive for grouped validation"
        )
    grouped: dict[str, list[SensorFileRecord]] = defaultdict(list)
    for record in train_records:
        grouped[record.class_name].append(record)
    selected_train: list[SensorFileRecord] = []
    selected_validation: list[SensorFileRecord] = []
    for class_name in sorted(grouped):
        records = sorted(grouped[class_name], key=lambda record: record.relative_path)
        if len(records) <= validation_files_per_class:
            raise MoonshotDataError(
                f"class {class_name!r} does not have enough files for grouped validation"
            )
        selected_train.extend(records[:-validation_files_per_class])
        selected_validation.extend(records[-validation_files_per_class:])
    overlap = {record.relative_path for record in selected_train} & {
        record.relative_path for record in selected_validation
    }
    if overlap:
        raise MoonshotDataError(f"grouped validation split leaked files: {sorted(overlap)}")
    return GroupedValidationSplit(
        train_records=tuple(sorted(selected_train, key=lambda record: record.relative_path)),
        validation_records=tuple(
            sorted(selected_validation, key=lambda record: record.relative_path)
        ),
    )


def prepare_moonshot_window_splits(
    dataset: BaseSensorDataset,
    *,
    diff_period: int,
    window_size: int,
    stride: int | None,
    validation_files_per_class: int,
    channel_set: str = MOONSHOT_ALL12_CHANNEL_SET,
) -> MoonshotPreparedSplits:
    from smelt.preprocessing.standardize import apply_window_standardizer, fit_window_standardizer
    from smelt.preprocessing.windows import generate_split_windows

    resolved_channel_set = resolve_moonshot_channel_set(channel_set)
    grouped_split = deterministic_grouped_validation_split(
        dataset.train_records,
        validation_files_per_class=validation_files_per_class,
    )
    train_records = preprocess_moonshot_records(
        grouped_split.train_records,
        diff_period=diff_period,
        channel_set=resolved_channel_set,
    )
    validation_records = preprocess_moonshot_records(
        grouped_split.validation_records,
        diff_period=diff_period,
        channel_set=resolved_channel_set,
    )
    test_records = preprocess_moonshot_records(
        dataset.test_records,
        diff_period=diff_period,
        channel_set=resolved_channel_set,
    )
    train_windows = generate_split_windows(
        train_records,
        window_size=window_size,
        stride=stride,
    )
    validation_windows = generate_split_windows(
        validation_records,
        window_size=window_size,
        stride=stride,
    )
    test_windows = generate_split_windows(
        test_records,
        window_size=window_size,
        stride=stride,
    )
    if train_windows.window_count == 0:
        raise MoonshotDataError("moonshot train split produced zero windows")
    if validation_windows.window_count == 0:
        raise MoonshotDataError("moonshot validation split produced zero windows")
    if test_windows.window_count == 0:
        raise MoonshotDataError("moonshot test split produced zero windows")

    standardizer = fit_window_standardizer(train_windows)
    standardized_train = apply_window_standardizer(train_windows, standardizer)
    standardized_validation = apply_window_standardizer(validation_windows, standardizer)
    standardized_test = apply_window_standardizer(test_windows, standardizer)
    feature_names = standardized_train.column_names
    class_names = tuple(sorted(dataset.class_vocab))
    view_manifest = {
        "track": MOONSHOT_TRACK,
        "view_mode": f"diff_{resolved_channel_set}",
        "channel_set": resolved_channel_set,
        "resolved_data_root": dataset.resolved_data_root,
        "differencing_period": diff_period,
        "window_size": window_size,
        "stride": standardized_train.stride,
        "retained_columns": list(feature_names),
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "train_record_count": len(train_records),
        "validation_record_count": len(validation_records),
        "test_record_count": len(test_records),
        "train_window_count": standardized_train.window_count,
        "validation_window_count": standardized_validation.window_count,
        "test_window_count": standardized_test.window_count,
        "standardization_shape": [standardizer.sample_count, standardizer.feature_count],
        "validation_files_per_class": validation_files_per_class,
    }
    return MoonshotPreparedSplits(
        class_names=class_names,
        feature_names=feature_names,
        retained_columns=feature_names,
        channel_set=resolved_channel_set,
        train_records=grouped_split.train_records,
        validation_records=grouped_split.validation_records,
        test_records=dataset.test_records,
        standardized_train_split=standardized_train,
        standardized_validation_split=standardized_validation,
        standardized_test_split=standardized_test,
        standardizer=standardizer,
        view_manifest=view_manifest,
    )


def resolve_moonshot_channel_set(channel_set: str) -> str:
    if channel_set not in MOONSHOT_CHANNEL_SETS:
        raise MoonshotDataError(
            f"unsupported moonshot channel_set {channel_set!r}; expected {MOONSHOT_CHANNEL_SETS}"
        )
    return channel_set


def preprocess_moonshot_records(
    records: tuple[SensorFileRecord, ...],
    *,
    diff_period: int,
    channel_set: str,
) -> tuple[PreprocessedSensorRecord, ...]:
    return tuple(
        preprocess_moonshot_record(
            record,
            diff_period=diff_period,
            channel_set=channel_set,
        )
        for record in records
    )


def preprocess_moonshot_record(
    record: SensorFileRecord,
    *,
    diff_period: int,
    channel_set: str,
) -> PreprocessedSensorRecord:
    retained_columns = resolve_channel_set_columns(record.column_names, channel_set)
    baseline_values = subtract_first_row(sensor_record_to_array(record))
    projected_values = project_columns(
        baseline_values,
        record.column_names,
        retained_columns,
    )
    diffed_values = apply_temporal_differencing(projected_values, diff_period)
    return PreprocessedSensorRecord(
        split=record.split,
        class_name=record.class_name,
        relative_path=record.relative_path,
        absolute_path=record.absolute_path,
        column_names=retained_columns,
        values=diffed_values,
        source_row_count=record.row_count,
        diff_period=diff_period,
    )


def resolve_channel_set_columns(
    observed_columns: tuple[str, ...],
    channel_set: str,
) -> tuple[str, ...]:
    resolved_channel_set = resolve_moonshot_channel_set(channel_set)
    if resolved_channel_set == MOONSHOT_ALL12_CHANNEL_SET:
        return observed_columns
    if resolved_channel_set == MOONSHOT_BENCHMARK6_CHANNEL_SET:
        return tuple(
            column_name
            for column_name in observed_columns
            if column_name not in set(EXACT_UPSTREAM_DROPPED_COLUMNS)
        )
    return tuple(
        column_name
        for column_name in observed_columns
        if column_name in set(EXACT_UPSTREAM_DROPPED_COLUMNS)
    )


def write_moonshot_view_manifest(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stack_window_labels(
    window_split: WindowedSplit,
    class_names: tuple[str, ...],
) -> np.ndarray:
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    return np.asarray(
        [class_to_index[window.class_name] for window in window_split.windows],
        dtype=np.int64,
    )
