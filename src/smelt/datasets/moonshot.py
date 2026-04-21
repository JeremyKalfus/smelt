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
class GroupedValidationFold:
    fold_index: int
    fold_count: int
    train_records: tuple[SensorFileRecord, ...]
    validation_records: tuple[SensorFileRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "fold_count": self.fold_count,
            "train_relative_paths": [record.relative_path for record in self.train_records],
            "validation_relative_paths": [
                record.relative_path for record in self.validation_records
            ],
            "validation_relative_paths_by_class": {
                record.class_name: record.relative_path for record in self.validation_records
            },
        }


@dataclass(slots=True)
class GroupedFoldManifest:
    fold_count: int
    files_per_class: int
    class_names: tuple[str, ...]
    folds: tuple[GroupedValidationFold, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_count": self.fold_count,
            "files_per_class": self.files_per_class,
            "class_names": list(self.class_names),
            "folds": [fold.to_dict() for fold in self.folds],
        }


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
    standardized_validation_split: WindowedSplit | None
    standardized_test_split: WindowedSplit | None
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
    return validate_grouped_validation_split(
        train_records=tuple(sorted(selected_train, key=lambda record: record.relative_path)),
        validation_records=tuple(
            sorted(selected_validation, key=lambda record: record.relative_path)
        ),
    )


def validate_grouped_validation_split(
    *,
    train_records: tuple[SensorFileRecord, ...],
    validation_records: tuple[SensorFileRecord, ...],
) -> GroupedValidationSplit:
    overlap = {record.relative_path for record in train_records} & {
        record.relative_path for record in validation_records
    }
    if overlap:
        raise MoonshotDataError(f"grouped validation split leaked files: {sorted(overlap)}")
    return GroupedValidationSplit(
        train_records=tuple(sorted(train_records, key=lambda record: record.relative_path)),
        validation_records=tuple(
            sorted(validation_records, key=lambda record: record.relative_path)
        ),
    )


def build_grouped_cv_fold_manifest(
    train_records: tuple[SensorFileRecord, ...],
    *,
    fold_count: int = 5,
) -> GroupedFoldManifest:
    if fold_count <= 1:
        raise MoonshotDataError("fold_count must be greater than one for grouped cv")
    grouped: dict[str, list[SensorFileRecord]] = defaultdict(list)
    for record in train_records:
        grouped[record.class_name].append(record)
    if not grouped:
        raise MoonshotDataError("grouped cv requires at least one training record")

    grouped_records = {
        class_name: tuple(sorted(records, key=lambda record: record.relative_path))
        for class_name, records in grouped.items()
    }
    for class_name, records in grouped_records.items():
        if len(records) != fold_count:
            raise MoonshotDataError(
                f"class {class_name!r} has {len(records)} training files; "
                f"grouped {fold_count}-fold cv requires exactly {fold_count}"
            )

    folds: list[GroupedValidationFold] = []
    for fold_index in range(fold_count):
        selected_train: list[SensorFileRecord] = []
        selected_validation: list[SensorFileRecord] = []
        for class_name in sorted(grouped_records):
            records = grouped_records[class_name]
            selected_validation.append(records[fold_index])
            selected_train.extend(
                record for record_index, record in enumerate(records) if record_index != fold_index
            )
        split = validate_grouped_validation_split(
            train_records=tuple(selected_train),
            validation_records=tuple(selected_validation),
        )
        folds.append(
            GroupedValidationFold(
                fold_index=fold_index,
                fold_count=fold_count,
                train_records=split.train_records,
                validation_records=split.validation_records,
            )
        )
    return GroupedFoldManifest(
        fold_count=fold_count,
        files_per_class=fold_count,
        class_names=tuple(sorted(grouped_records)),
        folds=tuple(folds),
    )


def grouped_cv_validation_split(
    train_records: tuple[SensorFileRecord, ...],
    *,
    fold_index: int,
    fold_count: int = 5,
) -> GroupedValidationSplit:
    manifest = build_grouped_cv_fold_manifest(train_records, fold_count=fold_count)
    if fold_index < 0 or fold_index >= manifest.fold_count:
        raise MoonshotDataError(
            f"fold_index {fold_index} is out of range for fold_count={manifest.fold_count}"
        )
    fold = manifest.folds[fold_index]
    return GroupedValidationSplit(
        train_records=fold.train_records,
        validation_records=fold.validation_records,
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
    grouped_split = deterministic_grouped_validation_split(
        dataset.train_records,
        validation_files_per_class=validation_files_per_class,
    )
    return prepare_moonshot_window_splits_from_records(
        class_names=tuple(sorted(dataset.class_vocab)),
        resolved_data_root=dataset.resolved_data_root,
        train_records=grouped_split.train_records,
        validation_records=grouped_split.validation_records,
        test_records=dataset.test_records,
        diff_period=diff_period,
        window_size=window_size,
        stride=stride,
        channel_set=channel_set,
        view_manifest_updates={
            "split_strategy": "deterministic_grouped_holdout",
            "validation_files_per_class": validation_files_per_class,
        },
    )


def prepare_moonshot_window_splits_from_records(
    *,
    class_names: tuple[str, ...],
    resolved_data_root: str,
    train_records: tuple[SensorFileRecord, ...],
    validation_records: tuple[SensorFileRecord, ...] = (),
    test_records: tuple[SensorFileRecord, ...] = (),
    diff_period: int,
    window_size: int,
    stride: int | None,
    channel_set: str = MOONSHOT_ALL12_CHANNEL_SET,
    view_manifest_updates: dict[str, Any] | None = None,
) -> MoonshotPreparedSplits:
    from smelt.preprocessing.standardize import apply_window_standardizer, fit_window_standardizer
    from smelt.preprocessing.windows import generate_split_windows

    resolved_channel_set = resolve_moonshot_channel_set(channel_set)
    processed_train_records = preprocess_moonshot_records(
        train_records,
        diff_period=diff_period,
        channel_set=resolved_channel_set,
    )
    processed_validation_records = preprocess_moonshot_records(
        validation_records,
        diff_period=diff_period,
        channel_set=resolved_channel_set,
    )
    processed_test_records = preprocess_moonshot_records(
        test_records,
        diff_period=diff_period,
        channel_set=resolved_channel_set,
    )
    train_windows = generate_split_windows(
        processed_train_records,
        window_size=window_size,
        stride=stride,
    )
    if train_windows.window_count == 0:
        raise MoonshotDataError("moonshot train split produced zero windows")
    standardizer = fit_window_standardizer(train_windows)
    standardized_train = apply_window_standardizer(train_windows, standardizer)

    standardized_validation = None
    if processed_validation_records:
        validation_windows = generate_split_windows(
            processed_validation_records,
            window_size=window_size,
            stride=stride,
        )
        if validation_windows.window_count == 0:
            raise MoonshotDataError("moonshot validation split produced zero windows")
        standardized_validation = apply_window_standardizer(validation_windows, standardizer)

    standardized_test = None
    if processed_test_records:
        test_windows = generate_split_windows(
            processed_test_records,
            window_size=window_size,
            stride=stride,
        )
        if test_windows.window_count == 0:
            raise MoonshotDataError("moonshot test split produced zero windows")
        standardized_test = apply_window_standardizer(test_windows, standardizer)

    feature_names = standardized_train.column_names
    view_manifest = {
        "track": MOONSHOT_TRACK,
        "view_mode": f"diff_{resolved_channel_set}",
        "channel_set": resolved_channel_set,
        "resolved_data_root": resolved_data_root,
        "differencing_period": diff_period,
        "window_size": window_size,
        "stride": standardized_train.stride,
        "retained_columns": list(feature_names),
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "train_record_count": len(processed_train_records),
        "validation_record_count": len(processed_validation_records),
        "test_record_count": len(processed_test_records),
        "train_window_count": standardized_train.window_count,
        "validation_window_count": (
            standardized_validation.window_count if standardized_validation is not None else 0
        ),
        "test_window_count": standardized_test.window_count if standardized_test is not None else 0,
        "standardization_shape": [standardizer.sample_count, standardizer.feature_count],
    }
    if view_manifest_updates:
        view_manifest.update(view_manifest_updates)
    return MoonshotPreparedSplits(
        class_names=class_names,
        feature_names=feature_names,
        retained_columns=feature_names,
        channel_set=resolved_channel_set,
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
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
