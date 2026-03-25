"""load exact-upstream base sensor recordings."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

from .contracts import (
    BENCHMARK_SENSOR_COLUMNS,
    RAW_SENSOR_COLUMNS,
    SPLIT_NAMES,
    TEST_SPLIT,
    TRAIN_SPLIT,
    BaseSensorDataset,
    SensorFileRecord,
)


class SensorLoaderError(Exception):
    """raised when base sensor files cannot be loaded safely."""


class SensorSchemaError(SensorLoaderError):
    """raised when a sensor csv drifts from the expected schema."""


def load_base_sensor_dataset(data_root: Path) -> BaseSensorDataset:
    resolved_root = data_root.expanduser().resolve()
    split_records = {
        TRAIN_SPLIT: tuple(load_split_records(resolved_root, TRAIN_SPLIT)),
        TEST_SPLIT: tuple(load_split_records(resolved_root, TEST_SPLIT)),
    }
    validate_split_class_vocab(
        split_records[TRAIN_SPLIT],
        split_records[TEST_SPLIT],
    )
    return BaseSensorDataset(
        resolved_data_root=str(resolved_root),
        raw_column_names=RAW_SENSOR_COLUMNS,
        train_records=split_records[TRAIN_SPLIT],
        test_records=split_records[TEST_SPLIT],
    )


def load_split_records(data_root: Path, split_name: str) -> list[SensorFileRecord]:
    split_dir = validate_split_dir(data_root, split_name)
    records: list[SensorFileRecord] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        csv_paths = sorted(path for path in class_dir.glob("*.csv") if path.is_file())
        if not csv_paths:
            raise SensorLoaderError(f"class directory has no csv files: {class_dir}")
        for csv_path in csv_paths:
            records.append(load_sensor_file(csv_path, data_root, split_name, class_dir.name))
    return records


def load_sensor_file(
    csv_path: Path,
    data_root: Path,
    split_name: str,
    class_name: str,
) -> SensorFileRecord:
    expected_columns = RAW_SENSOR_COLUMNS
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = tuple(next(reader))
            except StopIteration as exc:
                raise SensorSchemaError(f"csv file is empty: {csv_path}") from exc
            validate_column_names(header, expected_columns, csv_path)

            rows: list[tuple[float, ...]] = []
            for row_index, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise SensorSchemaError(
                        f"inconsistent row width in {csv_path}: "
                        f"row {row_index} has {len(row)} columns, expected {len(header)}"
                    )
                rows.append(parse_numeric_row(row, csv_path, row_index))
            if not rows:
                raise SensorSchemaError(f"csv file has no data rows: {csv_path}")
    except OSError as exc:
        raise SensorLoaderError(f"unable to read csv file {csv_path}: {exc}") from exc
    except UnicodeError as exc:
        raise SensorLoaderError(f"unable to decode csv file {csv_path}: {exc}") from exc
    except csv.Error as exc:
        raise SensorLoaderError(f"unable to parse csv file {csv_path}: {exc}") from exc

    return SensorFileRecord(
        split=split_name,
        class_name=class_name,
        relative_path=str(csv_path.relative_to(data_root)),
        absolute_path=str(csv_path.resolve()),
        column_names=header,
        rows=tuple(rows),
    )


def select_sensor_columns(
    record: SensorFileRecord,
    column_names: Sequence[str],
) -> SensorFileRecord:
    validate_requested_columns(record.column_names, column_names)
    column_index = {column_name: index for index, column_name in enumerate(record.column_names)}
    projected_rows = tuple(
        tuple(row[column_index[column_name]] for column_name in column_names) for row in record.rows
    )
    return SensorFileRecord(
        split=record.split,
        class_name=record.class_name,
        relative_path=record.relative_path,
        absolute_path=record.absolute_path,
        column_names=tuple(column_names),
        rows=projected_rows,
    )


def select_benchmark_sensor_columns(record: SensorFileRecord) -> SensorFileRecord:
    return select_sensor_columns(record, BENCHMARK_SENSOR_COLUMNS)


def validate_split_dir(data_root: Path, split_name: str) -> Path:
    if split_name not in SPLIT_NAMES:
        raise SensorLoaderError(f"unknown split: {split_name}")
    split_dir = data_root / split_name
    if not split_dir.exists():
        raise SensorLoaderError(f"required split directory is missing: {split_dir}")
    if not split_dir.is_dir():
        raise SensorLoaderError(f"split path is not a directory: {split_dir}")
    class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise SensorLoaderError(f"split directory has no class subdirectories: {split_dir}")
    return split_dir


def validate_column_names(
    observed_columns: Sequence[str],
    expected_columns: Sequence[str],
    csv_path: Path,
) -> None:
    if tuple(observed_columns) != tuple(expected_columns):
        raise SensorSchemaError(
            f"unexpected column schema in {csv_path}: "
            f"expected {list(expected_columns)}, found {list(observed_columns)}"
        )


def validate_requested_columns(
    available_columns: Sequence[str],
    requested_columns: Sequence[str],
) -> None:
    missing_columns = [
        column_name for column_name in requested_columns if column_name not in available_columns
    ]
    if missing_columns:
        raise SensorSchemaError(
            f"requested columns are missing from the sensor record: {missing_columns}"
        )


def validate_split_class_vocab(
    train_records: Sequence[SensorFileRecord],
    test_records: Sequence[SensorFileRecord],
) -> None:
    train_classes = {record.class_name for record in train_records}
    test_classes = {record.class_name for record in test_records}
    if train_classes != test_classes:
        missing_in_test = sorted(train_classes - test_classes)
        missing_in_train = sorted(test_classes - train_classes)
        raise SensorLoaderError(
            "class vocab mismatch across train/test: "
            f"missing_in_test={missing_in_test}, missing_in_train={missing_in_train}"
        )


def parse_numeric_row(
    row: Iterable[str],
    csv_path: Path,
    row_index: int,
) -> tuple[float, ...]:
    values: list[float] = []
    for column_index, value in enumerate(row, start=1):
        try:
            values.append(float(value))
        except ValueError as exc:
            raise SensorSchemaError(
                f"non-numeric value in {csv_path}: "
                f"row {row_index}, column {column_index}, value={value!r}"
            ) from exc
    return tuple(values)
