from __future__ import annotations

from pathlib import Path

import pytest

from smelt.datasets.base_loader import (
    SensorLoaderError,
    SensorSchemaError,
    load_base_sensor_dataset,
    load_sensor_file,
    select_benchmark_sensor_columns,
    select_sensor_columns,
)
from smelt.datasets.contracts import BENCHMARK_SENSOR_COLUMNS, RAW_SENSOR_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures"
EXPECTED_CLASSES = tuple(f"odor_{index:02d}" for index in range(50))


def test_load_base_sensor_dataset_preserves_identity_and_order() -> None:
    dataset = load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_valid")

    assert dataset.resolved_data_root == str((FIXTURE_ROOT / "smellnet_base_valid").resolve())
    assert dataset.raw_column_names == RAW_SENSOR_COLUMNS
    assert dataset.class_vocab == EXPECTED_CLASSES
    assert len(dataset.train_records) == 250
    assert len(dataset.test_records) == 50

    first_train = dataset.train_records[0]
    assert first_train.split == "offline_training"
    assert first_train.class_name == "odor_00"
    assert first_train.relative_path == "offline_training/odor_00/train_00.csv"
    assert first_train.column_names == RAW_SENSOR_COLUMNS
    assert first_train.row_count == 3
    assert first_train.column_count == 12
    assert first_train.rows[0] == (
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    )

    first_test = dataset.test_records[0]
    assert first_test.split == "offline_testing"
    assert first_test.class_name == "odor_00"
    assert first_test.relative_path == "offline_testing/odor_00/test_00.csv"


def test_select_benchmark_sensor_columns_preserves_identity() -> None:
    dataset = load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_valid")

    projected = select_benchmark_sensor_columns(dataset.train_records[0])
    assert projected.split == dataset.train_records[0].split
    assert projected.class_name == dataset.train_records[0].class_name
    assert projected.relative_path == dataset.train_records[0].relative_path
    assert projected.column_names == BENCHMARK_SENSOR_COLUMNS
    assert projected.row_count == dataset.train_records[0].row_count
    assert projected.rows[0] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_select_sensor_columns_rejects_missing_column() -> None:
    dataset = load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_valid")

    with pytest.raises(SensorSchemaError, match="requested columns are missing"):
        select_sensor_columns(dataset.train_records[0], ("NO2", "not_a_column"))


def test_load_base_sensor_dataset_rejects_missing_split() -> None:
    with pytest.raises(SensorLoaderError, match="required split directory is missing"):
        load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_invalid_missing_split")


def test_load_base_sensor_dataset_rejects_schema_drift() -> None:
    with pytest.raises(SensorSchemaError, match="unexpected column schema"):
        load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_invalid_schema")


def test_load_base_sensor_dataset_rejects_vocab_mismatch() -> None:
    with pytest.raises(SensorLoaderError, match="class vocab mismatch across train/test"):
        load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_invalid_vocab_mismatch")


def test_load_sensor_file_rejects_non_numeric_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        ",".join(RAW_SENSOR_COLUMNS)
        + "\n"
        + "1,2,3,4,5,6,7,8,9,10,11,12\n"
        + "1,2,3,4,5,6,oops,8,9,10,11,12\n",
        encoding="utf-8",
    )

    with pytest.raises(SensorSchemaError, match="non-numeric value"):
        load_sensor_file(csv_path, tmp_path, "offline_training", "tmp_class")
