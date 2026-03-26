from __future__ import annotations

from pathlib import Path

import pytest

from smelt.datasets import (
    BENCHMARK_SENSOR_COLUMNS,
    RAW_SENSOR_COLUMNS,
    SensorDataError,
    SensorLoaderError,
    SensorSchemaError,
    load_base_sensor_dataset,
    load_split_records,
    select_benchmark_sensor_columns,
    select_sensor_columns,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def test_load_base_sensor_dataset_preserves_identity_and_order() -> None:
    dataset = load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_valid")

    assert dataset.resolved_data_root == str((FIXTURE_ROOT / "smellnet_base_valid").resolve())
    assert dataset.raw_column_names == RAW_SENSOR_COLUMNS
    assert dataset.class_vocab == tuple(f"odor_{index:02d}" for index in range(50))
    assert len(dataset.train_records) == 250
    assert len(dataset.test_records) == 50

    first_train = dataset.train_records[0]
    last_test = dataset.test_records[-1]
    assert first_train.relative_path == "offline_training/odor_00/train_00.csv"
    assert first_train.split == "offline_training"
    assert first_train.class_name == "odor_00"
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
    assert last_test.relative_path == "offline_testing/odor_49/test_00.csv"
    assert last_test.split == "offline_testing"
    assert last_test.class_name == "odor_49"


def test_select_benchmark_sensor_columns_is_opt_in() -> None:
    dataset = load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_valid")
    raw_record = dataset.train_records[0]

    benchmark_record = select_benchmark_sensor_columns(raw_record)

    assert raw_record.column_names == RAW_SENSOR_COLUMNS
    assert benchmark_record.column_names == BENCHMARK_SENSOR_COLUMNS
    assert benchmark_record.rows[0] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


def test_select_sensor_columns_rejects_duplicate_requests() -> None:
    dataset = load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_valid")

    with pytest.raises(SensorSchemaError, match="requested columns contain duplicates"):
        select_sensor_columns(dataset.train_records[0], ("NO2", "NO2"))


def test_loader_fails_on_schema_drift() -> None:
    with pytest.raises(SensorSchemaError, match="unexpected column schema"):
        load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_invalid_schema")


def test_loader_fails_on_mismatched_class_vocab() -> None:
    with pytest.raises(SensorLoaderError, match="class vocab mismatch across train/test"):
        load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_invalid_vocab_mismatch")


def test_loader_fails_on_non_numeric_row(tmp_path: Path) -> None:
    data_root = tmp_path / "smellnet_base_invalid_non_numeric"
    train_dir = data_root / "offline_training" / "odor_00"
    test_dir = data_root / "offline_testing" / "odor_00"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    header = ",".join(RAW_SENSOR_COLUMNS)
    train_dir.joinpath("train_00.csv").write_text(
        header + "\n" + "1,2,3,4,5,6,7,8,9,10,11,12\n" + "oops,2,3,4,5,6,7,8,9,10,11,12\n",
        encoding="utf-8",
    )
    test_dir.joinpath("test_00.csv").write_text(
        header + "\n" + "1,2,3,4,5,6,7,8,9,10,11,12\n",
        encoding="utf-8",
    )

    with pytest.raises(SensorDataError, match="non-numeric value"):
        load_base_sensor_dataset(data_root)


def test_load_split_records_rejects_unknown_split() -> None:
    with pytest.raises(SensorLoaderError, match="unknown split"):
        load_split_records(FIXTURE_ROOT / "smellnet_base_valid", "unknown")
