"""exact-upstream preprocessing for base sensor recordings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from smelt.datasets.contracts import SensorFileRecord

EXACT_UPSTREAM_DROPPED_COLUMNS = (
    "Benzene",
    "Temperature",
    "Pressure",
    "Humidity",
    "Gas_Resistance",
    "Altitude",
)


class PreprocessingError(Exception):
    """raised when exact-upstream preprocessing cannot be applied safely."""


@dataclass(slots=True)
class PreprocessedSensorRecord:
    split: str
    class_name: str
    relative_path: str
    absolute_path: str
    column_names: tuple[str, ...]
    values: NDArray[np.float64]
    source_row_count: int
    diff_period: int
    baseline_subtracted: bool = True

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise PreprocessingError(
                f"preprocessed sensor values must be 2d, found shape {self.values.shape}"
            )
        if self.values.shape[1] != len(self.column_names):
            raise PreprocessingError(
                "preprocessed channel count does not match declared column names: "
                f"{self.values.shape[1]} vs {len(self.column_names)}"
            )

    @property
    def row_count(self) -> int:
        return int(self.values.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.values.shape[1])


def sensor_record_to_array(record: SensorFileRecord) -> NDArray[np.float64]:
    values = np.asarray(record.rows, dtype=np.float64)
    if values.ndim != 2:
        raise PreprocessingError(
            f"sensor record {record.relative_path} must be 2d, found shape {values.shape}"
        )
    if values.shape != (record.row_count, record.column_count):
        raise PreprocessingError(
            "sensor record shape does not match declared metadata for "
            f"{record.relative_path}: {values.shape} vs ({record.row_count}, {record.column_count})"
        )
    return values


def subtract_first_row(values: NDArray[np.float64]) -> NDArray[np.float64]:
    validate_sensor_matrix(values, context="baseline subtraction")
    return values - values[[0], :]


def resolve_retained_columns(
    observed_columns: Sequence[str],
    dropped_columns: Sequence[str] = EXACT_UPSTREAM_DROPPED_COLUMNS,
) -> tuple[str, ...]:
    observed_tuple = tuple(observed_columns)
    dropped_tuple = tuple(dropped_columns)
    if len(set(dropped_tuple)) != len(dropped_tuple):
        raise PreprocessingError(f"dropped columns contain duplicates: {list(dropped_tuple)}")
    missing_columns = [
        column_name for column_name in dropped_tuple if column_name not in observed_tuple
    ]
    if missing_columns:
        raise PreprocessingError(
            f"dropped columns are missing from the observed schema: {missing_columns}"
        )
    retained_columns = tuple(
        column_name for column_name in observed_tuple if column_name not in set(dropped_tuple)
    )
    if not retained_columns:
        raise PreprocessingError("no sensor columns remain after column dropping")
    return retained_columns


def project_columns(
    values: NDArray[np.float64],
    observed_columns: Sequence[str],
    retained_columns: Sequence[str],
) -> NDArray[np.float64]:
    validate_sensor_matrix(values, context="column projection")
    retained_tuple = tuple(retained_columns)
    if len(set(retained_tuple)) != len(retained_tuple):
        raise PreprocessingError(f"retained columns contain duplicates: {list(retained_tuple)}")
    missing_columns = [
        column_name for column_name in retained_tuple if column_name not in tuple(observed_columns)
    ]
    if missing_columns:
        raise PreprocessingError(
            f"retained columns are missing from the observed schema: {missing_columns}"
        )
    column_index = {column_name: index for index, column_name in enumerate(observed_columns)}
    indices = [column_index[column_name] for column_name in retained_tuple]
    return values[:, indices]


def apply_temporal_differencing(
    values: NDArray[np.float64],
    diff_period: int,
) -> NDArray[np.float64]:
    validate_sensor_matrix(values, context="temporal differencing")
    if diff_period < 0:
        raise PreprocessingError(f"diff period must be non-negative, found {diff_period}")
    if diff_period == 0:
        return values.copy()
    if diff_period >= values.shape[0]:
        return np.zeros((0, values.shape[1]), dtype=values.dtype)
    return values[diff_period:] - values[:-diff_period]


def preprocess_sensor_record(
    record: SensorFileRecord,
    *,
    dropped_columns: Sequence[str] = EXACT_UPSTREAM_DROPPED_COLUMNS,
    diff_period: int = 0,
) -> PreprocessedSensorRecord:
    raw_values = sensor_record_to_array(record)
    baseline_values = subtract_first_row(raw_values)
    retained_columns = resolve_retained_columns(record.column_names, dropped_columns)
    projected_values = project_columns(baseline_values, record.column_names, retained_columns)
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


def preprocess_split_records(
    records: Sequence[SensorFileRecord],
    *,
    dropped_columns: Sequence[str] = EXACT_UPSTREAM_DROPPED_COLUMNS,
    diff_period: int = 0,
) -> tuple[PreprocessedSensorRecord, ...]:
    records_tuple = tuple(records)
    if not records_tuple:
        raise PreprocessingError("cannot preprocess an empty split")
    split_names = {record.split for record in records_tuple}
    if len(split_names) != 1:
        raise PreprocessingError(
            f"expected one split per preprocessing call, found {sorted(split_names)}"
        )
    return tuple(
        preprocess_sensor_record(
            record,
            dropped_columns=dropped_columns,
            diff_period=diff_period,
        )
        for record in records_tuple
    )


def validate_sensor_matrix(values: NDArray[np.float64], *, context: str) -> None:
    if values.ndim != 2:
        raise PreprocessingError(
            f"{context} expects a 2d sensor matrix, found shape {values.shape}"
        )
    if values.shape[0] == 0:
        raise PreprocessingError(f"{context} expects at least one row")
    if values.shape[1] == 0:
        raise PreprocessingError(f"{context} expects at least one channel")
