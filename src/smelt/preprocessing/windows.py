"""sliding-window generation for exact-upstream base preprocessing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .base import PreprocessedSensorRecord


class WindowingError(Exception):
    """raised when exact-upstream sliding windows cannot be generated safely."""


@dataclass(slots=True)
class SkippedSensorRecord:
    split: str
    class_name: str
    relative_path: str
    absolute_path: str
    row_count: int
    window_size: int


@dataclass(slots=True)
class SensorWindow:
    split: str
    class_name: str
    relative_path: str
    absolute_path: str
    column_names: tuple[str, ...]
    window_index: int
    start_row: int
    stop_row: int
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise WindowingError(f"window values must be 2d, found shape {self.values.shape}")
        if self.values.shape[1] != len(self.column_names):
            raise WindowingError(
                "window channel count does not match declared column names: "
                f"{self.values.shape[1]} vs {len(self.column_names)}"
            )

    @property
    def window_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.values.shape[1])


@dataclass(slots=True)
class WindowedSplit:
    split: str
    column_names: tuple[str, ...]
    window_size: int
    stride: int
    source_record_count: int
    windows: tuple[SensorWindow, ...]
    skipped_records: tuple[SkippedSensorRecord, ...]

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise WindowingError(f"window size must be positive, found {self.window_size}")
        if self.stride <= 0:
            raise WindowingError(f"stride must be positive, found {self.stride}")
        for window in self.windows:
            if window.split != self.split:
                raise WindowingError(
                    f"window split {window.split} does not match split container {self.split}"
                )
            if window.column_names != self.column_names:
                raise WindowingError(
                    f"window columns {window.column_names} do not match {self.column_names}"
                )
            if window.window_size != self.window_size:
                raise WindowingError(
                    f"window size {window.window_size} does not match {self.window_size}"
                )

    @property
    def window_count(self) -> int:
        return len(self.windows)


def resolve_window_stride(window_size: int, stride: int | None) -> int:
    if window_size <= 0:
        raise WindowingError(f"window size must be positive, found {window_size}")
    if stride is None:
        return max(1, window_size // 2)
    if stride <= 0:
        raise WindowingError(f"stride must be positive, found {stride}")
    return stride


def generate_record_windows(
    record: PreprocessedSensorRecord,
    *,
    window_size: int,
    stride: int | None = None,
) -> tuple[SensorWindow, ...]:
    validate_record_window_shape(record)
    resolved_stride = resolve_window_stride(window_size, stride)
    if record.row_count < window_size:
        return ()

    windows: list[SensorWindow] = []
    for window_index, start_row in enumerate(
        range(0, record.row_count - window_size + 1, resolved_stride)
    ):
        stop_row = start_row + window_size
        windows.append(
            SensorWindow(
                split=record.split,
                class_name=record.class_name,
                relative_path=record.relative_path,
                absolute_path=record.absolute_path,
                column_names=record.column_names,
                window_index=window_index,
                start_row=start_row,
                stop_row=stop_row,
                values=record.values[start_row:stop_row].copy(),
            )
        )
    return tuple(windows)


def generate_split_windows(
    records: Sequence[PreprocessedSensorRecord],
    *,
    window_size: int,
    stride: int | None = None,
) -> WindowedSplit:
    records_tuple = tuple(records)
    if not records_tuple:
        raise WindowingError("cannot generate windows from an empty split")

    split_names = {record.split for record in records_tuple}
    if len(split_names) != 1:
        raise WindowingError(f"expected one split per windowing call, found {sorted(split_names)}")

    column_name_sets = {record.column_names for record in records_tuple}
    if len(column_name_sets) != 1:
        raise WindowingError(
            f"expected one channel schema per windowing call, found {column_name_sets}"
        )

    resolved_stride = resolve_window_stride(window_size, stride)
    windows: list[SensorWindow] = []
    skipped_records: list[SkippedSensorRecord] = []
    for record in records_tuple:
        if record.row_count < window_size:
            skipped_records.append(
                SkippedSensorRecord(
                    split=record.split,
                    class_name=record.class_name,
                    relative_path=record.relative_path,
                    absolute_path=record.absolute_path,
                    row_count=record.row_count,
                    window_size=window_size,
                )
            )
            continue
        windows.extend(
            generate_record_windows(
                record,
                window_size=window_size,
                stride=resolved_stride,
            )
        )

    split_name = next(iter(split_names))
    column_names = next(iter(column_name_sets))
    return WindowedSplit(
        split=split_name,
        column_names=column_names,
        window_size=window_size,
        stride=resolved_stride,
        source_record_count=len(records_tuple),
        windows=tuple(windows),
        skipped_records=tuple(skipped_records),
    )


def validate_record_window_shape(record: PreprocessedSensorRecord) -> None:
    if record.values.ndim != 2:
        raise WindowingError(
            f"preprocessed record {record.relative_path} must be 2d, found {record.values.shape}"
        )
    if record.channel_count == 0:
        raise WindowingError(f"preprocessed record {record.relative_path} has no channels")
    if record.row_count < 0:
        raise WindowingError(f"preprocessed record {record.relative_path} has a negative row count")


def stack_window_values(windows: Sequence[SensorWindow]) -> NDArray[np.float64]:
    windows_tuple = tuple(windows)
    if not windows_tuple:
        raise WindowingError("cannot stack an empty window collection")
    first_shape = windows_tuple[0].values.shape
    for window in windows_tuple:
        if window.values.shape != first_shape:
            raise WindowingError(
                "window shapes must match for stacking, found "
                f"{window.values.shape} vs {first_shape}"
            )
    return np.stack([window.values for window in windows_tuple], axis=0)


def rewrap_windows(
    original_split: WindowedSplit,
    values: NDArray[np.float64],
) -> WindowedSplit:
    if values.ndim != 3:
        raise WindowingError(f"standardized window array must be 3d, found shape {values.shape}")
    if values.shape[0] != len(original_split.windows):
        raise WindowingError(
            "standardized window count does not match the source split: "
            f"{values.shape[0]} vs {len(original_split.windows)}"
        )
    rebuilt_windows = []
    for index, window in enumerate(original_split.windows):
        rebuilt_windows.append(
            SensorWindow(
                split=window.split,
                class_name=window.class_name,
                relative_path=window.relative_path,
                absolute_path=window.absolute_path,
                column_names=window.column_names,
                window_index=window.window_index,
                start_row=window.start_row,
                stop_row=window.stop_row,
                values=values[index],
            )
        )
    return WindowedSplit(
        split=original_split.split,
        column_names=original_split.column_names,
        window_size=original_split.window_size,
        stride=original_split.stride,
        source_record_count=original_split.source_record_count,
        windows=tuple(rebuilt_windows),
        skipped_records=original_split.skipped_records,
    )
