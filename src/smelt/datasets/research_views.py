"""research-extension sensor views for fit-gcms data paths."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from smelt.preprocessing.base import (
    EXACT_UPSTREAM_DROPPED_COLUMNS,
    PreprocessedSensorRecord,
    PreprocessingError,
    apply_temporal_differencing,
    project_columns,
    resolve_retained_columns,
    sensor_record_to_array,
    subtract_first_row,
)

from .contracts import SensorFileRecord

RAW_ALIGNED_VIEW = "raw_aligned"
DIFF_VIEW = "diff"
FUSED_RAW_DIFF_VIEW = "fused_raw_diff"
RESEARCH_VIEW_MODES = (
    RAW_ALIGNED_VIEW,
    DIFF_VIEW,
    FUSED_RAW_DIFF_VIEW,
)


class ResearchViewError(Exception):
    """raised when a research-extension view cannot be built safely."""


def preprocess_sensor_record_for_view(
    record: SensorFileRecord,
    *,
    view_mode: str,
    dropped_columns: Sequence[str] = EXACT_UPSTREAM_DROPPED_COLUMNS,
    diff_period: int = 0,
) -> PreprocessedSensorRecord:
    resolved_mode = resolve_research_view_mode(view_mode)
    retained_columns, projected_values = build_retained_sensor_values(record, dropped_columns)
    view_values, feature_names = build_view_values(
        projected_values,
        retained_columns=retained_columns,
        view_mode=resolved_mode,
        diff_period=diff_period,
    )
    return PreprocessedSensorRecord(
        split=record.split,
        class_name=record.class_name,
        relative_path=record.relative_path,
        absolute_path=record.absolute_path,
        column_names=feature_names,
        values=view_values,
        source_row_count=record.row_count,
        diff_period=diff_period,
    )


def preprocess_split_records_for_view(
    records: Sequence[SensorFileRecord],
    *,
    view_mode: str,
    dropped_columns: Sequence[str] = EXACT_UPSTREAM_DROPPED_COLUMNS,
    diff_period: int = 0,
) -> tuple[PreprocessedSensorRecord, ...]:
    records_tuple = tuple(records)
    if not records_tuple:
        raise ResearchViewError("cannot preprocess an empty split for a research view")
    split_names = {record.split for record in records_tuple}
    if len(split_names) != 1:
        raise ResearchViewError(
            f"expected one split per research-view preprocessing call, found {sorted(split_names)}"
        )
    return tuple(
        preprocess_sensor_record_for_view(
            record,
            view_mode=view_mode,
            dropped_columns=dropped_columns,
            diff_period=diff_period,
        )
        for record in records_tuple
    )


def build_retained_sensor_values(
    record: SensorFileRecord,
    dropped_columns: Sequence[str] = EXACT_UPSTREAM_DROPPED_COLUMNS,
) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    try:
        raw_values = sensor_record_to_array(record)
        baseline_values = subtract_first_row(raw_values)
        retained_columns = resolve_retained_columns(record.column_names, dropped_columns)
        projected_values = project_columns(baseline_values, record.column_names, retained_columns)
    except PreprocessingError as exc:
        raise ResearchViewError(str(exc)) from exc
    return retained_columns, projected_values


def build_view_values(
    projected_values: NDArray[np.float64],
    *,
    retained_columns: Sequence[str],
    view_mode: str,
    diff_period: int,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    resolved_mode = resolve_research_view_mode(view_mode)
    retained_tuple = tuple(retained_columns)
    if resolved_mode == RAW_ALIGNED_VIEW:
        return align_raw_values(projected_values, diff_period), retained_tuple
    if resolved_mode == DIFF_VIEW:
        return build_diff_values(projected_values, diff_period), retained_tuple
    if diff_period <= 0:
        raise ResearchViewError("fused_raw_diff requires diff_period > 0")
    raw_values = align_raw_values(projected_values, diff_period)
    diff_values = build_diff_values(projected_values, diff_period)
    if raw_values.shape != diff_values.shape:
        raise ResearchViewError(
            "raw_aligned and diff shapes must match before fusion: "
            f"{raw_values.shape} vs {diff_values.shape}"
        )
    return (
        np.concatenate((raw_values, diff_values), axis=1),
        build_fused_feature_names(retained_tuple),
    )


def align_raw_values(
    projected_values: NDArray[np.float64],
    diff_period: int,
) -> NDArray[np.float64]:
    try:
        validate_diff_period(diff_period)
    except PreprocessingError as exc:
        raise ResearchViewError(str(exc)) from exc
    if diff_period == 0:
        return projected_values.copy()
    if diff_period >= projected_values.shape[0]:
        return np.zeros((0, projected_values.shape[1]), dtype=projected_values.dtype)
    return projected_values[diff_period:].copy()


def build_diff_values(
    projected_values: NDArray[np.float64],
    diff_period: int,
) -> NDArray[np.float64]:
    try:
        return apply_temporal_differencing(projected_values, diff_period)
    except PreprocessingError as exc:
        raise ResearchViewError(str(exc)) from exc


def build_fused_feature_names(retained_columns: Sequence[str]) -> tuple[str, ...]:
    retained_tuple = tuple(retained_columns)
    return tuple(f"raw_{column_name}" for column_name in retained_tuple) + tuple(
        f"diff_{column_name}" for column_name in retained_tuple
    )


def resolve_research_view_mode(view_mode: str) -> str:
    if view_mode not in RESEARCH_VIEW_MODES:
        raise ResearchViewError(
            f"unsupported research view mode: {view_mode!r}; expected one of {RESEARCH_VIEW_MODES}"
        )
    return view_mode


def validate_diff_period(diff_period: int) -> None:
    if diff_period < 0:
        raise ResearchViewError(f"diff period must be non-negative, found {diff_period}")
