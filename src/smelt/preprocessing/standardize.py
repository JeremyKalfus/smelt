"""train-only standardization for exact-upstream window tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .windows import WindowedSplit, WindowingError, rewrap_windows, stack_window_values

TRAIN_SPLIT = "offline_training"


class StandardizationError(Exception):
    """raised when train-only window standardization cannot be applied safely."""


@dataclass(slots=True)
class StandardizationStats:
    fitted_split: str
    window_count: int
    sample_count: int
    window_size: int
    feature_count: int
    column_names: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    scale: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "fitted_split": self.fitted_split,
            "window_count": self.window_count,
            "sample_count": self.sample_count,
            "window_size": self.window_size,
            "feature_count": self.feature_count,
            "column_names": list(self.column_names),
            "mean": list(self.mean),
            "std": list(self.std),
            "scale": list(self.scale),
        }


def fit_window_standardizer(
    windowed_split: WindowedSplit,
    *,
    expected_fit_split: str = TRAIN_SPLIT,
) -> StandardizationStats:
    if windowed_split.split != expected_fit_split:
        raise StandardizationError(
            "standardization stats must be fit on the training split only: "
            f"expected {expected_fit_split}, found {windowed_split.split}"
        )
    if not windowed_split.windows:
        raise StandardizationError("cannot fit standardization stats on an empty training split")

    try:
        values = stack_window_values(windowed_split.windows)
    except WindowingError as exc:
        raise StandardizationError(str(exc)) from exc

    sample_count = int(values.shape[0] * values.shape[1])
    flat_values = values.reshape(sample_count, values.shape[2])
    mean = flat_values.mean(axis=0)
    std = flat_values.std(axis=0, ddof=0)
    scale = np.where(std == 0.0, 1.0, std)

    return StandardizationStats(
        fitted_split=windowed_split.split,
        window_count=windowed_split.window_count,
        sample_count=sample_count,
        window_size=windowed_split.window_size,
        feature_count=int(values.shape[2]),
        column_names=windowed_split.column_names,
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
        scale=tuple(float(value) for value in scale),
    )


def apply_window_standardizer(
    windowed_split: WindowedSplit,
    stats: StandardizationStats,
) -> WindowedSplit:
    if windowed_split.column_names != stats.column_names:
        raise StandardizationError(
            f"window columns {windowed_split.column_names} do not match {stats.column_names}"
        )
    if windowed_split.window_count == 0:
        return windowed_split

    try:
        values = stack_window_values(windowed_split.windows)
    except WindowingError as exc:
        raise StandardizationError(str(exc)) from exc

    if values.shape[2] != stats.feature_count:
        raise StandardizationError(
            f"window feature count {values.shape[2]} does not match {stats.feature_count}"
        )

    mean = np.asarray(stats.mean, dtype=np.float64)
    scale = np.asarray(stats.scale, dtype=np.float64)
    standardized = ((values - mean) / scale).astype(np.float32, copy=False)

    try:
        return rewrap_windows(windowed_split, standardized)
    except WindowingError as exc:
        raise StandardizationError(str(exc)) from exc
