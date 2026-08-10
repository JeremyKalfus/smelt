"""measure how much class signal each raw channel carries on its own.

motivation: the 12-channel setting scores higher than the benchmark 6-channel
setting, and we want to know whether that comes from a genuine extra gas sensor
(BME680 `Gas_Resistance`) or from environmental channels that could be
fingerprinting recording context rather than odor chemistry.

method: reproduce the protocol's preprocessing (per-file baseline subtraction,
then g=25 differencing), reduce each file to per-channel summary statistics, and
fit a nearest-centroid classifier on the official training files only. each
channel (or channel group) is scored on the official test files. chance = 2%.

this is a diagnostic probe, not a competing model. it deliberately uses a weak
classifier so the numbers reflect channel information content rather than
architecture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smelt.datasets import load_base_sensor_dataset
from smelt.evaluation import write_dict_rows_csv, write_json
from smelt.preprocessing import preprocess_split_records

BENCHMARK_RETAINED = ("NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG")
DROPPED = ("Benzene", "Temperature", "Pressure", "Humidity", "Gas_Resistance", "Altitude")
DIFF_PERIOD = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--emit-csv", type=Path, default=Path("results/tables/channel_signal_probe.csv")
    )
    parser.add_argument(
        "--emit-json", type=Path, default=Path("results/tables/channel_signal_probe.json")
    )
    return parser.parse_args()


def file_features(values: np.ndarray) -> np.ndarray:
    """summary stats of one differenced channel within one file."""
    return np.asarray(
        [
            values.std(),
            np.abs(values).mean(),
            np.percentile(values, 10),
            np.percentile(values, 25),
            np.percentile(values, 50),
            np.percentile(values, 75),
            np.percentile(values, 90),
            values.max() - values.min(),
        ],
        dtype=np.float64,
    )


def build_matrix(records, column_names, channels) -> tuple[np.ndarray, list[str]]:
    idx = [column_names.index(c) for c in channels]
    rows, labels = [], []
    for record in records:
        arr = np.asarray(record.values, dtype=np.float64)
        rows.append(np.concatenate([file_features(arr[:, j]) for j in idx]))
        labels.append(record.class_name)
    return np.vstack(rows), labels


def nearest_centroid_accuracy(
    train_x: np.ndarray,
    train_y: list[str],
    test_x: np.ndarray,
    test_y: list[str],
) -> float:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    train_z = (train_x - mean) / std
    test_z = (test_x - mean) / std
    classes = sorted(set(train_y))
    centroids = np.vstack(
        [train_z[[i for i, y in enumerate(train_y) if y == cls]].mean(axis=0) for cls in classes]
    )
    # guard against channels that are entirely flat -> non-finite features
    centroids = np.nan_to_num(centroids)
    test_z = np.nan_to_num(test_z)
    distances = ((test_z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    predicted = [classes[i] for i in distances.argmin(axis=1)]
    return 100.0 * float(np.mean([p == t for p, t in zip(predicted, test_y, strict=True)]))


def main() -> int:
    args = parse_args()
    dataset = load_base_sensor_dataset(args.data_root)

    train = preprocess_split_records(
        dataset.train_records, dropped_columns=(), diff_period=DIFF_PERIOD
    )
    test = preprocess_split_records(
        dataset.test_records, dropped_columns=(), diff_period=DIFF_PERIOD
    )
    kept = list(train[0].column_names)

    groups: dict[str, tuple[str, ...]] = {name: (name,) for name in kept}
    groups["ALL12"] = tuple(kept)
    groups["BENCHMARK6"] = BENCHMARK_RETAINED
    groups["DROPPED6"] = DROPPED
    groups["BENCHMARK6+GasResistance"] = (*BENCHMARK_RETAINED, "Gas_Resistance")
    groups["DROPPED6_minus_GasResistance"] = tuple(c for c in DROPPED if c != "Gas_Resistance")
    groups["ENVIRONMENTAL4"] = ("Temperature", "Pressure", "Humidity", "Altitude")

    rows = []
    for name, channels in groups.items():
        missing = [c for c in channels if c not in kept]
        if missing:
            continue
        train_x, train_y = build_matrix(train, kept, channels)
        test_x, test_y = build_matrix(test, kept, channels)
        acc = nearest_centroid_accuracy(train_x, train_y, test_x, test_y)
        rows.append(
            {
                "group": name,
                "channel_count": str(len(channels)),
                "is_single_channel": str(len(channels) == 1).lower(),
                "benchmark_retained": str(all(c in BENCHMARK_RETAINED for c in channels)).lower(),
                "file_acc@1": f"{acc:.1f}",
                "vs_chance": f"{acc / 2.0:.1f}x",
            }
        )

    rows.sort(key=lambda r: -float(r["file_acc@1"]))
    write_dict_rows_csv(args.emit_csv, rows)
    write_json(
        args.emit_json,
        {
            "method": "nearest-centroid on per-file summary stats of the g=25 differenced signal",
            "train_files": len(train),
            "test_files": len(test),
            "chance_accuracy": 2.0,
            "diff_period": DIFF_PERIOD,
            "rows": rows,
        },
    )
    width = max(len(r["group"]) for r in rows)
    print(f"{'group'.ljust(width)}  n  file_acc@1  vs_chance")
    for r in rows:
        print(
            f"{r['group'].ljust(width)}  {r['channel_count']:>2}  "
            f"{r['file_acc@1']:>9}  {r['vs_chance']:>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
