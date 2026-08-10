"""quantify the environmental-channel class barcode and how preprocessing kills it.

the paper's App. Tables 15-16 show per-class environmental statistics with
within-class stds that are tiny relative to between-class gaps (e.g. barometric
pressure), plus a dead-BME680 block of classes sharing identical frozen values.
this probe measures how much class identity the environmental channels leak:

* view "raw_absolute": per-file mean of the raw channel levels (no
  preprocessing). this is what a model would see WITHOUT the benchmark's
  first-row subtraction and differencing.
* view "pipeline_diff": per-file summary stats after the pipeline's first-row
  subtraction and g=25 differencing — what m05/m05b models actually see.

a nearest-centroid classifier on training files, scored on official test files.
chance = 2%.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smelt.datasets import load_base_sensor_dataset
from smelt.evaluation import write_dict_rows_csv, write_json
from smelt.preprocessing import preprocess_split_records

ENV4 = ("Temperature", "Pressure", "Humidity", "Altitude")
GROUPS = {
    "Pressure": ("Pressure",),
    "Altitude": ("Altitude",),
    "Humidity": ("Humidity",),
    "Temperature": ("Temperature",),
    "ENV4": ENV4,
    "ENV4+GasResistance": (*ENV4, "Gas_Resistance"),
}
DIFF_PERIOD = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--emit-csv", type=Path, default=Path("results/tables/env_barcode_probe.csv")
    )
    parser.add_argument(
        "--emit-json", type=Path, default=Path("results/tables/env_barcode_probe.json")
    )
    return parser.parse_args()


def raw_features(arr: np.ndarray, idx: list[int]) -> np.ndarray:
    cols = arr[:, idx]
    return np.concatenate([cols.mean(axis=0), cols.std(axis=0)])


def diff_features(arr: np.ndarray, idx: list[int]) -> np.ndarray:
    feats = []
    for j in idx:
        v = arr[:, j]
        feats.extend([v.std(), np.abs(v).mean(), np.percentile(v, 25), np.percentile(v, 75)])
    return np.asarray(feats, dtype=np.float64)


def nearest_centroid(train_x, train_y, test_x, test_y) -> float:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    train_z = np.nan_to_num((train_x - mean) / std)
    test_z = np.nan_to_num((test_x - mean) / std)
    classes = sorted(set(train_y))
    centroids = np.vstack(
        [train_z[[i for i, y in enumerate(train_y) if y == cls]].mean(axis=0) for cls in classes]
    )
    distances = ((test_z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    predicted = [classes[i] for i in distances.argmin(axis=1)]
    return 100.0 * float(np.mean([p == t for p, t in zip(predicted, test_y, strict=True)]))


def main() -> int:
    args = parse_args()
    dataset = load_base_sensor_dataset(args.data_root)
    raw_cols = list(dataset.raw_column_names)

    diff_train = preprocess_split_records(
        dataset.train_records, dropped_columns=(), diff_period=DIFF_PERIOD
    )
    diff_test = preprocess_split_records(
        dataset.test_records, dropped_columns=(), diff_period=DIFF_PERIOD
    )
    diff_cols = list(diff_train[0].column_names)

    rows = []
    for group_name, channels in GROUPS.items():
        raw_idx = [raw_cols.index(c) for c in channels]
        raw_train = np.vstack(
            [
                raw_features(np.asarray(r.rows, dtype=np.float64), raw_idx)
                for r in dataset.train_records
            ]
        )
        raw_test = np.vstack(
            [
                raw_features(np.asarray(r.rows, dtype=np.float64), raw_idx)
                for r in dataset.test_records
            ]
        )
        train_y = [r.class_name for r in dataset.train_records]
        test_y = [r.class_name for r in dataset.test_records]
        raw_acc = nearest_centroid(raw_train, train_y, raw_test, test_y)

        diff_idx = [diff_cols.index(c) for c in channels]
        d_train = np.vstack(
            [diff_features(np.asarray(r.values, dtype=np.float64), diff_idx) for r in diff_train]
        )
        d_test = np.vstack(
            [diff_features(np.asarray(r.values, dtype=np.float64), diff_idx) for r in diff_test]
        )
        diff_acc = nearest_centroid(d_train, train_y, d_test, test_y)

        rows.append(
            {
                "group": group_name,
                "channel_count": str(len(channels)),
                "raw_absolute_file_acc@1": f"{raw_acc:.1f}",
                "pipeline_diff_file_acc@1": f"{diff_acc:.1f}",
                "suppression_factor": f"{raw_acc / max(diff_acc, 1e-9):.1f}x"
                if diff_acc > 0
                else "inf",
            }
        )

    write_dict_rows_csv(args.emit_csv, rows)
    write_json(
        args.emit_json,
        {
            "method": "nearest-centroid, train files only, scored on official test files",
            "chance_accuracy": 2.0,
            "raw_absolute": "per-file mean+std of raw channel levels (no preprocessing)",
            "pipeline_diff": "per-file stats after first-row subtraction and g=25 differencing",
            "rows": rows,
        },
    )
    width = max(len(r["group"]) for r in rows)
    print(f"{'group'.ljust(width)}  raw_absolute  pipeline_diff")
    for r in rows:
        print(
            f"{r['group'].ljust(width)}  {r['raw_absolute_file_acc@1']:>11}  "
            f"{r['pipeline_diff_file_acc@1']:>12}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
