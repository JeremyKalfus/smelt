"""audit per-channel data quality on the raw 12-column SMELLNET-BASE snapshot.

This is a data-quality diagnostic, not a modeling artifact. It exists to
document the behavior of the six channels dropped by the upstream benchmark
(`Benzene`, `Temperature`, `Pressure`, `Humidity`, `Gas_Resistance`,
`Altitude`), including the uint32 overflow sentinel that appears in the
`Benzene` channel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smelt.datasets import load_base_sensor_dataset
from smelt.evaluation import write_dict_rows_csv, write_json
from smelt.preprocessing import EXACT_UPSTREAM_DROPPED_COLUMNS

UINT32_SENTINEL = 4294967295.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--emit-csv",
        type=Path,
        default=Path("results/tables/channel_quality_audit.csv"),
    )
    parser.add_argument(
        "--emit-json",
        type=Path,
        default=Path("results/tables/channel_quality_audit.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_base_sensor_dataset(args.data_root)
    column_names = dataset.raw_column_names
    dropped = set(EXACT_UPSTREAM_DROPPED_COLUMNS)

    rows: list[dict[str, str]] = []
    for split_name, records in (
        ("offline_training", dataset.train_records),
        ("offline_testing", dataset.test_records),
    ):
        arrays = [np.asarray(record.rows, dtype=np.float64) for record in records]
        for channel_index, channel_name in enumerate(column_names):
            channel_values = [array[:, channel_index] for array in arrays]
            flat = np.concatenate(channel_values)
            sentinel_mask = flat == UINT32_SENTINEL
            clean = flat[~sentinel_mask]
            per_file_std = np.asarray([values.std() for values in channel_values])
            per_file_sentinel = np.asarray(
                [bool(np.any(values == UINT32_SENTINEL)) for values in channel_values]
            )
            rows.append(
                {
                    "split": split_name,
                    "channel": channel_name,
                    "benchmark_retained": str(channel_name not in dropped).lower(),
                    "file_count": str(len(records)),
                    "value_count": str(int(flat.size)),
                    "sentinel_value_fraction": f"{float(sentinel_mask.mean()):.6f}",
                    "files_with_sentinel": str(int(per_file_sentinel.sum())),
                    "files_flatline": str(int((per_file_std == 0.0).sum())),
                    "zero_value_fraction": f"{float((flat == 0.0).mean()):.6f}",
                    "min_excl_sentinel": f"{float(clean.min()):.4f}" if clean.size else "",
                    "max_excl_sentinel": f"{float(clean.max()):.4f}" if clean.size else "",
                    "mean_excl_sentinel": f"{float(clean.mean()):.4f}" if clean.size else "",
                    "std_excl_sentinel": f"{float(clean.std()):.4f}" if clean.size else "",
                }
            )

    payload = {
        "resolved_data_root": dataset.resolved_data_root,
        "sentinel_value": UINT32_SENTINEL,
        "sentinel_semantics": "uint32 overflow / sensor fault marker observed in raw channels",
        "benchmark_dropped_columns": sorted(dropped),
        "rows": rows,
    }
    write_dict_rows_csv(args.emit_csv, rows)
    write_json(args.emit_json, payload)
    for row in rows:
        print(
            f"{row['split']:<17} {row['channel']:<15} "
            f"retained={row['benchmark_retained']:<5} "
            f"sentinel%={100 * float(row['sentinel_value_fraction']):>7.3f} "
            f"flatline_files={row['files_flatline']:>3}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
