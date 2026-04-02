#!/usr/bin/env python3
"""update append-only locked moonshot protocol summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from smelt.evaluation.diagnostics import (
    build_moonshot_locked_run_row,
    build_moonshot_protocol_definition,
    build_moonshot_seed_summary,
    export_run_registry_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--figdata-root", type=Path, required=True)
    parser.add_argument("--file-level-root", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--locked-run-dir", type=Path, action="append", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    registry_paths = export_run_registry_artifacts(
        run_root=args.run_root,
        table_root=args.table_root,
        figdata_root=args.figdata_root,
        existing_run_ids=tuple(),
        file_level_root=args.file_level_root,
    )

    protocol_definition = build_moonshot_protocol_definition(args.protocol_config)
    locked_rows = [build_moonshot_locked_run_row(run_dir) for run_dir in args.locked_run_dir]
    seed_summary = build_moonshot_seed_summary(locked_rows)

    protocol_json = args.table_root / "m01c_protocol_definition.json"
    locked_csv = args.table_root / "m01c_locked_protocol_runs.csv"
    locked_json = args.table_root / "m01c_locked_protocol_runs.json"
    summary_csv = args.table_root / "m01c_seed_summary.csv"
    summary_json = args.table_root / "m01c_seed_summary.json"

    protocol_json.write_text(
        json.dumps(protocol_definition, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dict_rows_csv(locked_csv, locked_rows)
    locked_json.write_text(
        json.dumps({"rows": locked_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dict_rows_csv(summary_csv, flatten_seed_summary(seed_summary))
    summary_json.write_text(
        json.dumps(seed_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"run_registry_csv: {registry_paths.run_registry_csv}")
    print(f"run_registry_json: {registry_paths.run_registry_json}")
    print(f"metrics_long_csv: {registry_paths.metrics_long_csv}")
    print(f"training_histories_long_csv: {registry_paths.training_histories_long_csv}")
    print(f"file_level_metrics_long_csv: {registry_paths.file_level_metrics_long_csv}")
    print(f"protocol_json: {protocol_json.resolve()}")
    print(f"locked_csv: {locked_csv.resolve()}")
    print(f"locked_json: {locked_json.resolve()}")
    print(f"summary_csv: {summary_csv.resolve()}")
    print(f"summary_json: {summary_json.resolve()}")
    return 0


def flatten_seed_summary(seed_summary: dict[str, object]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, value in seed_summary.items():
        if not isinstance(value, dict):
            continue
        rows.append(
            {
                "metric_name": key,
                "mean": str(value.get("mean", "")),
                "std": str(value.get("std", "")),
                "min": str(value.get("min", "")),
                "max": str(value.get("max", "")),
            }
        )
    return rows


def write_dict_rows_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
