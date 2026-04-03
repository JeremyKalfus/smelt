#!/usr/bin/env python3
"""update append-only artifacts for the m02 moonshot run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from smelt.evaluation.diagnostics import (
    build_m02_architecture_summary,
    build_m02_comparison_row,
    export_run_registry_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--figdata-root", type=Path, required=True)
    parser.add_argument("--file-level-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--m02-run-dir", type=Path, required=True)
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
    comparison_row = build_m02_comparison_row(
        baseline_summary_path=args.baseline_summary,
        run_dir=args.m02_run_dir,
    )
    architecture_summary = build_m02_architecture_summary(args.m02_run_dir)

    comparison_csv = args.table_root / "m02_comparison.csv"
    comparison_json = args.table_root / "m02_comparison.json"
    architecture_json = args.table_root / "m02_architecture_summary.json"

    write_dict_rows_csv(comparison_csv, [comparison_row])
    comparison_json.write_text(
        json.dumps(comparison_row, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    architecture_json.write_text(
        json.dumps(architecture_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"run_registry_csv: {registry_paths.run_registry_csv}")
    print(f"run_registry_json: {registry_paths.run_registry_json}")
    print(f"metrics_long_csv: {registry_paths.metrics_long_csv}")
    print(f"training_histories_long_csv: {registry_paths.training_histories_long_csv}")
    print(f"file_level_metrics_long_csv: {registry_paths.file_level_metrics_long_csv}")
    print(f"comparison_csv: {comparison_csv.resolve()}")
    print(f"comparison_json: {comparison_json.resolve()}")
    print(f"architecture_json: {architecture_json.resolve()}")
    return 0


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
