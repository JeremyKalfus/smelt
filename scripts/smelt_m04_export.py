"""append-only export updates for m04 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from smelt.evaluation import write_dict_rows_csv, write_json
from smelt.evaluation.diagnostics import (
    build_m04_final_comparison_row,
    export_run_registry_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--figdata-root", type=Path, required=True)
    parser.add_argument("--file-level-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--ensemble-run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    table_root = args.table_root.expanduser().resolve()
    figdata_root = args.figdata_root.expanduser().resolve()
    registry = export_run_registry_artifacts(
        run_root=args.run_root.expanduser().resolve(),
        table_root=table_root,
        figdata_root=figdata_root,
        existing_run_ids=(),
        file_level_root=args.file_level_root.expanduser().resolve(),
    )
    comparison_row = build_m04_final_comparison_row(
        baseline_summary_path=args.baseline_summary.expanduser().resolve(),
        ensemble_run_dir=args.ensemble_run_dir.expanduser().resolve(),
    )
    comparison_csv = table_root / "m04_final_comparison.csv"
    comparison_json = table_root / "m04_final_comparison.json"
    write_dict_rows_csv(comparison_csv, [comparison_row])
    write_json(comparison_json, {"rows": [comparison_row]})

    print(f"run_registry_csv: {registry.run_registry_csv}")
    print(f"run_registry_json: {registry.run_registry_json}")
    print(f"metrics_long_csv: {registry.metrics_long_csv}")
    print(f"training_histories_long_csv: {registry.training_histories_long_csv}")
    print(f"file_level_metrics_long_csv: {registry.file_level_metrics_long_csv}")
    print(f"final_comparison_csv: {comparison_csv.resolve()}")
    print(f"final_comparison_json: {comparison_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
