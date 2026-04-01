"""update append-only registry and moonshot m01 comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from smelt.evaluation.diagnostics import export_run_registry_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--figdata-root", type=Path, required=True)
    parser.add_argument("--file-level-root", type=Path, required=True)
    parser.add_argument("--existing-comparison-json", type=Path, required=True)
    parser.add_argument("--moonshot-run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    existing_payload = load_json_file(args.existing_comparison_json)
    existing_run_ids = tuple(sorted({row["run_id"] for row in existing_payload["rows"]}))
    registry_paths = export_run_registry_artifacts(
        run_root=args.run_root,
        table_root=args.table_root,
        figdata_root=args.figdata_root,
        existing_run_ids=existing_run_ids,
        file_level_root=args.file_level_root,
    )
    moonshot_rows = load_json_file(args.moonshot_run_dir / "file_level_metrics_comparison.json")[
        "rows"
    ]
    combined_rows = list(existing_payload["rows"]) + list(moonshot_rows)
    comparison_csv = args.table_root / "moonshot_m01_comparison.csv"
    comparison_json = args.table_root / "moonshot_m01_comparison.json"
    best_existing = max(existing_payload["rows"], key=lambda row: float(row["file_acc@1"]))
    best_moonshot = max(moonshot_rows, key=lambda row: float(row["file_acc@1"]))
    write_dict_rows_csv(comparison_csv, combined_rows)
    write_json(
        comparison_json,
        {
            "rows": combined_rows,
            "best_existing": best_existing,
            "best_moonshot": best_moonshot,
        },
    )
    print(f"run_registry_csv: {registry_paths.run_registry_csv}")
    print(f"run_registry_json: {registry_paths.run_registry_json}")
    print(f"metrics_long_csv: {registry_paths.metrics_long_csv}")
    print(f"training_histories_long_csv: {registry_paths.training_histories_long_csv}")
    print(f"file_level_metrics_long_csv: {registry_paths.file_level_metrics_long_csv}")
    print(f"moonshot_comparison_csv: {comparison_csv.resolve()}")
    print(f"moonshot_comparison_json: {comparison_json.resolve()}")
    print(f"best_existing_run_id: {best_existing['run_id']}")
    print(f"best_existing_aggregator: {best_existing['aggregator']}")
    print(f"best_moonshot_run_id: {best_moonshot['run_id']}")
    print(f"best_moonshot_aggregator: {best_moonshot['aggregator']}")
    return 0


def load_json_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected json object in {path}")
    return payload


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


def write_json(output_path: Path, payload: dict) -> None:
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
