"""export run registry, figdata, and recipe diff artifacts for t10b."""

from __future__ import annotations

import argparse
from pathlib import Path

from smelt.evaluation.diagnostics import export_recipe_diff, export_run_registry_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--figdata-root", type=Path, required=True)
    parser.add_argument("--exact-config", type=Path, required=True)
    parser.add_argument("--research-config", type=Path, required=True)
    parser.add_argument("--existing-run-id", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registry_paths = export_run_registry_artifacts(
        run_root=args.run_root,
        table_root=args.table_root,
        figdata_root=args.figdata_root,
        existing_run_ids=tuple(args.existing_run_id),
    )
    recipe_paths = export_recipe_diff(
        exact_config_path=args.exact_config,
        research_config_path=args.research_config,
        output_csv=args.table_root / "t10b_recipe_diff.csv",
        output_json=args.table_root / "t10b_recipe_diff.json",
    )
    print(f"run_registry_csv: {registry_paths.run_registry_csv}")
    print(f"run_registry_json: {registry_paths.run_registry_json}")
    print(f"existing_run_summary_csv: {registry_paths.existing_run_summary_csv}")
    print(f"metrics_long_csv: {registry_paths.metrics_long_csv}")
    print(f"training_histories_long_csv: {registry_paths.training_histories_long_csv}")
    print(f"recipe_diff_csv: {recipe_paths.csv_path}")
    print(f"recipe_diff_json: {recipe_paths.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
