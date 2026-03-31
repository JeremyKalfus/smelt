"""update registry/figdata artifacts and write t11 baseline comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smelt.evaluation.diagnostics import (
    compare_research_supervised_recipe_compatibility,
    export_comparison_summary,
    export_run_registry_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--figdata-root", type=Path, required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--comparison-run-dir", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registry_paths = export_run_registry_artifacts(
        run_root=args.run_root,
        table_root=args.table_root,
        figdata_root=args.figdata_root,
        existing_run_ids=(),
    )
    compatibility = compare_research_supervised_recipe_compatibility(
        args.baseline_config,
        args.comparison_config,
    )
    comparison_paths = export_comparison_summary(
        baseline_run_dir=args.baseline_run_dir,
        comparison_run_dir=args.comparison_run_dir,
        output_csv=args.table_root / "t11_fair_baseline_comparison.csv",
        output_json=args.table_root / "t11_fair_baseline_comparison.json",
        label="t11_fine_tune_vs_fair_diff_baseline",
    )
    compatibility_path = args.table_root / "t11_baseline_compatibility.json"
    compatibility_path.write_text(
        json.dumps(compatibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"run_registry_csv: {registry_paths.run_registry_csv}")
    print(f"run_registry_json: {registry_paths.run_registry_json}")
    print(f"metrics_long_csv: {registry_paths.metrics_long_csv}")
    print(f"training_histories_long_csv: {registry_paths.training_histories_long_csv}")
    print(f"comparison_csv: {comparison_paths.csv_path}")
    print(f"comparison_json: {comparison_paths.json_path}")
    print(f"compatibility_json: {compatibility_path.resolve()}")
    print(f"compatible: {compatibility['compatible']}")
    print(f"material_mismatch_count: {len(compatibility['material_mismatches'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
