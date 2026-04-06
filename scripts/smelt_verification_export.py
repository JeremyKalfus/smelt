"""verification-only export pass for reproducibility and paper readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

from smelt.training.verification_sprint import run_verification_sprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--file-level-root", type=Path, required=True)
    parser.add_argument("--class-vocab-manifest-path", type=Path, required=True)
    parser.add_argument("--category-map-path", type=Path, required=True)
    parser.add_argument("--exact-regression-artifact-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifacts = run_verification_sprint(
        repo_root=args.repo_root,
        run_root=args.run_root,
        table_root=args.table_root,
        file_level_root=args.file_level_root,
        class_vocab_manifest_path=args.class_vocab_manifest_path,
        category_map_path=args.category_map_path,
        exact_regression_artifact_path=args.exact_regression_artifact_path,
    )
    print(f"verification_inventory_csv: {artifacts.inventory_csv}")
    print(f"verification_inventory_json: {artifacts.inventory_json}")
    print(f"verification_exact_upstream_json: {artifacts.exact_json}")
    print(f"verification_moonshot_protocol_json: {artifacts.moonshot_json}")
    print(f"verification_leakage_selection_audit_json: {artifacts.leakage_json}")
    print(f"verification_bootstrap_ci_csv: {artifacts.bootstrap_csv}")
    print(f"verification_bootstrap_ci_json: {artifacts.bootstrap_json}")
    print(f"paper_baseline_table_csv: {artifacts.paper_baseline_csv}")
    print(f"paper_ablation_table_csv: {artifacts.paper_ablation_csv}")
    print(f"paper_main_results_table_csv: {artifacts.paper_main_results_csv}")
    print(f"paper_diversity_table_csv: {artifacts.paper_diversity_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
