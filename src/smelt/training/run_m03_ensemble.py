"""build the validation-locked cross-seed ensemble baseline for m03."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from smelt.evaluation import export_classification_report, export_file_level_report, write_json

from .m03 import evaluate_locked_seed_ensemble
from .run import build_run_dir, validate_required_reference, write_run_metadata


class M03EnsembleRunError(Exception):
    """raised when the m03 ensemble baseline cannot proceed safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--export-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--category-map-path", type=Path, required=True)
    parser.add_argument("--class-vocab-manifest-path", type=Path, required=True)
    parser.add_argument("--exact-upstream-regression-path", type=Path, required=True)
    parser.add_argument("--primary-selection-json", type=Path, required=True)
    parser.add_argument("--protocol-definition-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir, metrics = run_m03_ensemble(
        run_dirs=tuple(path.expanduser().resolve() for path in args.run_dir),
        export_dirs=tuple(path.expanduser().resolve() for path in args.export_dir),
        output_root=args.output_root.expanduser().resolve(),
        table_root=args.table_root.expanduser().resolve(),
        category_map_path=args.category_map_path.expanduser().resolve(),
        class_vocab_manifest_path=args.class_vocab_manifest_path.expanduser().resolve(),
        exact_upstream_regression_path=args.exact_upstream_regression_path.expanduser().resolve(),
        primary_selection_json=args.primary_selection_json.expanduser().resolve(),
        protocol_definition_path=args.protocol_definition_path.expanduser().resolve(),
    )
    print(f"run_dir: {run_dir}")
    print(f"file_acc@1: {metrics.acc_at_1}")
    print(f"file_acc@5: {metrics.acc_at_5}")
    print(f"file_precision_macro: {metrics.precision_macro}")
    print(f"file_recall_macro: {metrics.recall_macro}")
    print(f"file_f1_macro: {metrics.f1_macro}")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    print(f"locked_primary_aggregator: {metadata.get('locked_primary_aggregator', '')}")
    print(f"selection_json: {Path(metadata.get('ensemble_selection_path', '')).resolve()}")
    return 0


def run_m03_ensemble(
    *,
    run_dirs: tuple[Path, ...],
    export_dirs: tuple[Path, ...],
    output_root: Path,
    table_root: Path,
    category_map_path: Path,
    class_vocab_manifest_path: Path,
    exact_upstream_regression_path: Path,
    primary_selection_json: Path,
    protocol_definition_path: Path,
) -> tuple[Path, Any]:
    validate_required_reference(category_map_path)
    validate_required_reference(class_vocab_manifest_path)
    validate_required_reference(exact_upstream_regression_path)
    validate_required_reference(primary_selection_json)
    validate_required_reference(protocol_definition_path)
    config_payload = {
        "track": "moonshot-enhanced-setting",
        "experiment_name": "m03_locked_seed_ensemble",
        "output_root": str(output_root),
        "category_map_path": str(category_map_path),
        "class_vocab_manifest_path": str(class_vocab_manifest_path),
        "exact_upstream_regression_path": str(exact_upstream_regression_path),
        "primary_selection_json": str(primary_selection_json),
        "protocol_definition_path": str(protocol_definition_path),
        "source_run_dirs": [str(path) for path in run_dirs],
        "source_export_dirs": [str(path) for path in export_dirs],
        "model_name": "locked_seed_ensemble",
        "channel_set": "all12",
        "diff_period": 25,
        "window_size": 100,
        "stride": 50,
        "view_mode": "diff_all12",
        "candidate_methods": ["mean_logits", "mean_probabilities", "vote"],
    }
    run_dir = build_run_dir(output_root, "m03_locked_seed_ensemble", config_payload)
    run_dir.mkdir(parents=True, exist_ok=False)

    from smelt.evaluation import load_category_mapping

    category_mapping = load_category_mapping(category_map_path)
    evaluation = evaluate_locked_seed_ensemble(
        export_dirs=export_dirs,
        run_dirs=run_dirs,
        category_mapping=category_mapping,
    )
    selected_method = str(evaluation["selected_method"])
    validation_results = evaluation["validation_results"]
    test_results = evaluation["test_results"]

    validation_rows = []
    test_rows = []
    validation_paths: dict[str, dict[str, str]] = {}
    test_paths: dict[str, dict[str, str]] = {}
    for method_name in ("mean_logits", "mean_probabilities", "vote"):
        validation_report = export_file_level_report(
            output_root=run_dir / "validation_file_level",
            run_name=method_name,
            result=validation_results[method_name],
            methods_summary={
                "track": "moonshot-enhanced-setting",
                "mode": "file_level_ensemble",
                "view_mode": "diff_all12",
                "channel_set": "all12",
                "selection_source": "validation_only",
                "candidate_method": method_name,
            },
        )
        test_report = export_file_level_report(
            output_root=run_dir / "file_level",
            run_name=method_name,
            result=test_results[method_name],
            methods_summary={
                "track": "moonshot-enhanced-setting",
                "mode": "file_level_ensemble",
                "view_mode": "diff_all12",
                "channel_set": "all12",
                "selection_source": "validation_only",
                "candidate_method": method_name,
            },
        )
        validation_paths[method_name] = validation_report.to_dict()
        test_paths[method_name] = test_report.to_dict()
        validation_rows.append(
            build_ensemble_row(
                run_id=run_dir.name,
                method_name=method_name,
                split_name="validation",
                metrics=validation_results[method_name].metrics,
                report_paths=validation_report.to_dict(),
            )
        )
        test_rows.append(
            build_ensemble_row(
                run_id=run_dir.name,
                method_name=method_name,
                split_name="test",
                metrics=test_results[method_name].metrics,
                report_paths=test_report.to_dict(),
            )
        )

    export_classification_report(
        output_root=run_dir.parent,
        run_name=run_dir.name,
        metrics=test_results[selected_method].metrics,
        methods_summary={
            "track": "moonshot-enhanced-setting",
            "mode": "file_level_ensemble",
            "view_mode": "diff_all12",
            "channel_set": "all12",
            "selection_source": "validation_only",
            "locked_primary_aggregator": selected_method,
            "metric_scope": "file_level",
        },
        overwrite=True,
    )

    selection_payload = {
        "selection_rule": {
            "primary": "validation_file_acc@1",
            "tie_break": "validation_file_macro_f1",
            "final_tie_break_order": [
                "mean_probabilities",
                "mean_logits",
                "vote",
            ],
            "source": "validation_only",
        },
        "selected_method": selected_method,
        "candidates": [
            {
                "method": method_name,
                "validation_file_acc@1": validation_results[method_name].metrics.acc_at_1,
                "validation_file_acc@5": validation_results[method_name].metrics.acc_at_5,
                "validation_file_macro_f1": validation_results[method_name].metrics.f1_macro,
            }
            for method_name in ("mean_logits", "mean_probabilities", "vote")
        ],
    }
    table_root.mkdir(parents=True, exist_ok=True)
    selection_path = table_root / "m03_ensemble_selection.json"
    write_json(selection_path, selection_payload)
    write_json(run_dir / "ensemble_selection.json", selection_payload)
    write_json(run_dir / "validation_file_level_metrics_comparison.json", {"rows": validation_rows})
    write_json(run_dir / "file_level_metrics_comparison.json", {"rows": test_rows})
    write_csv(run_dir / "validation_file_level_metrics_comparison.csv", validation_rows)
    write_csv(run_dir / "file_level_metrics_comparison.csv", test_rows)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=True),
        encoding="utf-8",
    )
    write_run_metadata(
        run_dir / "run_metadata.json",
        {
            "track": "moonshot-enhanced-setting",
            "mode": "file_level_ensemble",
            "view_mode": "diff_all12",
            "channel_set": "all12",
            "encoder_frozen": True,
            "file_level_model_family": "locked_seed_ensemble",
            "locked_primary_aggregator": selected_method,
            "primary_checkpoint_selection_metric": (
                "validation_file_acc@1_then_validation_file_macro_f1"
            ),
            "aggregator_selection_source": "validation_only",
            "checkpoint_path": "",
            "best_checkpoint_path": "",
            "window_counts": {
                "train": "",
                "validation": "",
                "test": "",
            },
            "file_level_primary_report": test_paths[selected_method],
            "validation_file_level_primary_report": validation_paths[selected_method],
            "reference_artifacts": {
                "category_map_path": str(category_map_path.resolve()),
                "class_vocab_manifest_path": str(class_vocab_manifest_path.resolve()),
                "exact_upstream_regression_path": str(exact_upstream_regression_path.resolve()),
                "primary_selection_json": str(primary_selection_json.resolve()),
                "protocol_definition_path": str(protocol_definition_path.resolve()),
            },
            "ensemble_selection_path": str(selection_path.resolve()),
            "source_run_ids": [path.name for path in run_dirs],
            "source_export_dirs": [str(path.resolve()) for path in export_dirs],
            "best_validation_summary": {
                "file_acc@1": validation_results[selected_method].metrics.acc_at_1,
                "file_acc@5": validation_results[selected_method].metrics.acc_at_5,
                "file_macro_f1": validation_results[selected_method].metrics.f1_macro,
            },
        },
    )
    shutil.copyfile(class_vocab_manifest_path, run_dir / "base_class_vocab.json")
    return run_dir, test_results[selected_method].metrics


def build_ensemble_row(
    *,
    run_id: str,
    method_name: str,
    split_name: str,
    metrics: Any,
    report_paths: dict[str, str],
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "track": "moonshot-enhanced-setting",
        "encoder_source_run_id": "m01c_locked_seed_ensemble",
        "file_level_model_family": "locked_seed_ensemble",
        "encoder_frozen": "true",
        "selection_rules": "validation_only_ensemble_method_selection",
        "split": split_name,
        "channel_set": "all12",
        "view_mode": "diff_all12",
        "g": "25",
        "window_size": "100",
        "stride": "50",
        "aggregator": method_name,
        "file_acc@1": str(metrics.acc_at_1),
        "file_acc@5": str(metrics.acc_at_5),
        "file_macro_precision": str(metrics.precision_macro),
        "file_macro_recall": str(metrics.recall_macro),
        "file_macro_f1": str(metrics.f1_macro),
        "summary_json": report_paths.get("summary_json", ""),
        "confusion_matrix_csv": report_paths.get("confusion_matrix_csv", ""),
        "per_category_accuracy_csv": report_paths.get("per_category_accuracy_csv", ""),
        "per_file_predictions_csv": report_paths.get("per_file_predictions_csv", ""),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
