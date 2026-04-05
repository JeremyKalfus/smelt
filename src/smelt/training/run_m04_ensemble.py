"""build the validation-locked heterogeneous ensemble bank for m04."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from smelt.evaluation import (
    export_classification_report,
    export_file_level_report,
    load_category_mapping,
    write_json,
)
from smelt.evaluation.diagnostics import build_m04_model_bank_row

from .m04 import (
    M04_ENSEMBLE_METHODS,
    build_diversity_matrix_rows,
    evaluate_m04_ensemble_candidates,
    export_model_bank_features,
    load_locked_file_score_bundles,
    write_csv,
    write_diversity_artifacts,
)
from .run import build_run_dir, validate_required_reference, write_run_metadata


class M04EnsembleRunError(Exception):
    """raised when the m04 ensemble bank cannot complete safely."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--category-map-path", type=Path, required=True)
    parser.add_argument("--class-vocab-manifest-path", type=Path, required=True)
    parser.add_argument("--exact-upstream-regression-path", type=Path, required=True)
    parser.add_argument("--protocol-definition-path", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir, metrics, selected_method = run_m04_ensemble(
        run_dirs=tuple(path.expanduser().resolve() for path in args.run_dir),
        output_root=args.output_root.expanduser().resolve(),
        export_root=args.export_root.expanduser().resolve(),
        table_root=args.table_root.expanduser().resolve(),
        category_map_path=args.category_map_path.expanduser().resolve(),
        class_vocab_manifest_path=args.class_vocab_manifest_path.expanduser().resolve(),
        exact_upstream_regression_path=args.exact_upstream_regression_path.expanduser().resolve(),
        protocol_definition_path=args.protocol_definition_path.expanduser().resolve(),
    )
    print(f"run_dir: {run_dir}")
    print(f"file_acc@1: {metrics.acc_at_1}")
    print(f"file_acc@5: {metrics.acc_at_5}")
    print(f"file_precision_macro: {metrics.precision_macro}")
    print(f"file_recall_macro: {metrics.recall_macro}")
    print(f"file_f1_macro: {metrics.f1_macro}")
    print(f"locked_primary_aggregator: {selected_method}")
    selection_path = args.table_root.expanduser().resolve() / "m04_ensemble_selection.json"
    print(f"ensemble_selection_json: {selection_path.resolve()}")
    return 0


def run_m04_ensemble(
    *,
    run_dirs: tuple[Path, ...],
    output_root: Path,
    export_root: Path,
    table_root: Path,
    category_map_path: Path,
    class_vocab_manifest_path: Path,
    exact_upstream_regression_path: Path,
    protocol_definition_path: Path,
) -> tuple[Path, Any, str]:
    validate_required_reference(category_map_path)
    validate_required_reference(class_vocab_manifest_path)
    validate_required_reference(exact_upstream_regression_path)
    validate_required_reference(protocol_definition_path)
    export_dirs = export_model_bank_features(run_dirs=run_dirs, output_root=export_root)
    category_mapping = load_category_mapping(category_map_path)

    validation_bundles = load_locked_file_score_bundles(
        run_dirs=run_dirs,
        export_dirs=export_dirs,
        split_name="validation",
    )
    test_bundles = load_locked_file_score_bundles(
        run_dirs=run_dirs,
        export_dirs=export_dirs,
        split_name="test",
    )
    diversity_rows = build_diversity_matrix_rows(validation_bundles=validation_bundles)
    selection_payload, candidates = evaluate_m04_ensemble_candidates(
        validation_bundles=validation_bundles,
        test_bundles=test_bundles,
        category_mapping=category_mapping,
    )
    selected_method = str(selection_payload["selected_method"])
    selected_candidate = candidates[selected_method]

    config_payload = {
        "track": "moonshot-enhanced-setting",
        "experiment_name": "m04_locked_heterogeneous_ensemble",
        "output_root": str(output_root),
        "export_root": str(export_root),
        "table_root": str(table_root),
        "category_map_path": str(category_map_path),
        "class_vocab_manifest_path": str(class_vocab_manifest_path),
        "exact_upstream_regression_path": str(exact_upstream_regression_path),
        "protocol_definition_path": str(protocol_definition_path),
        "source_run_dirs": [str(path) for path in run_dirs],
        "source_export_dirs": [str(path) for path in export_dirs],
        "model_name": "locked_heterogeneous_ensemble",
        "channel_set": "all12",
        "diff_period": 25,
        "window_size": 100,
        "stride": 50,
        "view_mode": "diff_all12",
        "candidate_methods": list(M04_ENSEMBLE_METHODS),
    }
    run_dir = build_run_dir(output_root, "m04_locked_heterogeneous_ensemble", config_payload)
    run_dir.mkdir(parents=True, exist_ok=False)

    validation_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    validation_paths: dict[str, dict[str, str]] = {}
    test_paths: dict[str, dict[str, str]] = {}
    for method_name in M04_ENSEMBLE_METHODS:
        validation_report = export_file_level_report(
            output_root=run_dir / "validation_file_level",
            run_name=method_name,
            result=candidates[method_name].validation_result,
            methods_summary={
                "track": "moonshot-enhanced-setting",
                "mode": "heterogeneous_file_level_ensemble",
                "view_mode": "diff_all12",
                "channel_set": "all12",
                "selection_source": "validation_only",
                "candidate_method": method_name,
            },
        )
        test_report = export_file_level_report(
            output_root=run_dir / "file_level",
            run_name=method_name,
            result=candidates[method_name].test_result,
            methods_summary={
                "track": "moonshot-enhanced-setting",
                "mode": "heterogeneous_file_level_ensemble",
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
                metrics=candidates[method_name].validation_result.metrics,
                report_paths=validation_report.to_dict(),
                member_run_ids=candidates[method_name].member_run_ids,
                weights=candidates[method_name].weights,
            )
        )
        test_rows.append(
            build_ensemble_row(
                run_id=run_dir.name,
                method_name=method_name,
                split_name="test",
                metrics=candidates[method_name].test_result.metrics,
                report_paths=test_report.to_dict(),
                member_run_ids=candidates[method_name].member_run_ids,
                weights=candidates[method_name].weights,
            )
        )

    export_classification_report(
        output_root=run_dir.parent,
        run_name=run_dir.name,
        metrics=selected_candidate.test_result.metrics,
        methods_summary={
            "track": "moonshot-enhanced-setting",
            "mode": "heterogeneous_file_level_ensemble",
            "view_mode": "diff_all12",
            "channel_set": "all12",
            "selection_source": "validation_only",
            "locked_primary_aggregator": selected_method,
            "metric_scope": "file_level",
        },
        overwrite=True,
    )

    table_root.mkdir(parents=True, exist_ok=True)
    model_bank_rows = [
        build_m04_model_bank_row(run_dir=run_dir_path, export_dir=export_dir)
        for run_dir_path, export_dir in zip(
            sorted(run_dirs, key=lambda path: path.name),
            export_dirs,
            strict=True,
        )
    ]
    model_bank_csv = table_root / "m04_model_bank.csv"
    model_bank_json = table_root / "m04_model_bank.json"
    diversity_csv = table_root / "m04_diversity_matrix.csv"
    diversity_json = table_root / "m04_diversity_matrix.json"
    selection_json = table_root / "m04_ensemble_selection.json"
    search_csv = table_root / "m04_ensemble_search_summary.csv"
    search_json = table_root / "m04_ensemble_search_summary.json"
    write_csv(model_bank_csv, model_bank_rows)
    write_json(model_bank_json, {"rows": model_bank_rows})
    write_diversity_artifacts(rows=diversity_rows, csv_path=diversity_csv, json_path=diversity_json)
    write_json(selection_json, selection_payload)
    search_rows = [
        candidates[method_name].to_summary_row(selected=method_name == selected_method)
        for method_name in M04_ENSEMBLE_METHODS
    ]
    write_csv(search_csv, search_rows)
    write_json(search_json, {"rows": search_rows})

    write_json(run_dir / "ensemble_selection.json", selection_payload)
    write_json(run_dir / "validation_file_level_metrics_comparison.json", {"rows": validation_rows})
    write_json(run_dir / "file_level_metrics_comparison.json", {"rows": test_rows})
    write_csv(run_dir / "validation_file_level_metrics_comparison.csv", validation_rows)
    write_csv(run_dir / "file_level_metrics_comparison.csv", test_rows)
    write_csv(run_dir / "m04_model_bank.csv", model_bank_rows)
    write_diversity_artifacts(
        rows=diversity_rows,
        csv_path=run_dir / "m04_diversity_matrix.csv",
        json_path=run_dir / "m04_diversity_matrix.json",
    )
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=True),
        encoding="utf-8",
    )
    write_run_metadata(
        run_dir / "run_metadata.json",
        {
            "track": "moonshot-enhanced-setting",
            "mode": "heterogeneous_file_level_ensemble",
            "view_mode": "diff_all12",
            "channel_set": "all12",
            "encoder_frozen": True,
            "file_level_model_family": "locked_heterogeneous_ensemble",
            "locked_primary_aggregator": selected_method,
            "primary_checkpoint_selection_metric": (
                "validation_file_acc@1_then_validation_file_macro_f1_then_diversity"
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
                "protocol_definition_path": str(protocol_definition_path.resolve()),
                "model_bank_path": str(model_bank_json.resolve()),
                "diversity_matrix_path": str(diversity_json.resolve()),
            },
            "ensemble_selection_path": str(selection_json.resolve()),
            "ensemble_search_summary_path": str(search_json.resolve()),
            "source_run_ids": [path.name for path in run_dirs],
            "source_export_dirs": [str(path.resolve()) for path in export_dirs],
            "source_families": sorted({row["family_name"] for row in model_bank_rows}),
            "best_validation_summary": {
                "file_acc@1": selected_candidate.validation_result.metrics.acc_at_1,
                "file_acc@5": selected_candidate.validation_result.metrics.acc_at_5,
                "file_macro_f1": selected_candidate.validation_result.metrics.f1_macro,
            },
            "selected_member_run_ids": list(selected_candidate.member_run_ids),
            "selected_weights": list(selected_candidate.weights),
            "avg_pairwise_agreement": selected_candidate.avg_pairwise_agreement,
            "avg_pairwise_correlation": selected_candidate.avg_pairwise_correlation,
        },
    )
    shutil.copyfile(class_vocab_manifest_path, run_dir / "base_class_vocab.json")
    return run_dir, selected_candidate.test_result.metrics, selected_method


def build_ensemble_row(
    *,
    run_id: str,
    method_name: str,
    split_name: str,
    metrics: Any,
    report_paths: dict[str, str],
    member_run_ids: tuple[str, ...],
    weights: tuple[float, ...],
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "track": "moonshot-enhanced-setting",
        "encoder_source_run_id": "m04_model_bank",
        "file_level_model_family": "locked_heterogeneous_ensemble",
        "encoder_frozen": "true",
        "selection_rules": "validation_only_diversity_locked_ensemble_selection",
        "split": split_name,
        "channel_set": "all12",
        "view_mode": "diff_all12",
        "g": "25",
        "window_size": "100",
        "stride": "50",
        "aggregator": method_name,
        "member_run_ids": json.dumps(list(member_run_ids)),
        "weights": json.dumps(list(weights)),
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


if __name__ == "__main__":
    raise SystemExit(main())
