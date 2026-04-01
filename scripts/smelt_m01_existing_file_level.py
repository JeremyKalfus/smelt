#!/usr/bin/env python3
"""evaluate existing runs with file-level aggregation for moonshot m01."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smelt.evaluation import (
    FILE_LEVEL_AGGREGATORS,
    FileLevelAggregationError,
    aggregate_file_level_metrics,
    build_window_prediction_bundle,
    export_file_level_report,
    load_window_prediction_bundle,
    write_window_prediction_bundle,
)
from smelt.evaluation.diagnostics import build_run_registry_entry, write_dict_rows_csv
from smelt.training.replay import load_replay_context
from smelt.training.run import collect_evaluation_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--comparison-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rows: list[dict[str, str]] = []
    strongest_row: dict[str, str] | None = None

    for run_dir in args.run_dir:
        replay = load_replay_context(run_dir)
        registry_entry = build_run_registry_entry(replay.run_dir)
        bundle_root = args.output_root / replay.run_dir.name
        bundle_root.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_root / "window_predictions.npz"

        try:
            bundle = load_window_prediction_bundle(bundle_path)
            reexported = False
        except FileLevelAggregationError:
            evaluation = collect_evaluation_outputs(
                model=replay.model,
                data_loader=replay.test_loader,
                device=replay.device,
                class_names=replay.class_names,
                category_mapping=replay.category_mapping,
            )
            bundle = build_window_prediction_bundle(
                class_names=replay.class_names,
                true_labels=evaluation.true_labels,
                predicted_labels=evaluation.predicted_labels,
                topk_indices=evaluation.topk_indices,
                logits=evaluation.logits,
                windows=replay.test_windows.windows,
            )
            write_window_prediction_bundle(bundle_path, bundle)
            reexported = True

        print(f"run_id: {replay.run_dir.name}")
        print(f"window_prediction_bundle: {bundle_path.resolve()}")
        print(f"reexported_from_checkpoint: {reexported}")

        for aggregator in FILE_LEVEL_AGGREGATORS:
            result = aggregate_file_level_metrics(
                bundle=bundle,
                category_mapping=replay.category_mapping,
                aggregator=aggregator,
            )
            report = export_file_level_report(
                output_root=bundle_root,
                run_name=aggregator,
                result=result,
                methods_summary={
                    "level": "file",
                    "aggregator": aggregator,
                    "source_run_dir": str(replay.run_dir.resolve()),
                    "window_prediction_bundle_path": str(bundle_path.resolve()),
                },
            )
            row = {
                "run_id": registry_entry["run_id"],
                "ticket_stage": registry_entry["ticket_stage"],
                "track": registry_entry["track"],
                "model_family": registry_entry["model_family"],
                "view_mode": registry_entry["view_mode"],
                "g": registry_entry["g"],
                "window_size": registry_entry["window_size"],
                "stride": registry_entry["stride"],
                "window_acc@1": registry_entry["acc@1"],
                "window_acc@5": registry_entry["acc@5"],
                "window_macro_f1": registry_entry["macro_f1"],
                "file_aggregator": aggregator,
                "file_acc@1": str(result.metrics.acc_at_1),
                "file_acc@5": str(result.metrics.acc_at_5),
                "file_macro_f1": str(result.metrics.f1_macro),
                "bundle_path": str(bundle_path.resolve()),
                "file_summary_path": report.summary_json,
                "file_confusion_matrix_path": report.confusion_matrix_csv,
                "file_per_category_accuracy_path": report.per_category_accuracy_csv,
                "file_per_predictions_path": report.per_file_predictions_csv,
            }
            rows.append(row)
            if strongest_row is None or float(row["file_acc@1"]) > float(
                strongest_row["file_acc@1"]
            ):
                strongest_row = row
            print(
                "  "
                f"{aggregator}: file_acc@1={result.metrics.acc_at_1}, "
                f"file_acc@5={result.metrics.acc_at_5}, "
                f"file_macro_f1={result.metrics.f1_macro}"
            )

    args.comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_json.parent.mkdir(parents=True, exist_ok=True)
    write_dict_rows_csv(args.comparison_csv, rows)
    args.comparison_json.write_text(
        json.dumps(
            {
                "rows": rows,
                "strongest_existing_file_level_baseline": strongest_row,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"comparison_csv: {args.comparison_csv.resolve()}")
    print(f"comparison_json: {args.comparison_json.resolve()}")
    if strongest_row is not None:
        print(f"strongest_run_id: {strongest_row['run_id']}")
        print(f"strongest_aggregator: {strongest_row['file_aggregator']}")
        print(f"strongest_file_acc@1: {strongest_row['file_acc@1']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
