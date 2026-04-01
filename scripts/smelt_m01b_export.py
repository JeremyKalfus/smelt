#!/usr/bin/env python3
"""update append-only m01b summaries and registry artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from smelt.evaluation.diagnostics import build_run_registry_entry, export_run_registry_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--figdata-root", type=Path, required=True)
    parser.add_argument("--file-level-root", type=Path, required=True)
    parser.add_argument("--existing-comparison-json", type=Path, required=True)
    parser.add_argument("--original-run-dir", type=Path, required=True)
    parser.add_argument("--benchmark6-run-dir", type=Path, required=True)
    parser.add_argument("--extra6-run-dir", type=Path, required=True)
    parser.add_argument("--seed2-run-dir", type=Path, required=True)
    parser.add_argument("--shuffled-run-dir", type=Path, required=True)
    parser.add_argument("--verification-summary-json", type=Path, required=True)
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

    original_summary = build_ablation_row(args.original_run_dir)
    benchmark6_summary = build_ablation_row(args.benchmark6_run_dir)
    extra6_summary = build_ablation_row(args.extra6_run_dir)
    seed2_summary = build_ablation_row(args.seed2_run_dir)
    shuffled_summary = build_ablation_row(args.shuffled_run_dir)

    anti_cheat_payload = build_anti_cheat_payload(
        verification_summary=load_json_file(args.verification_summary_json),
        shuffled_summary=shuffled_summary,
    )
    anti_cheat_json = args.table_root / "m01b_anti_cheat_summary.json"
    anti_cheat_json.write_text(
        json.dumps(anti_cheat_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ablation_rows = [
        original_summary,
        benchmark6_summary,
        extra6_summary,
        seed2_summary,
    ]
    ablation_csv = args.table_root / "m01b_ablation_summary.csv"
    ablation_json = args.table_root / "m01b_ablation_summary.json"
    write_dict_rows_csv(ablation_csv, ablation_rows)
    ablation_json.write_text(
        json.dumps({"rows": ablation_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    noagg_rows = build_no_aggregation_rows(
        run_dirs=(
            args.original_run_dir,
            args.benchmark6_run_dir,
            args.extra6_run_dir,
            args.seed2_run_dir,
        )
    )
    noagg_csv = args.table_root / "m01b_no_aggregation_vs_aggregation.csv"
    noagg_json = args.table_root / "m01b_no_aggregation_vs_aggregation.json"
    write_dict_rows_csv(noagg_csv, noagg_rows)
    noagg_json.write_text(
        json.dumps({"rows": noagg_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"run_registry_csv: {registry_paths.run_registry_csv}")
    print(f"run_registry_json: {registry_paths.run_registry_json}")
    print(f"metrics_long_csv: {registry_paths.metrics_long_csv}")
    print(f"training_histories_long_csv: {registry_paths.training_histories_long_csv}")
    print(f"file_level_metrics_long_csv: {registry_paths.file_level_metrics_long_csv}")
    print(f"anti_cheat_json: {anti_cheat_json.resolve()}")
    print(f"ablation_csv: {ablation_csv.resolve()}")
    print(f"ablation_json: {ablation_json.resolve()}")
    print(f"noagg_csv: {noagg_csv.resolve()}")
    print(f"noagg_json: {noagg_json.resolve()}")
    print(f"anti_cheat_overall_pass: {anti_cheat_payload['overall_anti_cheat_pass']}")
    return 0


def build_anti_cheat_payload(
    *,
    verification_summary: dict[str, Any],
    shuffled_summary: dict[str, str],
) -> dict[str, Any]:
    shuffled_window_acc = float(shuffled_summary["acc@1"])
    shuffled_best_file_acc = float(shuffled_summary["best_file_acc@1"])
    a4_pass = shuffled_window_acc <= 8.0 and shuffled_best_file_acc <= 15.0
    return {
        "a1_pass": bool(verification_summary["a1_pass"]),
        "a2_pass": bool(verification_summary["a2_pass"]),
        "a3_pass": bool(verification_summary["a3_pass"]),
        "a4_pass": a4_pass,
        "overall_anti_cheat_pass": bool(verification_summary["a1_pass"])
        and bool(verification_summary["a2_pass"])
        and bool(verification_summary["a3_pass"])
        and a4_pass,
        "verification": verification_summary,
        "shuffled_control": shuffled_summary,
    }


def build_ablation_row(run_dir: Path) -> dict[str, str]:
    registry_entry = build_run_registry_entry(run_dir)
    file_level_rows = load_json_file(run_dir / "file_level_metrics_comparison.json")["rows"]
    best_file_row = max(file_level_rows, key=lambda row: float(row["file_acc@1"]))
    return {
        **registry_entry,
        "best_file_aggregator": str(best_file_row["aggregator"]),
        "best_file_acc@1": str(best_file_row["file_acc@1"]),
        "best_file_acc@5": str(best_file_row["file_acc@5"]),
        "best_file_macro_precision": str(best_file_row["file_macro_precision"]),
        "best_file_macro_recall": str(best_file_row["file_macro_recall"]),
        "best_file_macro_f1": str(best_file_row["file_macro_f1"]),
        "best_file_summary_metrics_path": str(best_file_row["file_summary_metrics_path"]),
        "best_file_confusion_matrix_path": str(best_file_row["file_confusion_matrix_path"]),
        "best_file_per_category_accuracy_path": str(
            best_file_row["file_per_category_accuracy_path"]
        ),
        "best_file_predictions_path": str(best_file_row["file_predictions_path"]),
    }


def build_no_aggregation_rows(*, run_dirs: tuple[Path, ...]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        registry_entry = build_run_registry_entry(run_dir)
        rows.append(
            {
                "run_id": registry_entry["run_id"],
                "ticket_stage": registry_entry["ticket_stage"],
                "track": registry_entry["track"],
                "model_family": registry_entry["model_family"],
                "channel_set": registry_entry["channel_set"],
                "view_mode": registry_entry["view_mode"],
                "g": registry_entry["g"],
                "window_size": registry_entry["window_size"],
                "stride": registry_entry["stride"],
                "aggregator": "window_none",
                "window_acc@1": registry_entry["acc@1"],
                "window_acc@5": registry_entry["acc@5"],
                "window_macro_f1": registry_entry["macro_f1"],
                "file_acc@1": "",
                "file_acc@5": "",
                "file_macro_precision": "",
                "file_macro_recall": "",
                "file_macro_f1": "",
                "file_summary_metrics_path": "",
                "file_confusion_matrix_path": "",
                "file_per_category_accuracy_path": "",
                "file_predictions_path": "",
            }
        )
        for file_row in load_json_file(run_dir / "file_level_metrics_comparison.json")["rows"]:
            enriched = dict(file_row)
            enriched.setdefault("ticket_stage", registry_entry["ticket_stage"])
            enriched.setdefault("track", registry_entry["track"])
            enriched.setdefault("model_family", registry_entry["model_family"])
            enriched.setdefault("channel_set", registry_entry["channel_set"])
            enriched.setdefault("view_mode", registry_entry["view_mode"])
            enriched.setdefault("g", registry_entry["g"])
            enriched.setdefault("window_size", registry_entry["window_size"])
            enriched.setdefault("stride", registry_entry["stride"])
            rows.append(enriched)
    return rows


def load_json_file(path: Path) -> dict[str, Any]:
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


if __name__ == "__main__":
    raise SystemExit(main())
