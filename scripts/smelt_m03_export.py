"""append-only export updates for m03 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smelt.evaluation import write_dict_rows_csv, write_json
from smelt.evaluation.diagnostics import (
    build_m03_file_level_run_row,
    build_m03_seed_summary,
    export_run_registry_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--figdata-root", type=Path, required=True)
    parser.add_argument("--file-level-root", type=Path, required=True)
    parser.add_argument("--primary-selection-json", type=Path, required=True)
    parser.add_argument("--ensemble-run-dir", type=Path, required=True)
    parser.add_argument("--learned-run-dir", type=Path, action="append", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    table_root = args.table_root.expanduser().resolve()
    figdata_root = args.figdata_root.expanduser().resolve()
    ensemble_run_dir = args.ensemble_run_dir.expanduser().resolve()
    learned_run_dirs = tuple(path.expanduser().resolve() for path in args.learned_run_dir)

    registry = export_run_registry_artifacts(
        run_root=args.run_root.expanduser().resolve(),
        table_root=table_root,
        figdata_root=figdata_root,
        existing_run_ids=(),
        file_level_root=args.file_level_root.expanduser().resolve(),
    )
    ensemble_row = build_m03_file_level_run_row(ensemble_run_dir)
    learned_rows = [build_m03_file_level_run_row(run_dir) for run_dir in learned_run_dirs]
    seed_summary = build_m03_seed_summary(learned_rows)

    primary_selection = json.loads(
        args.primary_selection_json.expanduser().resolve().read_text(encoding="utf-8")
    )
    protocol_definition = {
        "track": "moonshot-enhanced-setting",
        "frozen_window_encoder_protocol": {
            "channel_set": "all12",
            "view_mode": "diff_all12",
            "g": 25,
            "window_size": 100,
            "stride": 50,
            "grouped_validation": "grouped_by_file_within_training_split",
            "standardization": "train_only",
            "selection_source": "validation_only",
        },
        "primary_encoder_selection": primary_selection,
        "ensemble_selection_rule": {
            "primary": "validation_file_acc@1",
            "tie_break": "validation_file_macro_f1",
            "final_tie_break_order": [
                "mean_probabilities",
                "mean_logits",
                "vote",
            ],
            "source": "validation_only",
        },
        "learned_file_model": {
            "family": "attention_deepsets",
            "encoder_frozen": True,
            "input_features": "penultimate_window_embeddings",
            "checkpoint_selection_rule": {
                "primary": "validation_file_acc@1",
                "tie_break": "validation_file_macro_f1",
                "source": "validation_only",
            },
            "primary_metric": "file_acc@1",
        },
    }

    ensemble_csv = table_root / "m03_ensemble_summary.csv"
    ensemble_json = table_root / "m03_ensemble_summary.json"
    learned_csv = table_root / "m03_learned_file_model_runs.csv"
    learned_json = table_root / "m03_learned_file_model_runs.json"
    seed_summary_csv = table_root / "m03_seed_summary.csv"
    seed_summary_json = table_root / "m03_seed_summary.json"
    protocol_json = table_root / "m03_protocol_definition.json"

    write_dict_rows_csv(ensemble_csv, [ensemble_row])
    write_json(ensemble_json, {"rows": [ensemble_row]})
    write_dict_rows_csv(learned_csv, learned_rows)
    write_json(learned_json, {"rows": learned_rows})
    write_dict_rows_csv(
        seed_summary_csv,
        [
            {
                "n_runs": str(seed_summary["n_runs"]),
                "file_acc@1_mean": str(seed_summary["file_acc@1"]["mean"]),
                "file_acc@1_std": str(seed_summary["file_acc@1"]["std"]),
                "file_acc@5_mean": str(seed_summary["file_acc@5"]["mean"]),
                "file_acc@5_std": str(seed_summary["file_acc@5"]["std"]),
                "file_macro_f1_mean": str(seed_summary["file_macro_f1"]["mean"]),
                "file_macro_f1_std": str(seed_summary["file_macro_f1"]["std"]),
            }
        ],
    )
    write_json(seed_summary_json, seed_summary)
    write_json(protocol_json, protocol_definition)

    print(f"run_registry_csv: {registry.run_registry_csv}")
    print(f"run_registry_json: {registry.run_registry_json}")
    print(f"metrics_long_csv: {registry.metrics_long_csv}")
    print(f"training_histories_long_csv: {registry.training_histories_long_csv}")
    print(f"file_level_metrics_long_csv: {registry.file_level_metrics_long_csv}")
    print(f"ensemble_summary_csv: {ensemble_csv.resolve()}")
    print(f"ensemble_summary_json: {ensemble_json.resolve()}")
    print(f"learned_runs_csv: {learned_csv.resolve()}")
    print(f"learned_runs_json: {learned_json.resolve()}")
    print(f"seed_summary_csv: {seed_summary_csv.resolve()}")
    print(f"seed_summary_json: {seed_summary_json.resolve()}")
    print(f"protocol_definition_json: {protocol_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
