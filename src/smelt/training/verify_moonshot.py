"""anti-cheat verification for moonshot-enhanced-setting runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smelt.datasets import load_base_sensor_dataset, prepare_moonshot_window_splits
from smelt.evaluation import (
    FILE_LEVEL_AGGREGATORS,
    ClassificationMetrics,
    aggregate_file_level_metrics,
    build_window_prediction_bundle,
    compute_classification_metrics,
    export_classification_report,
    export_file_level_report,
    load_window_prediction_bundle,
    write_window_prediction_bundle,
)

from .replay import load_replay_context
from .run import build_run_dir, collect_evaluation_outputs
from .run_moonshot import load_moonshot_run_config

FLOAT_TOLERANCE = 1e-9


class MoonshotVerificationError(Exception):
    """raised when moonshot anti-cheat verification fails."""


@dataclass(slots=True)
class MoonshotVerificationResult:
    verification_dir: Path
    summary_metrics_path: Path
    confusion_matrix_path: Path
    per_category_accuracy_path: Path
    predictions_path: Path
    file_level_summary_paths: dict[str, str]
    metric_comparison_path: Path
    independent_recompute_path: Path
    leakage_audit_path: Path
    verification_summary_path: Path
    a1_pass: bool
    a2_pass: bool
    a3_pass: bool
    window_acc_at_1: float
    window_acc_at_5: float
    window_precision_macro: float
    window_recall_macro: float
    window_f1_macro: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = verify_moonshot_run(args.run_dir)
    print(f"verification_dir: {result.verification_dir}")
    print(f"summary_metrics_path: {result.summary_metrics_path}")
    print(f"confusion_matrix_path: {result.confusion_matrix_path}")
    print(f"per_category_accuracy_path: {result.per_category_accuracy_path}")
    print(f"predictions_path: {result.predictions_path}")
    print(f"metric_comparison_path: {result.metric_comparison_path}")
    print(f"independent_recompute_path: {result.independent_recompute_path}")
    print(f"leakage_audit_path: {result.leakage_audit_path}")
    print(f"verification_summary_path: {result.verification_summary_path}")
    print(f"a1_pass: {result.a1_pass}")
    print(f"a2_pass: {result.a2_pass}")
    print(f"a3_pass: {result.a3_pass}")
    print(f"window_acc@1: {result.window_acc_at_1}")
    print(f"window_acc@5: {result.window_acc_at_5}")
    print(f"window_precision_macro: {result.window_precision_macro}")
    print(f"window_recall_macro: {result.window_recall_macro}")
    print(f"window_f1_macro: {result.window_f1_macro}")
    for aggregator in FILE_LEVEL_AGGREGATORS:
        print(f"{aggregator}_summary_path: {result.file_level_summary_paths[aggregator]}")
    return 0


def verify_moonshot_run(run_dir: Path) -> MoonshotVerificationResult:
    resolved_run_dir = run_dir.expanduser().resolve()
    saved_summary = load_saved_summary_metrics(resolved_run_dir / "summary_metrics.json")
    saved_file_level = load_saved_file_level_metrics(
        resolved_run_dir / "file_level_metrics_comparison.json"
    )
    replay = load_replay_context(resolved_run_dir)

    evaluation = collect_evaluation_outputs(
        model=replay.model,
        data_loader=replay.test_loader,
        device=replay.device,
        class_names=replay.class_names,
        category_mapping=replay.category_mapping,
    )
    verification_name = build_run_dir(
        resolved_run_dir / "verification",
        "m01b_anti_cheat_eval_only",
        {"run_dir": str(resolved_run_dir)},
    ).name
    report_paths = export_classification_report(
        output_root=resolved_run_dir / "verification",
        run_name=verification_name,
        metrics=evaluation.metrics,
        methods_summary={
            "original_run_dir": str(resolved_run_dir),
            "mode": "eval_only_replay",
        },
    )
    verification_dir = Path(report_paths.report_dir)
    predictions_path = verification_dir / "predictions.npz"
    bundle = build_window_prediction_bundle(
        class_names=evaluation.metrics.class_names,
        true_labels=evaluation.true_labels,
        predicted_labels=evaluation.predicted_labels,
        topk_indices=evaluation.topk_indices,
        logits=evaluation.logits,
        windows=replay.test_windows.windows,
    )
    write_window_prediction_bundle(predictions_path, bundle)

    compare_metrics(saved_summary, evaluation.metrics, context="window_eval_only")
    if int(evaluation.metrics.confusion_matrix.sum()) != bundle.sample_count:
        raise MoonshotVerificationError(
            "recomputed confusion matrix total does not match sample count"
        )

    file_level_summary_paths: dict[str, str] = {}
    reevaluated_file_metrics: dict[str, dict[str, float]] = {}
    for aggregator in FILE_LEVEL_AGGREGATORS:
        result = aggregate_file_level_metrics(
            bundle=bundle,
            category_mapping=replay.category_mapping,
            aggregator=aggregator,
        )
        report = export_file_level_report(
            output_root=verification_dir / "file_level",
            run_name=aggregator,
            result=result,
            methods_summary={
                "original_run_dir": str(resolved_run_dir),
                "mode": "eval_only_replay",
                "aggregator": aggregator,
            },
        )
        file_level_summary_paths[aggregator] = report.summary_json
        reevaluated_file_metrics[aggregator] = metrics_to_dict(result.metrics)
        compare_metric_dicts(
            saved_file_level[aggregator],
            metrics_to_dict(result.metrics),
            context=f"file_eval_only:{aggregator}",
        )
        if int(result.metrics.confusion_matrix.sum()) != len(result.rows):
            raise MoonshotVerificationError(
                f"file-level confusion total mismatch for aggregator {aggregator!r}"
            )

    independent_path = verification_dir / "independent_recompute.json"
    independent_payload = recompute_from_saved_bundle(
        bundle_path=resolved_run_dir / "predictions.npz",
        category_mapping=replay.category_mapping,
        class_names=replay.class_names,
        saved_summary=saved_summary,
        saved_file_level=saved_file_level,
    )
    independent_path.write_text(
        json.dumps(independent_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    config = load_moonshot_run_config(resolved_run_dir / "resolved_config.yaml")
    dataset = load_base_sensor_dataset(Path(config.data_root))
    prepared = prepare_moonshot_window_splits(
        dataset,
        diff_period=config.diff_period,
        window_size=config.window_size,
        stride=config.stride,
        validation_files_per_class=config.validation_files_per_class,
        channel_set=config.channel_set,
    )
    leakage_audit = build_leakage_audit(prepared=prepared)
    leakage_audit_path = verification_dir / "leakage_audit.json"
    leakage_audit_path.write_text(
        json.dumps(leakage_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not leakage_audit["passed"]:
        raise MoonshotVerificationError("moonshot leakage audit failed")

    metric_comparison_path = verification_dir / "metric_comparison.json"
    metric_comparison_path.write_text(
        json.dumps(
            {
                "a1_pass": True,
                "a2_pass": True,
                "window_eval_only_match": True,
                "file_eval_only_match": True,
                "confusion_total": int(evaluation.metrics.confusion_matrix.sum()),
                "test_window_count": bundle.sample_count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    verification_summary_path = verification_dir / "verification_summary.json"
    verification_summary_path.write_text(
        json.dumps(
            {
                "a1_pass": True,
                "a2_pass": True,
                "a3_pass": True,
                "verification_dir": str(verification_dir.resolve()),
                "summary_metrics_path": report_paths.summary_json,
                "predictions_path": str(predictions_path.resolve()),
                "file_level_summary_paths": file_level_summary_paths,
                "leakage_audit_path": str(leakage_audit_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return MoonshotVerificationResult(
        verification_dir=verification_dir,
        summary_metrics_path=Path(report_paths.summary_json),
        confusion_matrix_path=Path(report_paths.confusion_matrix_csv),
        per_category_accuracy_path=Path(report_paths.per_category_accuracy_csv),
        predictions_path=predictions_path,
        file_level_summary_paths=file_level_summary_paths,
        metric_comparison_path=metric_comparison_path,
        independent_recompute_path=independent_path,
        leakage_audit_path=leakage_audit_path,
        verification_summary_path=verification_summary_path,
        a1_pass=True,
        a2_pass=True,
        a3_pass=True,
        window_acc_at_1=evaluation.metrics.acc_at_1,
        window_acc_at_5=evaluation.metrics.acc_at_5,
        window_precision_macro=evaluation.metrics.precision_macro,
        window_recall_macro=evaluation.metrics.recall_macro,
        window_f1_macro=evaluation.metrics.f1_macro,
    )


def recompute_from_saved_bundle(
    *,
    bundle_path: Path,
    category_mapping: dict[str, str],
    class_names: tuple[str, ...],
    saved_summary: dict[str, float],
    saved_file_level: dict[str, dict[str, float]],
) -> dict[str, Any]:
    bundle = load_window_prediction_bundle(bundle_path)
    window_metrics = compute_classification_metrics(
        class_names=class_names,
        true_labels=bundle.true_labels,
        predicted_labels=bundle.predicted_labels,
        topk_indices=bundle.topk_indices,
        category_mapping=category_mapping,
    )
    compare_metric_dicts(
        saved_summary,
        metrics_to_dict(window_metrics),
        context="window_saved_bundle",
    )
    file_level_payload: dict[str, dict[str, float]] = {}
    for aggregator in FILE_LEVEL_AGGREGATORS:
        result = aggregate_file_level_metrics(
            bundle=bundle,
            category_mapping=category_mapping,
            aggregator=aggregator,
        )
        file_level_payload[aggregator] = metrics_to_dict(result.metrics)
        compare_metric_dicts(
            saved_file_level[aggregator],
            file_level_payload[aggregator],
            context=f"file_saved_bundle:{aggregator}",
        )
    return {
        "window_metrics": metrics_to_dict(window_metrics),
        "file_level_metrics": file_level_payload,
        "match": True,
    }


def build_leakage_audit(*, prepared: Any) -> dict[str, Any]:
    train_files = sorted(record.relative_path for record in prepared.train_records)
    validation_files = sorted(record.relative_path for record in prepared.validation_records)
    test_files = sorted(record.relative_path for record in prepared.test_records)
    train_set = set(train_files)
    validation_set = set(validation_files)
    test_set = set(test_files)
    boundary_violation_count = 0
    for split_name, split in (
        ("train", prepared.standardized_train_split),
        ("validation", prepared.standardized_validation_split),
        ("test", prepared.standardized_test_split),
    ):
        split_file_set = {
            "train": train_set,
            "validation": validation_set,
            "test": test_set,
        }[split_name]
        for window in split.windows:
            if window.relative_path not in split_file_set:
                boundary_violation_count += 1
            if window.start_row < 0 or window.stop_row <= window.start_row:
                boundary_violation_count += 1
    passed = (
        not (train_set & validation_set)
        and not (train_set & test_set)
        and not (validation_set & test_set)
        and boundary_violation_count == 0
    )
    return {
        "passed": passed,
        "train_files": train_files,
        "validation_files": validation_files,
        "test_files": test_files,
        "train_validation_overlap_count": len(train_set & validation_set),
        "train_test_overlap_count": len(train_set & test_set),
        "validation_test_overlap_count": len(validation_set & test_set),
        "boundary_violation_count": boundary_violation_count,
        "standardizer_fit_source": "training_files_only",
        "train_window_count": prepared.standardized_train_split.window_count,
        "validation_window_count": prepared.standardized_validation_split.window_count,
        "test_window_count": prepared.standardized_test_split.window_count,
        "warnings": [],
    }


def load_saved_summary_metrics(summary_path: Path) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "acc@1": float(payload["acc@1"]),
        "acc@5": float(payload["acc@5"]),
        "precision_macro": float(payload["precision_macro"]),
        "recall_macro": float(payload["recall_macro"]),
        "f1_macro": float(payload["f1_macro"]),
    }


def load_saved_file_level_metrics(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    return {
        str(row["aggregator"]): {
            "acc@1": float(row["file_acc@1"]),
            "acc@5": float(row["file_acc@5"]),
            "precision_macro": float(row["file_macro_precision"]),
            "recall_macro": float(row["file_macro_recall"]),
            "f1_macro": float(row["file_macro_f1"]),
        }
        for row in rows
    }


def metrics_to_dict(metrics: ClassificationMetrics) -> dict[str, float]:
    return {
        "acc@1": float(metrics.acc_at_1),
        "acc@5": float(metrics.acc_at_5),
        "precision_macro": float(metrics.precision_macro),
        "recall_macro": float(metrics.recall_macro),
        "f1_macro": float(metrics.f1_macro),
    }


def compare_metrics(
    saved: dict[str, float],
    observed: ClassificationMetrics,
    *,
    context: str,
) -> None:
    compare_metric_dicts(saved, metrics_to_dict(observed), context=context)


def compare_metric_dicts(
    saved: dict[str, float],
    observed: dict[str, float],
    *,
    context: str,
) -> None:
    for key, saved_value in saved.items():
        observed_value = observed[key]
        if abs(saved_value - observed_value) > FLOAT_TOLERANCE:
            raise MoonshotVerificationError(
                f"{context} mismatch for {key}: saved={saved_value}, observed={observed_value}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
