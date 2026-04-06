"""verification-only exports for paper readiness and reproducibility."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smelt.evaluation import (
    ClassificationMetrics,
    compute_classification_metrics,
    load_category_mapping,
    load_prediction_bundle,
    write_dict_rows_csv,
    write_json,
)
from smelt.evaluation.diagnostics import build_run_registry_entry
from smelt.training.verify import (
    VerificationResult,
    recompute_metrics_from_saved_predictions,
    verify_exact_upstream_run,
)
from smelt.training.verify_moonshot import (
    MoonshotVerificationResult,
    compare_metric_dicts,
    verify_moonshot_run,
)

BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 17
M01C_RUN_IDS = (
    "m01c_cnn_all12_diff_locked_seed13-20260402-014020-e0a36a77",
    "m01c_cnn_all12_diff_locked_seed42-20260402-013907-e83f43b0",
    "m01c_cnn_all12_diff_locked_seed7-20260402-014020-d8e02738",
)
INVENTORY_RUN_IDS = (
    "e0_transformer_cls_w100_g25-20260401-020535-86efd02e",
    "f1_cnn_cls_w100_g25-20260401-030915-420a9ae4",
    *M01C_RUN_IDS,
    "m03_locked_seed_ensemble-20260402-130946-7c5810b9",
    "m04_locked_heterogeneous_ensemble-20260403-200848-537fd92a",
)
M03_ENSEMBLE_RUN_ID = "m03_locked_seed_ensemble-20260402-130946-7c5810b9"
M04_ENSEMBLE_RUN_ID = "m04_locked_heterogeneous_ensemble-20260403-200848-537fd92a"
EXACT_TRANSFORMER_RUN_ID = "e0_transformer_cls_w100_g25-20260401-020535-86efd02e"
EXACT_CNN_RUN_ID = "f1_cnn_cls_w100_g25-20260401-030915-420a9ae4"


class VerificationSprintError(Exception):
    """raised when verification artifacts cannot be produced safely."""


@dataclass(slots=True)
class VerificationArtifactPaths:
    inventory_csv: str
    inventory_json: str
    exact_json: str
    moonshot_json: str
    leakage_json: str
    bootstrap_csv: str
    bootstrap_json: str
    paper_baseline_csv: str
    paper_ablation_csv: str
    paper_main_results_csv: str
    paper_diversity_csv: str


@dataclass(slots=True)
class FilePredictionTable:
    class_names: tuple[str, ...]
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    topk_indices: np.ndarray
    relative_paths: tuple[str, ...]


def run_verification_sprint(
    *,
    repo_root: Path,
    run_root: Path,
    table_root: Path,
    file_level_root: Path,
    class_vocab_manifest_path: Path,
    category_map_path: Path,
    exact_regression_artifact_path: Path,
) -> VerificationArtifactPaths:
    repo_root = repo_root.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    table_root = table_root.expanduser().resolve()
    file_level_root = file_level_root.expanduser().resolve()
    class_vocab_manifest_path = class_vocab_manifest_path.expanduser().resolve()
    category_map_path = category_map_path.expanduser().resolve()
    exact_regression_artifact_path = exact_regression_artifact_path.expanduser().resolve()
    table_root.mkdir(parents=True, exist_ok=True)

    class_names = load_class_vocab(class_vocab_manifest_path)
    category_mapping = load_category_mapping(category_map_path)

    inventory_rows = build_inventory_rows(run_root)
    inventory_csv = table_root / "verification_inventory.csv"
    inventory_json = table_root / "verification_inventory.json"
    write_dict_rows_csv(inventory_csv, inventory_rows)
    write_json(inventory_json, {"rows": inventory_rows})

    exact_transformer = verify_exact_upstream_run(run_root / EXACT_TRANSFORMER_RUN_ID)
    exact_cnn = verify_exact_upstream_run(run_root / EXACT_CNN_RUN_ID)
    exact_saved_recompute = build_exact_saved_prediction_recompute(
        run_root=run_root,
        file_level_root=file_level_root,
        category_mapping=category_mapping,
    )
    exact_json = table_root / "verification_exact_upstream.json"
    exact_payload = build_exact_verification_payload(
        transformer=exact_transformer,
        cnn=exact_cnn,
        saved_recompute=exact_saved_recompute,
        regression_artifact_path=exact_regression_artifact_path,
    )
    write_json(exact_json, exact_payload)

    m01c_results = tuple(verify_moonshot_run(run_root / run_id) for run_id in M01C_RUN_IDS)
    moonshot_json = table_root / "verification_moonshot_protocol.json"
    moonshot_payload = build_moonshot_protocol_payload(
        repo_root=repo_root,
        run_root=run_root,
        category_mapping=category_mapping,
        class_names=class_names,
        m01c_results=m01c_results,
    )
    write_json(moonshot_json, moonshot_payload)

    leakage_json = table_root / "verification_leakage_selection_audit.json"
    leakage_payload = build_leakage_audit_payload(
        repo_root=repo_root,
        run_root=run_root,
        exact_payload=exact_payload,
        moonshot_payload=moonshot_payload,
    )
    if not leakage_payload["overall_pass"]:
        write_json(leakage_json, leakage_payload)
        raise VerificationSprintError("leakage/selection audit failed")
    write_json(leakage_json, leakage_payload)

    bootstrap_rows = build_bootstrap_rows(
        run_root=run_root,
        file_level_root=file_level_root,
        class_names=class_names,
        category_mapping=category_mapping,
    )
    bootstrap_csv = table_root / "verification_bootstrap_ci.csv"
    bootstrap_json = table_root / "verification_bootstrap_ci.json"
    write_dict_rows_csv(bootstrap_csv, bootstrap_rows)
    write_json(bootstrap_json, {"rows": bootstrap_rows})

    paper_baseline_csv = table_root / "paper_baseline_table.csv"
    paper_ablation_csv = table_root / "paper_ablation_table.csv"
    paper_main_results_csv = table_root / "paper_main_results_table.csv"
    paper_diversity_csv = table_root / "paper_diversity_table.csv"
    write_dict_rows_csv(paper_baseline_csv, build_paper_baseline_rows(run_root))
    write_dict_rows_csv(paper_ablation_csv, build_paper_ablation_rows(table_root))
    write_dict_rows_csv(paper_main_results_csv, build_paper_main_results_rows(table_root))
    write_dict_rows_csv(paper_diversity_csv, build_paper_diversity_rows(table_root))

    return VerificationArtifactPaths(
        inventory_csv=str(inventory_csv.resolve()),
        inventory_json=str(inventory_json.resolve()),
        exact_json=str(exact_json.resolve()),
        moonshot_json=str(moonshot_json.resolve()),
        leakage_json=str(leakage_json.resolve()),
        bootstrap_csv=str(bootstrap_csv.resolve()),
        bootstrap_json=str(bootstrap_json.resolve()),
        paper_baseline_csv=str(paper_baseline_csv.resolve()),
        paper_ablation_csv=str(paper_ablation_csv.resolve()),
        paper_main_results_csv=str(paper_main_results_csv.resolve()),
        paper_diversity_csv=str(paper_diversity_csv.resolve()),
    )


def build_inventory_rows(run_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_id in INVENTORY_RUN_IDS:
        run_dir = run_root / run_id
        entry = build_run_registry_entry(run_dir)
        rows.append(
            {
                "run_id": run_id,
                "track": entry["track"],
                "model_family": entry["model_family"],
                "channel_set": entry["channel_set"],
                "view_mode": entry["view_mode"],
                "g": entry["g"],
                "window_size": entry["window_size"],
                "stride": entry["stride"],
                "primary_metric_used_at_stage": infer_primary_metric(run_id),
                "saved_summary_metrics_path": entry["summary_metrics_path"],
                "confusion_matrix_path": entry["confusion_matrix_path"],
                "per_category_accuracy_path": entry["per_category_accuracy_path"],
                "training_history_path": entry["training_history_path"],
                "prediction_bundle_path": resolve_prediction_bundle_path(run_dir, entry),
                "checkpoint_path": entry["checkpoint_path"],
                "run_metadata_path": resolve_optional_path(run_dir / "run_metadata.json"),
            }
        )
    return rows


def infer_primary_metric(run_id: str) -> str:
    if run_id.startswith(("e0_", "f1_")):
        return "window_acc@1"
    if run_id.startswith("m01c_"):
        return "locked_file_acc@1"
    return "file_acc@1"


def resolve_prediction_bundle_path(run_dir: Path, entry: dict[str, str]) -> str:
    bundle_path = run_dir / "predictions.npz"
    if bundle_path.is_file():
        return str(bundle_path.resolve())
    return entry.get("predictions_path", "")


def resolve_optional_path(path: Path) -> str:
    return str(path.resolve()) if path.is_file() else ""


def build_exact_saved_prediction_recompute(
    *,
    run_root: Path,
    file_level_root: Path,
    category_mapping: dict[str, str],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    run_paths = {
        EXACT_TRANSFORMER_RUN_ID: file_level_root
        / EXACT_TRANSFORMER_RUN_ID
        / "window_predictions.npz",
        EXACT_CNN_RUN_ID: file_level_root / EXACT_CNN_RUN_ID / "window_predictions.npz",
    }
    for run_id, prediction_path in run_paths.items():
        if not prediction_path.is_file():
            results[run_id] = {
                "available": False,
                "match": False,
                "prediction_path": "",
            }
            continue
        metrics = recompute_metrics_from_saved_predictions(
            predictions_path=prediction_path,
            category_mapping=category_mapping,
        )
        saved_summary = load_json(run_root / run_id / "summary_metrics.json")
        compare_metric_dicts(
            {
                "acc@1": float(saved_summary["acc@1"]),
                "acc@5": float(saved_summary["acc@5"]),
                "precision_macro": float(saved_summary["precision_macro"]),
                "recall_macro": float(saved_summary["recall_macro"]),
                "f1_macro": float(saved_summary["f1_macro"]),
            },
            metrics_to_dict(metrics),
            context=f"exact_saved_bundle:{run_id}",
        )
        results[run_id] = {
            "available": True,
            "match": True,
            "prediction_path": str(prediction_path.resolve()),
            "metrics": metrics_to_dict(metrics),
        }
    return results


def build_exact_verification_payload(
    *,
    transformer: VerificationResult,
    cnn: VerificationResult,
    saved_recompute: dict[str, Any],
    regression_artifact_path: Path,
) -> dict[str, Any]:
    regression_artifact = load_json(regression_artifact_path)
    regression_checks = {
        "retained_channel_count": int(regression_artifact["retained_channel_count"]) == 6,
        "g": int(regression_artifact["differencing_period"]) == 25,
        "window_size": int(regression_artifact["window_size"]) == 100,
        "stride": int(regression_artifact["stride"]) == 50,
        "train_window_count": int(regression_artifact["train_window_count"]) == 2512,
        "test_window_count": int(regression_artifact["test_window_count"]) == 502,
    }
    payload = {
        "regression_artifact_path": str(regression_artifact_path.resolve()),
        "regression_rerun": regression_artifact,
        "regression_checks": regression_checks,
        "regression_contract": {
            "retained_channel_count": "6",
            "g": "25",
            "window_size": "100",
            "stride": "50",
            "train_window_count": "2512",
            "test_window_count": "502",
        },
        "runs": {
            EXACT_TRANSFORMER_RUN_ID: verification_result_to_dict(transformer),
            EXACT_CNN_RUN_ID: verification_result_to_dict(cnn),
        },
        "saved_prediction_recompute": saved_recompute,
    }
    payload["overall_pass"] = bool(
        transformer.a1_pass
        and transformer.a2_pass
        and transformer.a3_pass
        and cnn.a1_pass
        and cnn.a2_pass
        and cnn.a3_pass
        and all(regression_checks.values())
        and all(not row["available"] or row["match"] for row in saved_recompute.values())
    )
    return payload


def verification_result_to_dict(result: VerificationResult) -> dict[str, Any]:
    leakage_audit = load_json(result.leakage_audit_path)
    return {
        "verification_dir": str(result.verification_dir.resolve()),
        "summary_metrics_path": str(result.summary_metrics_path.resolve()),
        "confusion_matrix_path": str(result.confusion_matrix_path.resolve()),
        "per_category_accuracy_path": str(result.per_category_accuracy_path.resolve()),
        "predictions_path": str(result.predictions_path.resolve()),
        "metric_comparison_path": str(result.metric_comparison_path.resolve()),
        "leakage_audit_path": str(result.leakage_audit_path.resolve()),
        "a1_pass": result.a1_pass,
        "a2_pass": result.a2_pass,
        "a3_pass": result.a3_pass,
        "window_acc@1": result.acc_at_1,
        "window_acc@5": result.acc_at_5,
        "window_macro_precision": result.precision_macro,
        "window_macro_recall": result.recall_macro,
        "window_macro_f1": result.f1_macro,
        "test_window_count": result.test_window_count,
        "leakage_audit": leakage_audit,
    }


def build_moonshot_protocol_payload(
    *,
    repo_root: Path,
    run_root: Path,
    category_mapping: dict[str, str],
    class_names: tuple[str, ...],
    m01c_results: tuple[MoonshotVerificationResult, ...],
) -> dict[str, Any]:
    protocol_definition = load_json(repo_root / "results/tables/m01c_protocol_definition.json")
    protocol_checks = {
        "track": protocol_definition.get("track") == "moonshot-enhanced-setting",
        "channel_set": protocol_definition.get("channel_set") == "all12",
        "view_mode": protocol_definition.get("view_mode") == "diff_all12",
        "g": int(protocol_definition.get("g", 0)) == 25,
        "window_size": int(protocol_definition.get("window_size", 0)) == 100,
        "stride": int(protocol_definition.get("stride", 0)) == 50,
        "grouped_validation_policy": protocol_definition.get("grouped_validation_policy", {}).get(
            "type", ""
        )
        == "grouped_by_file_within_training_split"
        and int(
            protocol_definition.get("grouped_validation_policy", {}).get(
                "validation_files_per_class", 0
            )
        )
        == 1,
        "standardization_policy": protocol_definition.get("standardization_policy")
        == "train_only_standardization",
        "locked_aggregator_rule_source": protocol_definition.get("locked_aggregator_rule", {}).get(
            "source", ""
        )
        == "validation_only",
        "checkpoint_rule_source": protocol_definition.get("checkpoint_selection_rule", {}).get(
            "source", ""
        )
        == "validation_only",
    }
    m01c_payload = {
        run_id: moonshot_verification_to_dict(result)
        for run_id, result in zip(M01C_RUN_IDS, m01c_results, strict=True)
    }
    m04_payload = verify_saved_file_level_run(
        run_dir=run_root / M04_ENSEMBLE_RUN_ID,
        category_mapping=category_mapping,
        class_names=class_names,
    )
    m04_selection = load_json(repo_root / "results/tables/m04_ensemble_selection.json")
    m04_selection_validation_only = (
        m04_payload["run_metadata"].get("aggregator_selection_source", "") == "validation_only"
        and m04_selection.get("selection_rule", {}).get("source", "") == "validation_only"
        and is_m04_selection_validation_locked(m04_selection)
    )
    payload = {
        "protocol_definition_path": str(
            (repo_root / "results/tables/m01c_protocol_definition.json").resolve()
        ),
        "protocol_checks": protocol_checks,
        "locked_m01c_runs": m01c_payload,
        "m04_final_ensemble": m04_payload,
        "m04_selection_validation_only": m04_selection_validation_only,
        "m04_no_test_derived_aggregator_or_checkpoint_selection": m04_selection_validation_only,
    }
    payload["overall_pass"] = bool(
        all(protocol_checks.values())
        and all(
            row["a1_pass"] and row["a2_pass"] and row["a3_pass"] for row in m01c_payload.values()
        )
        and m04_payload["saved_metrics_match"]
        and m04_payload["final_reported_metrics_match"]
        and m04_selection_validation_only
    )
    return payload


def moonshot_verification_to_dict(result: MoonshotVerificationResult) -> dict[str, Any]:
    verification_summary = load_json(result.verification_summary_path)
    leakage_audit = load_json(result.leakage_audit_path)
    independent = load_json(result.independent_recompute_path)
    return {
        "verification_dir": str(result.verification_dir.resolve()),
        "summary_metrics_path": str(result.summary_metrics_path.resolve()),
        "confusion_matrix_path": str(result.confusion_matrix_path.resolve()),
        "per_category_accuracy_path": str(result.per_category_accuracy_path.resolve()),
        "predictions_path": str(result.predictions_path.resolve()),
        "file_level_summary_paths": result.file_level_summary_paths,
        "metric_comparison_path": str(result.metric_comparison_path.resolve()),
        "independent_recompute_path": str(result.independent_recompute_path.resolve()),
        "verification_summary_path": str(result.verification_summary_path.resolve()),
        "a1_pass": result.a1_pass,
        "a2_pass": result.a2_pass,
        "a3_pass": result.a3_pass,
        "window_acc@1": result.window_acc_at_1,
        "window_acc@5": result.window_acc_at_5,
        "window_macro_precision": result.window_precision_macro,
        "window_macro_recall": result.window_recall_macro,
        "window_macro_f1": result.window_f1_macro,
        "verification_summary": verification_summary,
        "independent_recompute": independent,
        "leakage_audit": leakage_audit,
    }


def verify_saved_file_level_run(
    *,
    run_dir: Path,
    category_mapping: dict[str, str],
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    metadata = load_json(run_dir / "run_metadata.json")
    summary_metrics = load_json(run_dir / "summary_metrics.json")
    file_level_comparison = load_json(run_dir / "file_level_metrics_comparison.json")
    locked_aggregator = str(metadata.get("locked_primary_aggregator", ""))
    if not locked_aggregator:
        raise VerificationSprintError(f"run is missing locked_primary_aggregator: {run_dir}")
    summary_path = run_dir / "file_level" / locked_aggregator / "summary_metrics.json"
    per_file_predictions_path = (
        run_dir / "file_level" / locked_aggregator / "per_file_predictions.csv"
    )
    recomputed = recompute_metrics_from_per_file_predictions(
        per_file_predictions_path=per_file_predictions_path,
        class_names=class_names,
        category_mapping=category_mapping,
    )
    saved_summary = load_json(summary_path)
    compare_metric_dicts(
        {
            "acc@1": float(saved_summary["acc@1"]),
            "acc@5": float(saved_summary["acc@5"]),
            "precision_macro": float(saved_summary["precision_macro"]),
            "recall_macro": float(saved_summary["recall_macro"]),
            "f1_macro": float(saved_summary["f1_macro"]),
        },
        metrics_to_dict(recomputed),
        context=f"saved_file_level:{run_dir.name}",
    )
    comparison_row = next(
        row for row in file_level_comparison["rows"] if str(row["aggregator"]) == locked_aggregator
    )
    compare_metric_dicts(
        {
            "acc@1": float(comparison_row["file_acc@1"]),
            "acc@5": float(comparison_row["file_acc@5"]),
            "precision_macro": float(comparison_row["file_macro_precision"]),
            "recall_macro": float(comparison_row["file_macro_recall"]),
            "f1_macro": float(comparison_row["file_macro_f1"]),
        },
        metrics_to_dict(recomputed),
        context=f"comparison_row:{run_dir.name}",
    )
    final_comparison_path = Path("results/tables/m04_final_comparison.json")
    reported_match = True
    if run_dir.name == M04_ENSEMBLE_RUN_ID and final_comparison_path.is_file():
        final_rows = load_json(final_comparison_path)["rows"]
        final_row = next(row for row in final_rows if row["ensemble_run_id"] == run_dir.name)
        if (
            float(final_row["ensemble_file_acc@1"]) != recomputed.acc_at_1
            or float(final_row["ensemble_file_macro_f1"]) != recomputed.f1_macro
        ):
            reported_match = False
    return {
        "run_id": run_dir.name,
        "run_metadata": metadata,
        "summary_metrics": summary_metrics,
        "locked_primary_aggregator": locked_aggregator,
        "saved_file_summary_path": str(summary_path.resolve()),
        "per_file_predictions_path": str(per_file_predictions_path.resolve()),
        "recomputed_metrics": metrics_to_dict(recomputed),
        "saved_metrics_match": True,
        "final_reported_metrics_match": reported_match,
    }


def recompute_metrics_from_per_file_predictions(
    *,
    per_file_predictions_path: Path,
    class_names: tuple[str, ...],
    category_mapping: dict[str, str],
) -> ClassificationMetrics:
    table = load_file_prediction_table(per_file_predictions_path, class_names)
    return compute_classification_metrics(
        class_names=table.class_names,
        true_labels=table.true_labels,
        predicted_labels=table.predicted_labels,
        topk_indices=table.topk_indices,
        category_mapping=category_mapping,
    )


def load_file_prediction_table(
    per_file_predictions_path: Path,
    class_names: tuple[str, ...],
) -> FilePredictionTable:
    class_to_index = {name: index for index, name in enumerate(class_names)}
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    topk_rows: list[np.ndarray] = []
    relative_paths: list[str] = []
    with per_file_predictions_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            true_class = str(row["true_class"])
            predicted_class = str(row["predicted_class"])
            true_labels.append(class_to_index[true_class])
            predicted_labels.append(class_to_index[predicted_class])
            top5_classes = tuple(json.loads(row["top5_classes"]))
            topk_rows.append(
                np.asarray(
                    [class_to_index[class_name] for class_name in top5_classes], dtype=np.int64
                )
            )
            relative_paths.append(str(row["relative_path"]))
    return FilePredictionTable(
        class_names=class_names,
        true_labels=np.asarray(true_labels, dtype=np.int64),
        predicted_labels=np.asarray(predicted_labels, dtype=np.int64),
        topk_indices=np.stack(topk_rows, axis=0),
        relative_paths=tuple(relative_paths),
    )


def build_leakage_audit_payload(
    *,
    repo_root: Path,
    run_root: Path,
    exact_payload: dict[str, Any],
    moonshot_payload: dict[str, Any],
) -> dict[str, Any]:
    run_moonshot_text = (repo_root / "src/smelt/training/run_moonshot.py").read_text(
        encoding="utf-8"
    )
    run_m04_text = (repo_root / "src/smelt/training/m04.py").read_text(encoding="utf-8")
    m04_selection = load_json(repo_root / "results/tables/m04_ensemble_selection.json")
    m03_selection = load_json(repo_root / "results/tables/m03_ensemble_selection.json")
    exact_transformer_audit = exact_payload["runs"][EXACT_TRANSFORMER_RUN_ID]["leakage_audit"]
    exact_cnn_audit = exact_payload["runs"][EXACT_CNN_RUN_ID]["leakage_audit"]
    moonshot_audits = {
        run_id: row["leakage_audit"] for run_id, row in moonshot_payload["locked_m01c_runs"].items()
    }
    checks = {
        "exact_transformer_no_overlap": exact_transformer_audit["overlap_count"] == 0,
        "exact_cnn_no_overlap": exact_cnn_audit["overlap_count"] == 0,
        "exact_transformer_no_boundary_violation": exact_transformer_audit[
            "boundary_violation_count"
        ]
        == 0,
        "exact_cnn_no_boundary_violation": exact_cnn_audit["boundary_violation_count"] == 0,
        "exact_standardizer_training_only": exact_transformer_audit["standardizer_fit_source_split"]
        == "offline_training"
        and exact_cnn_audit["standardizer_fit_source_split"] == "offline_training",
        "moonshot_no_split_overlap": all(
            audit["train_validation_overlap_count"] == 0
            and audit["train_test_overlap_count"] == 0
            and audit["validation_test_overlap_count"] == 0
            for audit in moonshot_audits.values()
        ),
        "moonshot_no_boundary_violation": all(
            audit["boundary_violation_count"] == 0 for audit in moonshot_audits.values()
        ),
        "moonshot_standardizer_training_only": all(
            audit["standardizer_fit_source"] == "training_files_only"
            for audit in moonshot_audits.values()
        ),
        "moonshot_aggregator_selection_validation_only": (
            '"aggregator_selection_source": "validation_only"' in run_moonshot_text
        ),
        "moonshot_checkpoint_selection_validation_only": (
            "validation_file_acc@1_then_validation_file_macro_f1" in run_moonshot_text
        ),
        "m03_ensemble_selection_validation_only": m03_selection["selection_rule"]["source"]
        == "validation_only",
        "m04_ensemble_selection_validation_only": m04_selection["selection_rule"]["source"]
        == "validation_only",
        "m04_selection_logic_uses_validation_metrics": "validation_result.metrics.acc_at_1"
        in run_m04_text
        and "validation_result.metrics.f1_macro" in run_m04_text,
        "m04_selection_logic_not_keyed_on_test_metrics": "test_result.metrics.acc_at_1"
        not in (extract_function_source(run_m04_text, "select_m04_ensemble_method")),
    }
    return {
        "exact_upstream_audits": {
            EXACT_TRANSFORMER_RUN_ID: exact_transformer_audit,
            EXACT_CNN_RUN_ID: exact_cnn_audit,
        },
        "moonshot_m01c_audits": moonshot_audits,
        "selection_checks": checks,
        "m03_selection_path": str(
            (repo_root / "results/tables/m03_ensemble_selection.json").resolve()
        ),
        "m04_selection_path": str(
            (repo_root / "results/tables/m04_ensemble_selection.json").resolve()
        ),
        "overall_pass": all(checks.values()),
    }


def extract_function_source(source_text: str, function_name: str) -> str:
    marker = f"def {function_name}("
    start = source_text.find(marker)
    if start < 0:
        return ""
    next_def = source_text.find("\ndef ", start + 1)
    if next_def < 0:
        return source_text[start:]
    return source_text[start:next_def]


def build_bootstrap_rows(
    *,
    run_root: Path,
    file_level_root: Path,
    class_names: tuple[str, ...],
    category_mapping: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    exact_bundle = load_prediction_bundle(
        file_level_root / EXACT_CNN_RUN_ID / "window_predictions.npz"
    )
    exact_file_table = convert_window_bundle_to_file_table(
        bundle=exact_bundle,
        aggregator="mean_probabilities",
    )
    rows.append(
        bootstrap_row_from_table(
            label="exact_upstream_cnn_file_level",
            metric_source="single_run_file_level",
            table=exact_file_table,
            class_names=exact_file_table.class_names,
            category_mapping=category_mapping,
        )
    )

    m01c_tables = [
        load_file_prediction_table(
            run_root
            / run_id
            / "file_level"
            / load_json(run_root / run_id / "run_metadata.json")["locked_primary_aggregator"]
            / "per_file_predictions.csv",
            class_names,
        )
        for run_id in M01C_RUN_IDS
    ]
    rows.append(
        bootstrap_row_from_tables_mean(
            label="m01c_locked_seed_mean_reference",
            metric_source="mean_across_locked_seed_runs",
            tables=m01c_tables,
            class_names=class_names,
            category_mapping=category_mapping,
        )
    )
    m04_table = load_file_prediction_table(
        run_root
        / M04_ENSEMBLE_RUN_ID
        / "file_level"
        / "diversity_greedy_probabilities"
        / "per_file_predictions.csv",
        class_names,
    )
    rows.append(
        bootstrap_row_from_table(
            label="m04_locked_heterogeneous_ensemble",
            metric_source="single_run_file_level",
            table=m04_table,
            class_names=class_names,
            category_mapping=category_mapping,
        )
    )
    return rows


def bootstrap_row_from_table(
    *,
    label: str,
    metric_source: str,
    table: FilePredictionTable,
    class_names: tuple[str, ...],
    category_mapping: dict[str, str],
) -> dict[str, str]:
    point_metrics = compute_classification_metrics(
        class_names=class_names,
        true_labels=table.true_labels,
        predicted_labels=table.predicted_labels,
        topk_indices=table.topk_indices,
        category_mapping=category_mapping,
    )
    point_f1 = compute_observed_macro_f1(
        true_labels=table.true_labels,
        predicted_labels=table.predicted_labels,
    )
    acc_samples, f1_samples = bootstrap_metrics(
        tables=[table],
        class_names=class_names,
        category_mapping=category_mapping,
    )
    return bootstrap_row(
        label=label,
        metric_source=metric_source,
        n_files=int(table.true_labels.shape[0]),
        point_acc=point_metrics.acc_at_1,
        point_f1=point_f1,
        acc_samples=acc_samples,
        f1_samples=f1_samples,
    )


def bootstrap_row_from_tables_mean(
    *,
    label: str,
    metric_source: str,
    tables: list[FilePredictionTable],
    class_names: tuple[str, ...],
    category_mapping: dict[str, str],
) -> dict[str, str]:
    point_acc = float(
        np.mean(
            [
                compute_classification_metrics(
                    class_names=class_names,
                    true_labels=table.true_labels,
                    predicted_labels=table.predicted_labels,
                    topk_indices=table.topk_indices,
                    category_mapping=category_mapping,
                ).acc_at_1
                for table in tables
            ]
        )
    )
    point_f1 = float(
        np.mean(
            [
                compute_observed_macro_f1(
                    true_labels=table.true_labels,
                    predicted_labels=table.predicted_labels,
                )
                for table in tables
            ]
        )
    )
    acc_samples, f1_samples = bootstrap_metrics(
        tables=tables,
        class_names=class_names,
        category_mapping=category_mapping,
    )
    return bootstrap_row(
        label=label,
        metric_source=metric_source,
        n_files=int(tables[0].true_labels.shape[0]),
        point_acc=point_acc,
        point_f1=point_f1,
        acc_samples=acc_samples,
        f1_samples=f1_samples,
    )


def bootstrap_metrics(
    *,
    tables: list[FilePredictionTable],
    class_names: tuple[str, ...],
    category_mapping: dict[str, str],
    bootstrap_seed: int = BOOTSTRAP_SEED,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(bootstrap_seed)
    n_files = int(tables[0].true_labels.shape[0])
    for table in tables[1:]:
        if int(table.true_labels.shape[0]) != n_files:
            raise VerificationSprintError("bootstrap tables must have the same file count")
    acc_samples = np.zeros(n_resamples, dtype=np.float64)
    f1_samples = np.zeros(n_resamples, dtype=np.float64)
    for sample_index in range(n_resamples):
        indices = rng.integers(0, n_files, size=n_files, endpoint=False)
        sample_acc: list[float] = []
        sample_f1: list[float] = []
        for table in tables:
            metrics = compute_classification_metrics(
                class_names=class_names,
                true_labels=table.true_labels[indices],
                predicted_labels=table.predicted_labels[indices],
                topk_indices=table.topk_indices[indices],
                category_mapping=category_mapping,
            )
            sample_acc.append(metrics.acc_at_1)
            sample_f1.append(
                compute_observed_macro_f1(
                    true_labels=table.true_labels[indices],
                    predicted_labels=table.predicted_labels[indices],
                )
            )
        acc_samples[sample_index] = float(np.mean(sample_acc))
        f1_samples[sample_index] = float(np.mean(sample_f1))
    return acc_samples, f1_samples


def bootstrap_row(
    *,
    label: str,
    metric_source: str,
    n_files: int,
    point_acc: float,
    point_f1: float,
    acc_samples: np.ndarray,
    f1_samples: np.ndarray,
) -> dict[str, str]:
    return {
        "label": label,
        "metric_source": metric_source,
        "bootstrap_seed": str(BOOTSTRAP_SEED),
        "n_bootstrap_resamples": str(BOOTSTRAP_RESAMPLES),
        "n_files": str(n_files),
        "point_file_acc@1": str(point_acc),
        "file_acc@1_ci_lower": str(float(np.quantile(acc_samples, 0.025))),
        "file_acc@1_ci_upper": str(float(np.quantile(acc_samples, 0.975))),
        "point_file_macro_f1": str(point_f1),
        "file_macro_f1_ci_lower": str(float(np.quantile(f1_samples, 0.025))),
        "file_macro_f1_ci_upper": str(float(np.quantile(f1_samples, 0.975))),
    }


def convert_window_bundle_to_file_table(
    *,
    bundle: Any,
    aggregator: str,
) -> FilePredictionTable:
    from smelt.evaluation import build_file_score_bundle

    score_bundle = build_file_score_bundle(bundle=bundle, aggregator=aggregator)
    return FilePredictionTable(
        class_names=score_bundle.class_names,
        true_labels=score_bundle.true_labels,
        predicted_labels=score_bundle.predicted_labels,
        topk_indices=score_bundle.topk_indices,
        relative_paths=score_bundle.relative_paths,
    )


def build_paper_baseline_rows(run_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_id, label in (
        (EXACT_TRANSFORMER_RUN_ID, "exact_upstream_transformer"),
        (EXACT_CNN_RUN_ID, "exact_upstream_cnn"),
    ):
        summary = load_json(run_root / run_id / "summary_metrics.json")
        rows.append(
            {
                "label": label,
                "run_id": run_id,
                "track": "exact-upstream",
                "window_acc@1": str(summary["acc@1"]),
                "window_acc@5": str(summary["acc@5"]),
                "window_macro_f1": str(summary["f1_macro"]),
            }
        )
    return rows


def build_paper_ablation_rows(table_root: Path) -> list[dict[str, str]]:
    payload = load_json(table_root / "m01b_ablation_summary.json")
    rows: list[dict[str, str]] = []
    for row in payload["rows"]:
        rows.append(
            {
                "label": str(row["channel_set"]),
                "run_id": str(row["run_id"]),
                "window_acc@1": str(row["acc@1"]),
                "window_acc@5": str(row["acc@5"]),
                "window_macro_f1": str(row["macro_f1"]),
                "best_file_aggregator": str(row["best_file_aggregator"]),
                "best_file_acc@1": str(row["best_file_acc@1"]),
                "best_file_acc@5": str(row["best_file_acc@5"]),
                "best_file_macro_f1": str(row["best_file_macro_f1"]),
            }
        )
    return rows


def build_paper_main_results_rows(table_root: Path) -> list[dict[str, str]]:
    m01c = load_json(table_root / "m01c_seed_summary.json")
    m03 = load_json(table_root / "m03_ensemble_summary.json")["rows"][0]
    m04 = load_json(table_root / "m04_final_comparison.json")["rows"][0]
    return [
        {
            "label": "m01c_locked_seed_mean",
            "run_id": json.dumps(m01c["run_ids"]),
            "window_acc@1": str(m01c["acc@1"]["mean"]),
            "window_macro_f1": str(m01c["macro_f1"]["mean"]),
            "file_acc@1": str(m01c["file_acc@1_locked"]["mean"]),
            "file_macro_f1": str(m01c["file_macro_f1_locked"]["mean"]),
        },
        {
            "label": "m03_locked_ensemble",
            "run_id": str(m03["run_id"]),
            "window_acc@1": "",
            "window_macro_f1": "",
            "file_acc@1": str(m03["file_acc@1"]),
            "file_macro_f1": str(m03["file_macro_f1"]),
        },
        {
            "label": "m04_locked_heterogeneous_ensemble",
            "run_id": str(m04["ensemble_run_id"]),
            "window_acc@1": "",
            "window_macro_f1": "",
            "file_acc@1": str(m04["ensemble_file_acc@1"]),
            "file_macro_f1": str(m04["ensemble_file_macro_f1"]),
        },
    ]


def build_paper_diversity_rows(table_root: Path) -> list[dict[str, str]]:
    search_rows = load_json(table_root / "m04_ensemble_search_summary.json")["rows"]
    output_rows: list[dict[str, str]] = []
    for row in search_rows:
        output_rows.append(
            {
                "method_name": str(row["method_name"]),
                "selected": str(row["selected"]),
                "n_members": str(row["n_members"]),
                "validation_file_acc@1": str(row["validation_file_acc@1"]),
                "validation_file_macro_f1": str(row["validation_file_macro_f1"]),
                "avg_pairwise_agreement": str(row["avg_pairwise_agreement"]),
                "avg_pairwise_correlation": str(row["avg_pairwise_correlation"]),
            }
        )
    return output_rows


def is_m04_selection_validation_locked(selection_payload: dict[str, Any]) -> bool:
    candidates = selection_payload["candidates"]
    selected_method = str(selection_payload["selected_method"])
    tie_order = {
        name: index
        for index, name in enumerate(selection_payload["selection_rule"]["final_tie_break_order"])
    }
    expected = min(
        candidates,
        key=lambda candidate: (
            -float(candidate["validation_file_acc@1"]),
            -float(candidate["validation_file_macro_f1"]),
            float(candidate["avg_pairwise_agreement"]),
            float(candidate["avg_pairwise_correlation"]),
            tie_order[str(candidate["method_name"])],
        ),
    )
    return str(expected["method_name"]) == selected_method


def load_class_vocab(path: Path) -> tuple[str, ...]:
    payload = load_json(path)
    return tuple(str(value) for value in payload["class_vocab"])


def compute_observed_macro_f1(
    *,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> float:
    observed = sorted({int(value) for value in np.asarray(true_labels, dtype=np.int64).tolist()})
    if not observed:
        return 0.0
    f1_values: list[float] = []
    for class_index in observed:
        true_positive = int(
            np.sum((true_labels == class_index) & (predicted_labels == class_index))
        )
        false_positive = int(
            np.sum((true_labels != class_index) & (predicted_labels == class_index))
        )
        false_negative = int(
            np.sum((true_labels == class_index) & (predicted_labels != class_index))
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        if precision + recall == 0.0:
            f1_values.append(0.0)
            continue
        f1_values.append((2.0 * precision * recall) / (precision + recall))
    return float(np.mean(np.asarray(f1_values, dtype=np.float64)) * 100.0)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_to_dict(metrics: ClassificationMetrics) -> dict[str, float]:
    return {
        "acc@1": float(metrics.acc_at_1),
        "acc@5": float(metrics.acc_at_5),
        "precision_macro": float(metrics.precision_macro),
        "recall_macro": float(metrics.recall_macro),
        "f1_macro": float(metrics.f1_macro),
    }
