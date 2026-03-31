"""diagnostic exports for run registries and recipe comparisons."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from smelt.models import ExactResearchInceptionClassifier
from smelt.training.run import build_classifier_model, load_run_config
from smelt.training.run_research import load_research_run_config


class DiagnosticExportError(Exception):
    """raised when diagnostic exports cannot be built safely."""


@dataclass(slots=True)
class DiagnosticArtifactPaths:
    run_registry_csv: str
    run_registry_json: str
    existing_run_summary_csv: str
    metrics_long_csv: str
    training_histories_long_csv: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_registry_csv": self.run_registry_csv,
            "run_registry_json": self.run_registry_json,
            "existing_run_summary_csv": self.existing_run_summary_csv,
            "metrics_long_csv": self.metrics_long_csv,
            "training_histories_long_csv": self.training_histories_long_csv,
        }


@dataclass(slots=True)
class RecipeDiffArtifactPaths:
    csv_path: str
    json_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "csv_path": self.csv_path,
            "json_path": self.json_path,
        }


@dataclass(slots=True)
class ComparisonArtifactPaths:
    csv_path: str
    json_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "csv_path": self.csv_path,
            "json_path": self.json_path,
        }


def export_run_registry_artifacts(
    *,
    run_root: Path,
    table_root: Path,
    figdata_root: Path,
    existing_run_ids: tuple[str, ...],
) -> DiagnosticArtifactPaths:
    run_dirs = discover_run_dirs(run_root)
    entries = [build_run_registry_entry(run_dir) for run_dir in run_dirs]
    metrics_rows = build_metrics_long_rows(entries)
    history_rows = build_training_history_rows(entries)
    existing_entries = [entry for entry in entries if entry["run_id"] in set(existing_run_ids)]

    table_root.mkdir(parents=True, exist_ok=True)
    figdata_root.mkdir(parents=True, exist_ok=True)

    run_registry_csv = table_root / "run_registry.csv"
    run_registry_json = table_root / "run_registry.json"
    existing_summary_csv = table_root / "t10b_existing_run_summary.csv"
    metrics_long_csv = figdata_root / "metrics_long.csv"
    training_histories_long_csv = figdata_root / "training_histories_long.csv"

    write_dict_rows_csv(run_registry_csv, entries)
    run_registry_json.write_text(
        json.dumps({"runs": entries}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dict_rows_csv(existing_summary_csv, existing_entries)
    write_dict_rows_csv(metrics_long_csv, metrics_rows)
    write_dict_rows_csv(training_histories_long_csv, history_rows)

    return DiagnosticArtifactPaths(
        run_registry_csv=str(run_registry_csv.resolve()),
        run_registry_json=str(run_registry_json.resolve()),
        existing_run_summary_csv=str(existing_summary_csv.resolve()),
        metrics_long_csv=str(metrics_long_csv.resolve()),
        training_histories_long_csv=str(training_histories_long_csv.resolve()),
    )


def export_recipe_diff(
    *,
    exact_config_path: Path,
    research_config_path: Path,
    output_csv: Path,
    output_json: Path,
) -> RecipeDiffArtifactPaths:
    exact_recipe = build_recipe_snapshot(exact_config_path)
    research_recipe = build_recipe_snapshot(research_config_path)
    rows = [exact_recipe, research_recipe]
    differences = build_recipe_differences(exact_recipe, research_recipe)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_dict_rows_csv(output_csv, rows)
    output_json.write_text(
        json.dumps(
            {
                "left": exact_recipe,
                "right": research_recipe,
                "differences": differences,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return RecipeDiffArtifactPaths(
        csv_path=str(output_csv.resolve()),
        json_path=str(output_json.resolve()),
    )


def export_comparison_summary(
    *,
    baseline_run_dir: Path,
    comparison_run_dir: Path,
    output_csv: Path,
    output_json: Path,
    label: str,
) -> ComparisonArtifactPaths:
    baseline_entry = build_run_registry_entry(baseline_run_dir)
    comparison_entry = build_run_registry_entry(comparison_run_dir)
    summary = {
        "label": label,
        "baseline_run_id": baseline_entry["run_id"],
        "baseline_track": baseline_entry["track"],
        "baseline_model_family": baseline_entry["model_family"],
        "baseline_view_mode": baseline_entry["view_mode"],
        "baseline_acc@1": baseline_entry["acc@1"],
        "baseline_acc@5": baseline_entry["acc@5"],
        "baseline_macro_f1": baseline_entry["macro_f1"],
        "comparison_run_id": comparison_entry["run_id"],
        "comparison_track": comparison_entry["track"],
        "comparison_model_family": comparison_entry["model_family"],
        "comparison_view_mode": comparison_entry["view_mode"],
        "comparison_acc@1": comparison_entry["acc@1"],
        "comparison_acc@5": comparison_entry["acc@5"],
        "comparison_macro_f1": comparison_entry["macro_f1"],
    }
    try:
        baseline_acc = float(baseline_entry["acc@1"])
        comparison_acc = float(comparison_entry["acc@1"])
        baseline_f1 = float(baseline_entry["macro_f1"])
        comparison_f1 = float(comparison_entry["macro_f1"])
    except ValueError:
        baseline_acc = comparison_acc = baseline_f1 = comparison_f1 = float("nan")
    summary["delta_acc@1"] = stringify(comparison_acc - baseline_acc)
    summary["delta_macro_f1"] = stringify(comparison_f1 - baseline_f1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_dict_rows_csv(output_csv, [summary])
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ComparisonArtifactPaths(
        csv_path=str(output_csv.resolve()),
        json_path=str(output_json.resolve()),
    )


def discover_run_dirs(run_root: Path) -> tuple[Path, ...]:
    if not run_root.exists():
        raise DiagnosticExportError(f"run root does not exist: {run_root}")
    return tuple(sorted(path for path in run_root.iterdir() if path.is_dir()))


def build_run_registry_entry(run_dir: Path) -> dict[str, str]:
    summary = load_json_file(run_dir / "summary_metrics.json")
    metadata = load_json_file(run_dir / "run_metadata.json")
    config = load_yaml_file(run_dir / "resolved_config.yaml")
    training_history_path = run_dir / "training_history.csv"
    research_view_manifest_path = run_dir / "research_view_manifest.json"
    research_view_manifest = (
        load_json_file(research_view_manifest_path) if research_view_manifest_path.exists() else {}
    )

    run_id = run_dir.name
    window_size = int(config.get("window_size", 0) or 0)
    configured_stride = config.get("stride")
    if configured_stride is None and window_size > 0:
        stride = window_size // 2
    else:
        stride = int(configured_stride or 0)

    track = str(metadata.get("track", config.get("track", "")))
    view_mode = resolve_view_mode(track=track, metadata=metadata, config=config)
    entry = {
        "run_id": run_id,
        "ticket_stage": infer_ticket_stage(run_id),
        "track": track,
        "model_family": infer_model_family(run_id, config),
        "view_mode": view_mode,
        "g": stringify(config.get("diff_period")),
        "window_size": stringify(config.get("window_size")),
        "stride": stringify(stride),
        "train_window_count": stringify(resolve_window_count(summary, metadata, "train")),
        "test_window_count": stringify(resolve_window_count(summary, metadata, "test")),
        "acc@1": stringify(summary.get("acc@1")),
        "acc@5": stringify(summary.get("acc@5")),
        "macro_precision": stringify(summary.get("precision_macro")),
        "macro_recall": stringify(summary.get("recall_macro")),
        "macro_f1": stringify(summary.get("f1_macro")),
        "resolved_config_path": resolve_artifact_path(run_dir / "resolved_config.yaml"),
        "summary_metrics_path": resolve_artifact_path(run_dir / "summary_metrics.json"),
        "confusion_matrix_path": resolve_artifact_path(run_dir / "confusion_matrix.csv"),
        "per_category_accuracy_path": resolve_artifact_path(run_dir / "per_category_accuracy.csv"),
        "checkpoint_path": resolve_artifact_path(run_dir / "checkpoint_final.pt"),
        "training_history_path": resolve_artifact_path(training_history_path),
        "run_metadata_path": resolve_artifact_path(run_dir / "run_metadata.json"),
        "predictions_path": resolve_artifact_path(run_dir / "predictions.npz"),
        "research_view_manifest_path": resolve_artifact_path(research_view_manifest_path),
        "feature_count": stringify(
            research_view_manifest.get(
                "feature_count",
                len(summary.get("methods", {}).get("feature_names", [])),
            )
        ),
    }
    return entry


def build_metrics_long_rows(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    metric_fields = ("acc@1", "acc@5", "macro_precision", "macro_recall", "macro_f1")
    rows: list[dict[str, str]] = []
    for entry in entries:
        for metric_name in metric_fields:
            rows.append(
                {
                    "run_id": entry["run_id"],
                    "ticket_stage": entry["ticket_stage"],
                    "track": entry["track"],
                    "model_family": entry["model_family"],
                    "view_mode": entry["view_mode"],
                    "g": entry["g"],
                    "window_size": entry["window_size"],
                    "stride": entry["stride"],
                    "metric_name": metric_name,
                    "metric_value": entry[metric_name],
                }
            )
    return rows


def build_training_history_rows(entries: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in entries:
        history_path = (
            Path(entry["training_history_path"]) if entry["training_history_path"] else None
        )
        if history_path is None or not history_path.exists():
            continue
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "run_id": entry["run_id"],
                        "ticket_stage": entry["ticket_stage"],
                        "track": entry["track"],
                        "model_family": entry["model_family"],
                        "view_mode": entry["view_mode"],
                        "g": entry["g"],
                        "window_size": entry["window_size"],
                        "stride": entry["stride"],
                        "epoch": row.get("epoch", ""),
                        "train_loss": row.get("train_loss", ""),
                        "train_acc@1": row.get("train_acc@1", ""),
                    }
                )
    return rows


def build_recipe_snapshot(config_path: Path) -> dict[str, str]:
    raw_config = load_yaml_file(config_path)
    track = str(raw_config.get("track", ""))
    if track == "exact-upstream":
        config = load_run_config(config_path)
        model = build_classifier_model(config=config, input_dim=6, num_classes=50)
        dropout = stringify(getattr(config.model, "dropout", ""))
        return {
            "config_path": str(config_path.resolve()),
            "track": config.track,
            "experiment_name": config.experiment_name,
            "model_family": config.model_name,
            "view_mode": "diff" if config.diff_period > 0 else "raw",
            "feature_count": "6",
            "optimizer": "adam",
            "lr": stringify(config.lr),
            "weight_decay": stringify(config.weight_decay),
            "batch_size": stringify(config.batch_size),
            "epochs": stringify(config.epochs),
            "scheduler": "",
            "early_stopping": "",
            "loss_function": "cross_entropy",
            "grad_clip": stringify(config.grad_clip),
            "regularization_dropout": dropout,
            "augmentation": "",
            "parameter_count": stringify(count_trainable_parameters(model)),
        }
    if track == "research-extension":
        config = load_research_run_config(config_path)
        input_dim = 12 if config.view_mode == "fused_raw_diff" else 6
        model = ExactResearchInceptionClassifier(
            input_dim=input_dim,
            num_classes=50,
            stem_channels=config.model.stem_channels,
            branch_channels=config.model.branch_channels,
            bottleneck_channels=config.model.bottleneck_channels,
            num_blocks=config.model.num_blocks,
            residual_interval=config.model.residual_interval,
            activation_name=config.model.activation_name,
            dropout=config.model.dropout,
            head_hidden_dim=config.model.head_hidden_dim,
        )
        return {
            "config_path": str(config_path.resolve()),
            "track": config.track,
            "experiment_name": config.experiment_name,
            "model_family": "inception",
            "view_mode": config.view_mode,
            "feature_count": stringify(input_dim),
            "optimizer": "adam",
            "lr": stringify(config.lr),
            "weight_decay": stringify(config.weight_decay),
            "batch_size": stringify(config.batch_size),
            "epochs": stringify(config.epochs),
            "scheduler": "",
            "early_stopping": "",
            "loss_function": "cross_entropy",
            "grad_clip": stringify(config.grad_clip),
            "regularization_dropout": stringify(config.model.dropout),
            "augmentation": "",
            "parameter_count": stringify(count_trainable_parameters(model)),
        }
    raise DiagnosticExportError(f"unsupported track for recipe snapshot: {track!r}")


def build_recipe_differences(
    left: dict[str, str],
    right: dict[str, str],
) -> dict[str, dict[str, str]]:
    differences: dict[str, dict[str, str]] = {}
    for key in sorted(set(left) | set(right)):
        left_value = left.get(key, "")
        right_value = right.get(key, "")
        if left_value == right_value:
            continue
        differences[key] = {"left": left_value, "right": right_value}
    return differences


def infer_ticket_stage(run_id: str) -> str:
    if run_id.startswith("t11_"):
        return "t11"
    if run_id.startswith("e0_transformer"):
        return "t07"
    if run_id.startswith("f1_cnn"):
        return "t08"
    if run_id.startswith("e1_inception"):
        return "t10"
    if run_id.startswith("t10b_"):
        return "t10b"
    return ""


def compare_research_supervised_recipe_compatibility(
    baseline_config_path: Path,
    comparison_config_path: Path,
) -> dict[str, object]:
    baseline = load_yaml_file(baseline_config_path)
    comparison = load_yaml_file(comparison_config_path)
    allowed_differences = {
        "experiment_name",
        "config_path",
        "output_root",
        "mode",
        "pretrained_checkpoint_path",
        "gcms_class_map_manifest_path",
        "projection_dim",
        "gcms_hidden_dim",
        "temperature",
    }
    allowed_model_differences: set[str] = set()
    material_mismatches: dict[str, dict[str, str]] = {}
    for key in sorted(set(baseline) | set(comparison)):
        if key in allowed_differences:
            continue
        baseline_value = baseline.get(key)
        comparison_value = comparison.get(key)
        if key == "model":
            baseline_model = baseline_value if isinstance(baseline_value, dict) else {}
            comparison_model = comparison_value if isinstance(comparison_value, dict) else {}
            for model_key in sorted(set(baseline_model) | set(comparison_model)):
                if model_key in allowed_model_differences:
                    continue
                if baseline_model.get(model_key) != comparison_model.get(model_key):
                    material_mismatches[f"model.{model_key}"] = {
                        "baseline": stringify(baseline_model.get(model_key)),
                        "comparison": stringify(comparison_model.get(model_key)),
                    }
            continue
        if baseline_value != comparison_value:
            material_mismatches[key] = {
                "baseline": stringify(baseline_value),
                "comparison": stringify(comparison_value),
            }
    return {
        "compatible": not material_mismatches,
        "material_mismatches": material_mismatches,
        "baseline_config_path": str(baseline_config_path.resolve()),
        "comparison_config_path": str(comparison_config_path.resolve()),
    }


def infer_model_family(run_id: str, config: dict[str, Any]) -> str:
    model_name = config.get("model_name")
    if isinstance(model_name, str) and model_name:
        return model_name
    experiment_name = str(config.get("experiment_name", run_id))
    if "inception" in experiment_name:
        return "inception"
    if "transformer" in experiment_name:
        return "transformer"
    if "cnn" in experiment_name:
        return "cnn"
    return ""


def resolve_view_mode(
    *,
    track: str,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> str:
    if "view_mode" in metadata:
        return str(metadata["view_mode"])
    if "view_mode" in config:
        return str(config["view_mode"])
    if track == "exact-upstream":
        diff_period = int(config.get("diff_period", 0) or 0)
        return "diff" if diff_period > 0 else "raw"
    return ""


def resolve_window_count(
    summary: dict[str, Any],
    metadata: dict[str, Any],
    split_name: str,
) -> Any:
    methods = summary.get("methods", {})
    key = f"{split_name}_window_count"
    if key in methods:
        return methods[key]
    window_counts = metadata.get("window_counts", {})
    return window_counts.get(split_name, "")


def resolve_artifact_path(path: Path) -> str:
    if path.exists():
        return str(path.resolve())
    return ""


def count_trainable_parameters(model: Any) -> int:
    return int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )


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


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DiagnosticExportError(f"unable to read json file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DiagnosticExportError(f"invalid json file: {path}") from exc
    if not isinstance(payload, dict):
        raise DiagnosticExportError(f"expected json object in {path}")
    return payload


def load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise DiagnosticExportError(f"unable to read yaml file: {path}") from exc
    if not isinstance(payload, dict):
        raise DiagnosticExportError(f"expected yaml mapping in {path}")
    return payload


def stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
