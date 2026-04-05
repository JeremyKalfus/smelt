"""helpers for m04 heterogeneous moonshot ensemble selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from smelt.evaluation import (
    MEAN_LOGITS_AGGREGATOR,
    FileScoreBundle,
    build_file_level_result_from_predictions,
    build_file_score_bundle,
    write_json,
)

from .m03 import (
    WINDOW_FEATURE_BUNDLE_NAME,
    export_locked_moonshot_run_embeddings,
    load_locked_primary_aggregator,
    load_window_feature_bundle,
)

M04_ENSEMBLE_METHODS = (
    "mean_probabilities_all",
    "mean_logits_all",
    "weighted_probabilities_all",
    "greedy_forward_probabilities",
    "diversity_greedy_probabilities",
)
M04_ENSEMBLE_PREFERENCE = (
    "diversity_greedy_probabilities",
    "greedy_forward_probabilities",
    "weighted_probabilities_all",
    "mean_probabilities_all",
    "mean_logits_all",
)
M04_MAX_ENSEMBLE_MEMBERS = 5
M04_OPTIONAL_TRANSFORMER_PARAMETER_LIMIT = 15_000_000


class M04Error(Exception):
    """raised when the m04 ensemble bank cannot proceed safely."""


@dataclass(slots=True)
class OptionalFamilyDecision:
    family_name: str
    status: str
    reason: str
    device: str
    parameter_count: int
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "family_name": self.family_name,
            "status": self.status,
            "reason": self.reason,
            "device": self.device,
            "parameter_count": self.parameter_count,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
        }


@dataclass(slots=True)
class EnsembleCandidateResult:
    method_name: str
    member_run_ids: tuple[str, ...]
    weights: tuple[float, ...]
    validation_result: Any
    test_result: Any
    avg_pairwise_agreement: float
    avg_pairwise_correlation: float

    def to_summary_row(self, *, selected: bool) -> dict[str, str]:
        return {
            "method_name": self.method_name,
            "member_run_ids": json.dumps(list(self.member_run_ids)),
            "n_members": str(len(self.member_run_ids)),
            "weights": json.dumps([round(weight, 8) for weight in self.weights]),
            "validation_file_acc@1": str(self.validation_result.metrics.acc_at_1),
            "validation_file_acc@5": str(self.validation_result.metrics.acc_at_5),
            "validation_file_macro_f1": str(self.validation_result.metrics.f1_macro),
            "avg_pairwise_agreement": str(self.avg_pairwise_agreement),
            "avg_pairwise_correlation": str(self.avg_pairwise_correlation),
            "selected": str(selected).lower(),
        }


def decide_optional_transformer_family(
    *,
    smoke_succeeded: bool,
    smoke_payload: dict[str, Any] | None,
    parameter_limit: int = M04_OPTIONAL_TRANSFORMER_PARAMETER_LIMIT,
) -> OptionalFamilyDecision:
    payload = smoke_payload or {}
    parameter_count = int(payload.get("parameter_count", 0) or 0)
    if not smoke_succeeded:
        return OptionalFamilyDecision(
            family_name="patch_transformer",
            status="skipped",
            reason="device_smoke_failed",
            device=str(payload.get("device", "")),
            parameter_count=parameter_count,
            batch_size=int(payload.get("batch_size", 0) or 0),
            gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 0) or 0),
            effective_batch_size=int(payload.get("effective_batch_size", 0) or 0),
        )
    if parameter_count > parameter_limit:
        return OptionalFamilyDecision(
            family_name="patch_transformer",
            status="skipped",
            reason="parameter_count_above_optional_limit",
            device=str(payload.get("device", "")),
            parameter_count=parameter_count,
            batch_size=int(payload.get("batch_size", 0) or 0),
            gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 0) or 0),
            effective_batch_size=int(payload.get("effective_batch_size", 0) or 0),
        )
    return OptionalFamilyDecision(
        family_name="patch_transformer",
        status="run",
        reason="feasible",
        device=str(payload.get("device", "")),
        parameter_count=parameter_count,
        batch_size=int(payload.get("batch_size", 0) or 0),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 0) or 0),
        effective_batch_size=int(payload.get("effective_batch_size", 0) or 0),
    )


def export_model_bank_features(
    *,
    run_dirs: tuple[Path, ...],
    output_root: Path,
) -> tuple[Path, ...]:
    output_root.mkdir(parents=True, exist_ok=True)
    export_dirs: list[Path] = []
    for run_dir in sorted(run_dirs, key=lambda path: path.name):
        export_dir = output_root / run_dir.name
        bundle_path = export_dir / WINDOW_FEATURE_BUNDLE_NAME
        if not bundle_path.is_file():
            export_dir = export_locked_moonshot_run_embeddings(
                run_dir=run_dir,
                output_root=output_root,
            )
        export_dirs.append(export_dir)
    return tuple(export_dirs)


def load_locked_file_score_bundles(
    *,
    run_dirs: tuple[Path, ...],
    export_dirs: tuple[Path, ...],
    split_name: str,
) -> dict[str, FileScoreBundle]:
    if len(run_dirs) != len(export_dirs):
        raise M04Error("run_dirs and export_dirs must be the same length")
    bundles: dict[str, FileScoreBundle] = {}
    for run_dir, export_dir in zip(run_dirs, export_dirs, strict=True):
        window_bundle = load_window_feature_bundle(export_dir / WINDOW_FEATURE_BUNDLE_NAME)
        split_bundle = window_bundle.filter_split(split_name)
        locked_aggregator = load_locked_primary_aggregator(run_dir)
        bundles[run_dir.name] = build_file_score_bundle(
            bundle=split_bundle.to_prediction_bundle(),
            aggregator=locked_aggregator,
        )
    validate_aligned_file_score_bundles(tuple(bundles.values()))
    return bundles


def validate_aligned_file_score_bundles(bundles: tuple[FileScoreBundle, ...]) -> None:
    if not bundles:
        raise M04Error("at least one file score bundle is required")
    reference = bundles[0]
    for bundle in bundles[1:]:
        if bundle.class_names != reference.class_names:
            raise M04Error("file score bundles disagree on class names")
        checks = {
            "true_labels": np.array_equal(bundle.true_labels, reference.true_labels),
            "split_names": bundle.split_names == reference.split_names,
            "relative_paths": bundle.relative_paths == reference.relative_paths,
            "absolute_paths": bundle.absolute_paths == reference.absolute_paths,
            "num_windows": np.array_equal(bundle.num_windows, reference.num_windows),
        }
        mismatches = sorted(name for name, matched in checks.items() if not matched)
        if mismatches:
            raise M04Error(f"file score bundles are not aligned: {mismatches}")


def score_bundle_to_probabilities(bundle: FileScoreBundle) -> NDArray[np.float64]:
    if bundle.aggregator == MEAN_LOGITS_AGGREGATOR:
        return softmax_rows(bundle.scores)
    scores = np.asarray(bundle.scores, dtype=np.float64)
    score_sum = np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)
    return scores / score_sum


def score_bundle_to_logits_like(bundle: FileScoreBundle) -> NDArray[np.float64]:
    if bundle.aggregator == MEAN_LOGITS_AGGREGATOR:
        return np.asarray(bundle.scores, dtype=np.float64)
    return np.log(np.clip(score_bundle_to_probabilities(bundle), 1e-12, 1.0))


def build_diversity_matrix_rows(
    *,
    validation_bundles: dict[str, FileScoreBundle],
) -> list[dict[str, str]]:
    run_ids = sorted(validation_bundles)
    rows: list[dict[str, str]] = []
    for left_id in run_ids:
        for right_id in run_ids:
            left_bundle = validation_bundles[left_id]
            right_bundle = validation_bundles[right_id]
            left_probabilities = score_bundle_to_probabilities(left_bundle)
            right_probabilities = score_bundle_to_probabilities(right_bundle)
            agreement = float(
                np.mean(left_bundle.predicted_labels == right_bundle.predicted_labels)
            )
            disagreement = float(1.0 - agreement)
            correlation = compute_probability_correlation(left_probabilities, right_probabilities)
            rows.append(
                {
                    "run_id_a": left_id,
                    "run_id_b": right_id,
                    "agreement": str(agreement),
                    "disagreement": str(disagreement),
                    "probability_correlation": str(correlation),
                }
            )
    return rows


def compute_probability_correlation(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> float:
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_flat.size != right_flat.size:
        raise M04Error("probability vectors must be aligned for correlation")
    if np.allclose(left_flat, left_flat[0]) or np.allclose(right_flat, right_flat[0]):
        return 1.0 if np.allclose(left_flat, right_flat) else 0.0
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def evaluate_m04_ensemble_candidates(
    *,
    validation_bundles: dict[str, FileScoreBundle],
    test_bundles: dict[str, FileScoreBundle],
    category_mapping: dict[str, str],
    max_members: int = M04_MAX_ENSEMBLE_MEMBERS,
) -> tuple[dict[str, Any], dict[str, EnsembleCandidateResult]]:
    run_ids = tuple(sorted(validation_bundles))
    diversity_lookup = build_pairwise_diversity_lookup(validation_bundles)
    validation_single_metrics = {
        run_id: compute_file_result_from_score_bundle(
            validation_bundles[run_id],
            category_mapping=category_mapping,
            aggregator_name=validation_bundles[run_id].aggregator,
        )
        for run_id in run_ids
    }
    candidates = {
        "mean_probabilities_all": evaluate_fixed_subset_candidate(
            method_name="mean_probabilities_all",
            member_run_ids=run_ids,
            validation_bundles=validation_bundles,
            test_bundles=test_bundles,
            category_mapping=category_mapping,
            mode="probabilities",
            weights=uniform_weights(len(run_ids)),
            diversity_lookup=diversity_lookup,
        ),
        "mean_logits_all": evaluate_fixed_subset_candidate(
            method_name="mean_logits_all",
            member_run_ids=run_ids,
            validation_bundles=validation_bundles,
            test_bundles=test_bundles,
            category_mapping=category_mapping,
            mode="logits",
            weights=uniform_weights(len(run_ids)),
            diversity_lookup=diversity_lookup,
        ),
        "weighted_probabilities_all": evaluate_fixed_subset_candidate(
            method_name="weighted_probabilities_all",
            member_run_ids=run_ids,
            validation_bundles=validation_bundles,
            test_bundles=test_bundles,
            category_mapping=category_mapping,
            mode="probabilities",
            weights=normalize_weights(
                tuple(
                    float(validation_single_metrics[run_id].metrics.acc_at_1) for run_id in run_ids
                )
            ),
            diversity_lookup=diversity_lookup,
        ),
        "greedy_forward_probabilities": greedy_subset_candidate(
            method_name="greedy_forward_probabilities",
            run_ids=run_ids,
            validation_bundles=validation_bundles,
            test_bundles=test_bundles,
            category_mapping=category_mapping,
            diversity_lookup=diversity_lookup,
            max_members=max_members,
            diversity_aware=False,
        ),
        "diversity_greedy_probabilities": greedy_subset_candidate(
            method_name="diversity_greedy_probabilities",
            run_ids=run_ids,
            validation_bundles=validation_bundles,
            test_bundles=test_bundles,
            category_mapping=category_mapping,
            diversity_lookup=diversity_lookup,
            max_members=max_members,
            diversity_aware=True,
        ),
    }
    selected_method = select_m04_ensemble_method(candidates)
    selection_payload = {
        "selection_rule": {
            "source": "validation_only",
            "primary": "validation_file_acc@1",
            "tie_break": "validation_file_macro_f1",
            "secondary_tie_break": (
                "lower_average_pairwise_agreement_then_lower_probability_correlation"
            ),
            "final_tie_break_order": list(M04_ENSEMBLE_PREFERENCE),
            "max_members": max_members,
        },
        "selected_method": selected_method,
        "selected_member_run_ids": list(candidates[selected_method].member_run_ids),
        "selected_weights": list(candidates[selected_method].weights),
        "candidates": [
            candidate_to_dict(candidates[method_name]) for method_name in M04_ENSEMBLE_METHODS
        ],
    }
    return selection_payload, candidates


def candidate_to_dict(candidate: EnsembleCandidateResult) -> dict[str, Any]:
    return {
        "method_name": candidate.method_name,
        "member_run_ids": list(candidate.member_run_ids),
        "weights": list(candidate.weights),
        "validation_file_acc@1": candidate.validation_result.metrics.acc_at_1,
        "validation_file_acc@5": candidate.validation_result.metrics.acc_at_5,
        "validation_file_macro_f1": candidate.validation_result.metrics.f1_macro,
        "test_file_acc@1": candidate.test_result.metrics.acc_at_1,
        "test_file_acc@5": candidate.test_result.metrics.acc_at_5,
        "test_file_macro_f1": candidate.test_result.metrics.f1_macro,
        "avg_pairwise_agreement": candidate.avg_pairwise_agreement,
        "avg_pairwise_correlation": candidate.avg_pairwise_correlation,
    }


def select_m04_ensemble_method(candidates: dict[str, EnsembleCandidateResult]) -> str:
    tie_order = {name: index for index, name in enumerate(M04_ENSEMBLE_PREFERENCE)}
    return min(
        candidates.values(),
        key=lambda candidate: (
            -candidate.validation_result.metrics.acc_at_1,
            -candidate.validation_result.metrics.f1_macro,
            candidate.avg_pairwise_agreement,
            candidate.avg_pairwise_correlation,
            tie_order[candidate.method_name],
        ),
    ).method_name


def evaluate_fixed_subset_candidate(
    *,
    method_name: str,
    member_run_ids: tuple[str, ...],
    validation_bundles: dict[str, FileScoreBundle],
    test_bundles: dict[str, FileScoreBundle],
    category_mapping: dict[str, str],
    mode: str,
    weights: tuple[float, ...],
    diversity_lookup: dict[tuple[str, str], dict[str, float]],
) -> EnsembleCandidateResult:
    validation_result = evaluate_score_ensemble(
        bundles=tuple(validation_bundles[run_id] for run_id in member_run_ids),
        category_mapping=category_mapping,
        mode=mode,
        method_name=method_name,
        weights=weights,
    )
    test_result = evaluate_score_ensemble(
        bundles=tuple(test_bundles[run_id] for run_id in member_run_ids),
        category_mapping=category_mapping,
        mode=mode,
        method_name=method_name,
        weights=weights,
    )
    diversity = summarize_subset_diversity(member_run_ids, diversity_lookup)
    return EnsembleCandidateResult(
        method_name=method_name,
        member_run_ids=member_run_ids,
        weights=weights,
        validation_result=validation_result,
        test_result=test_result,
        avg_pairwise_agreement=diversity["avg_pairwise_agreement"],
        avg_pairwise_correlation=diversity["avg_pairwise_correlation"],
    )


def greedy_subset_candidate(
    *,
    method_name: str,
    run_ids: tuple[str, ...],
    validation_bundles: dict[str, FileScoreBundle],
    test_bundles: dict[str, FileScoreBundle],
    category_mapping: dict[str, str],
    diversity_lookup: dict[tuple[str, str], dict[str, float]],
    max_members: int,
    diversity_aware: bool,
) -> EnsembleCandidateResult:
    selected: list[str] = []
    best_candidate: EnsembleCandidateResult | None = None
    remaining = list(run_ids)
    while remaining and len(selected) < max_members:
        candidate_results: list[EnsembleCandidateResult] = []
        for run_id in remaining:
            member_run_ids = tuple(sorted((*selected, run_id)))
            candidate_results.append(
                evaluate_fixed_subset_candidate(
                    method_name=method_name,
                    member_run_ids=member_run_ids,
                    validation_bundles=validation_bundles,
                    test_bundles=test_bundles,
                    category_mapping=category_mapping,
                    mode="probabilities",
                    weights=uniform_weights(len(member_run_ids)),
                    diversity_lookup=diversity_lookup,
                )
            )
        chosen = min(
            candidate_results,
            key=lambda candidate: (
                -candidate.validation_result.metrics.acc_at_1,
                -candidate.validation_result.metrics.f1_macro,
                candidate.avg_pairwise_agreement if diversity_aware else 0.0,
                candidate.avg_pairwise_correlation if diversity_aware else 0.0,
                len(candidate.member_run_ids),
                candidate.member_run_ids,
            ),
        )
        selected = list(chosen.member_run_ids)
        remaining = [run_id for run_id in run_ids if run_id not in selected]
        if best_candidate is None or is_better_m04_candidate(
            chosen,
            best_candidate,
            diversity_aware=diversity_aware,
        ):
            best_candidate = chosen
    if best_candidate is None:
        raise M04Error(f"unable to build greedy subset candidate for {method_name}")
    return best_candidate


def is_better_m04_candidate(
    candidate: EnsembleCandidateResult,
    incumbent: EnsembleCandidateResult,
    *,
    diversity_aware: bool,
) -> bool:
    if candidate.validation_result.metrics.acc_at_1 > incumbent.validation_result.metrics.acc_at_1:
        return True
    if candidate.validation_result.metrics.acc_at_1 < incumbent.validation_result.metrics.acc_at_1:
        return False
    if candidate.validation_result.metrics.f1_macro > incumbent.validation_result.metrics.f1_macro:
        return True
    if candidate.validation_result.metrics.f1_macro < incumbent.validation_result.metrics.f1_macro:
        return False
    if diversity_aware:
        if candidate.avg_pairwise_agreement < incumbent.avg_pairwise_agreement:
            return True
        if candidate.avg_pairwise_agreement > incumbent.avg_pairwise_agreement:
            return False
        if candidate.avg_pairwise_correlation < incumbent.avg_pairwise_correlation:
            return True
        if candidate.avg_pairwise_correlation > incumbent.avg_pairwise_correlation:
            return False
    if len(candidate.member_run_ids) < len(incumbent.member_run_ids):
        return True
    if len(candidate.member_run_ids) > len(incumbent.member_run_ids):
        return False
    return candidate.member_run_ids < incumbent.member_run_ids


def evaluate_score_ensemble(
    *,
    bundles: tuple[FileScoreBundle, ...],
    category_mapping: dict[str, str],
    mode: str,
    method_name: str,
    weights: tuple[float, ...],
) -> Any:
    validate_aligned_file_score_bundles(bundles)
    if len(bundles) != len(weights):
        raise M04Error("weights must align with bundles")
    reference = bundles[0]
    if mode == "probabilities":
        stacked = np.stack([score_bundle_to_probabilities(bundle) for bundle in bundles], axis=0)
        combined_scores = weighted_average(stacked, weights)
    elif mode == "logits":
        stacked = np.stack([score_bundle_to_logits_like(bundle) for bundle in bundles], axis=0)
        combined_scores = weighted_average(stacked, weights)
    else:
        raise M04Error(f"unsupported score ensemble mode {mode!r}")
    topk = stable_topk_per_row(combined_scores)
    predicted = np.asarray(topk[:, 0], dtype=np.int64)
    return build_file_level_result_from_predictions(
        aggregator=method_name,
        class_names=reference.class_names,
        true_labels=reference.true_labels,
        predicted_labels=predicted,
        topk_indices=topk,
        split_names=reference.split_names,
        relative_paths=reference.relative_paths,
        absolute_paths=reference.absolute_paths,
        num_windows=reference.num_windows,
        category_mapping=category_mapping,
    )


def compute_file_result_from_score_bundle(
    bundle: FileScoreBundle,
    *,
    category_mapping: dict[str, str],
    aggregator_name: str,
) -> Any:
    return build_file_level_result_from_predictions(
        aggregator=aggregator_name,
        class_names=bundle.class_names,
        true_labels=bundle.true_labels,
        predicted_labels=bundle.predicted_labels,
        topk_indices=bundle.topk_indices,
        split_names=bundle.split_names,
        relative_paths=bundle.relative_paths,
        absolute_paths=bundle.absolute_paths,
        num_windows=bundle.num_windows,
        category_mapping=category_mapping,
    )


def build_pairwise_diversity_lookup(
    validation_bundles: dict[str, FileScoreBundle],
) -> dict[tuple[str, str], dict[str, float]]:
    rows = build_diversity_matrix_rows(validation_bundles=validation_bundles)
    return {
        (row["run_id_a"], row["run_id_b"]): {
            "agreement": float(row["agreement"]),
            "correlation": float(row["probability_correlation"]),
        }
        for row in rows
    }


def summarize_subset_diversity(
    member_run_ids: tuple[str, ...],
    diversity_lookup: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    if len(member_run_ids) <= 1:
        return {
            "avg_pairwise_agreement": 1.0,
            "avg_pairwise_correlation": 1.0,
        }
    agreements: list[float] = []
    correlations: list[float] = []
    for index, left_id in enumerate(member_run_ids):
        for right_id in member_run_ids[index + 1 :]:
            payload = diversity_lookup[(left_id, right_id)]
            agreements.append(payload["agreement"])
            correlations.append(payload["correlation"])
    return {
        "avg_pairwise_agreement": float(np.mean(agreements)),
        "avg_pairwise_correlation": float(np.mean(correlations)),
    }


def normalize_weights(raw_weights: tuple[float, ...]) -> tuple[float, ...]:
    weights = np.asarray(raw_weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0:
        raise M04Error("ensemble weights must have positive total mass")
    return tuple((weights / total).tolist())


def uniform_weights(size: int) -> tuple[float, ...]:
    if size <= 0:
        raise M04Error("uniform_weights requires a positive size")
    return tuple([1.0 / size] * size)


def weighted_average(
    stacked: NDArray[np.float64],
    weights: tuple[float, ...],
) -> NDArray[np.float64]:
    normalized_weights = np.asarray(normalize_weights(weights), dtype=np.float64)
    return np.tensordot(normalized_weights, stacked, axes=(0, 0))


def stable_topk_per_row(scores: NDArray[np.float64], k: int = 5) -> NDArray[np.int64]:
    return np.stack(
        [np.argsort(-row, kind="mergesort")[: min(k, row.shape[0])] for row in scores],
        axis=0,
    ).astype(np.int64, copy=False)


def softmax_rows(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def write_diversity_artifacts(
    *,
    rows: list[dict[str, str]],
    csv_path: Path,
    json_path: Path,
) -> None:
    write_csv(csv_path, rows)
    write_json(json_path, {"rows": rows})


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
