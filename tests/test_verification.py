from __future__ import annotations

from pathlib import Path

import numpy as np

from smelt.datasets import load_base_sensor_dataset
from smelt.evaluation import compute_classification_metrics
from smelt.training.run import ExactUpstreamRunConfig, TransformerModelConfig
from smelt.training.verify import (
    build_leakage_audit,
    compare_metrics_pair,
    prepare_verification_inputs,
    recompute_metrics_from_saved_predictions,
)


def make_config() -> ExactUpstreamRunConfig:
    return ExactUpstreamRunConfig(
        track="exact-upstream",
        experiment_name="verify-smoke",
        config_path="config.yaml",
        data_root="data",
        output_root="results/runs",
        category_map_path="/Users/jeremykalfus/CodingProjects/smelt/configs/exact-upstream/category_map.json",
        preprocessing_summary_path="preprocessing.json",
        class_vocab_manifest_path="class_vocab.json",
        gcms_class_map_manifest_path="gcms_map.json",
        seed=42,
        device="cpu",
        epochs=1,
        batch_size=8,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        diff_period=0,
        window_size=3,
        stride=None,
        num_workers=0,
        shuffle_train_labels=False,
        model_name="transformer",
        model=TransformerModelConfig(),
    )


def test_saved_prediction_recompute_matches_metrics(tmp_path: Path) -> None:
    category_mapping = {
        "alpha": "nuts",
        "beta": "spices",
        "gamma": "herbs",
        "delta": "fruits",
        "epsilon": "vegetables",
    }
    predictions_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        class_names=np.asarray(["alpha", "beta", "gamma", "delta", "epsilon"]),
        true_labels=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        predicted_labels=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        topk_indices=np.asarray(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 2, 3, 4],
                [2, 0, 1, 3, 4],
                [3, 0, 1, 2, 4],
                [4, 0, 1, 2, 3],
            ],
            dtype=np.int64,
        ),
        logits=np.eye(5, dtype=np.float32),
    )

    metrics = recompute_metrics_from_saved_predictions(
        predictions_path=predictions_path,
        category_mapping=category_mapping,
    )
    expected = compute_classification_metrics(
        class_names=("alpha", "beta", "gamma", "delta", "epsilon"),
        true_labels=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        predicted_labels=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        topk_indices=np.asarray(
            [
                [0, 1, 2, 3, 4],
                [1, 0, 2, 3, 4],
                [2, 0, 1, 3, 4],
                [3, 0, 1, 2, 4],
                [4, 0, 1, 2, 3],
            ],
            dtype=np.int64,
        ),
        category_mapping=category_mapping,
    )

    compare_metrics_pair(
        metrics,
        expected,
        context="test_saved_prediction_recompute",
    )

    assert metrics.acc_at_1 == 100.0
    assert metrics.acc_at_5 == 100.0


def test_prepare_verification_inputs_builds_expected_shapes() -> None:
    config = make_config()
    data_root = Path("/Users/jeremykalfus/CodingProjects/smelt/tests/fixtures/smellnet_base_valid")
    dataset = load_base_sensor_dataset(data_root)

    prepared = prepare_verification_inputs(dataset, config)

    assert prepared["class_count"] == 50
    assert prepared["input_dim"] == 6
    assert prepared["test_loader"] is not None


def test_leakage_audit_reports_no_overlap_on_valid_fixture() -> None:
    config = make_config()
    data_root = Path("/Users/jeremykalfus/CodingProjects/smelt/tests/fixtures/smellnet_base_valid")
    dataset = load_base_sensor_dataset(data_root)

    audit = build_leakage_audit(dataset, config)

    assert audit["overlap_count"] == 0
    assert audit["standardizer_fit_source_split"] == "offline_training"
    assert audit["boundary_violation_count"] == 0
    assert audit["test_derived_fit_detected"] is False
