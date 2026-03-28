from __future__ import annotations

from pathlib import Path

import numpy as np

from smelt.datasets import load_base_sensor_dataset
from smelt.evaluation import compute_classification_metrics
from smelt.training.run import ExactUpstreamRunConfig, TransformerModelConfig
from smelt.training.verify import (
    build_leakage_audit,
    compare_metrics_pair,
    compare_saved_metrics,
    recompute_metrics_from_saved_predictions,
)


def test_compare_saved_metrics_accepts_identical_values() -> None:
    metrics = compute_classification_metrics(
        class_names=("alpha", "beta"),
        true_labels=np.asarray([0, 1], dtype=np.int64),
        predicted_labels=np.asarray([0, 1], dtype=np.int64),
        topk_indices=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        category_mapping={"alpha": "nuts", "beta": "spices"},
    )

    compare_saved_metrics(
        {
            "acc@1": metrics.acc_at_1,
            "acc@5": metrics.acc_at_5,
            "precision_macro": metrics.precision_macro,
            "recall_macro": metrics.recall_macro,
            "f1_macro": metrics.f1_macro,
        },
        metrics,
    )


def test_recompute_metrics_from_saved_predictions_round_trips(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        class_names=np.asarray(["alpha", "beta"]),
        true_labels=np.asarray([0, 1], dtype=np.int64),
        predicted_labels=np.asarray([0, 1], dtype=np.int64),
        topk_indices=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        logits=np.asarray([[4.0, 1.0], [1.0, 4.0]], dtype=np.float32),
    )

    metrics = recompute_metrics_from_saved_predictions(
        predictions_path=predictions_path,
        category_mapping={"alpha": "nuts", "beta": "spices"},
    )
    expected = compute_classification_metrics(
        class_names=("alpha", "beta"),
        true_labels=np.asarray([0, 1], dtype=np.int64),
        predicted_labels=np.asarray([0, 1], dtype=np.int64),
        topk_indices=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        category_mapping={"alpha": "nuts", "beta": "spices"},
    )

    compare_metrics_pair(metrics, expected, context="round_trip")


def test_build_leakage_audit_reports_no_overlap_for_valid_fixture() -> None:
    data_root = Path("tests/fixtures/smellnet_base_valid")
    dataset = load_base_sensor_dataset(data_root)
    config = ExactUpstreamRunConfig(
        track="exact-upstream",
        experiment_name="verify",
        config_path=str(Path("tests/config.yaml").resolve()),
        data_root=str(data_root.resolve()),
        output_root="results/runs",
        category_map_path="configs/exact-upstream/category_map.json",
        preprocessing_summary_path="artifacts/methods/t04_preprocessing_smoke.json",
        class_vocab_manifest_path="artifacts/manifests/base_class_vocab.json",
        gcms_class_map_manifest_path="artifacts/manifests/gcms_class_map.json",
        seed=42,
        device="cpu",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        diff_period=0,
        window_size=3,
        stride=1,
        num_workers=0,
        shuffle_train_labels=False,
        model_name="transformer",
        model=TransformerModelConfig(model_dim=32, num_heads=4, num_layers=1, dropout=0.1),
    )

    audit = build_leakage_audit(dataset, config)

    assert audit["overlap_count"] == 0
    assert audit["standardizer_fit_source_split"] == "offline_training"
    assert audit["test_derived_fit_detected"] is False
