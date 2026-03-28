from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from smelt.evaluation import export_classification_report, load_category_mapping
from smelt.models import ExactUpstreamTransformerClassifier
from smelt.training.run import (
    ExactUpstreamRunConfig,
    TransformerModelConfig,
    build_dataloader,
    collect_evaluation_outputs,
    evaluate_classifier,
    load_run_config,
    maybe_shuffle_train_labels,
    resolve_device,
    train_classifier,
)


def test_transformer_forward_pass_has_expected_shape() -> None:
    model = ExactUpstreamTransformerClassifier(
        input_dim=6,
        num_classes=4,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    )
    batch = torch.randn(3, 100, 6)

    logits = model(batch)

    assert logits.shape == (3, 4)


def test_training_loop_runs_one_batch_without_runtime_errors() -> None:
    model = ExactUpstreamTransformerClassifier(
        input_dim=6,
        num_classes=3,
        model_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
    )
    windows = np.random.default_rng(0).normal(size=(4, 20, 6)).astype(np.float32)
    labels = np.asarray([0, 1, 2, 0], dtype=np.int64)
    loader = build_dataloader(windows, labels, batch_size=4, shuffle=False, num_workers=0)

    history = train_classifier(
        model=model,
        train_loader=loader,
        device=torch.device("cpu"),
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
    )

    assert len(history) == 1
    assert history[0].epoch == 1


def test_evaluation_loop_writes_report_artifacts(tmp_path: Path) -> None:
    model = ExactUpstreamTransformerClassifier(
        input_dim=6,
        num_classes=5,
        model_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
    )
    windows = np.random.default_rng(1).normal(size=(5, 20, 6)).astype(np.float32)
    labels = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    loader = build_dataloader(windows, labels, batch_size=5, shuffle=False, num_workers=0)
    category_map_path = tmp_path / "category_map.json"
    category_map_path.write_text(
        json.dumps(
            {
                "alpha": "nuts",
                "beta": "spices",
                "gamma": "herbs",
                "delta": "fruits",
                "epsilon": "vegetables",
            }
        ),
        encoding="utf-8",
    )

    metrics = evaluate_classifier(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        class_names=("alpha", "beta", "gamma", "delta", "epsilon"),
        category_mapping=load_category_mapping(category_map_path),
    )
    report_paths = export_classification_report(
        output_root=tmp_path,
        run_name="smoke_run",
        metrics=metrics,
        methods_summary={"mode": "test"},
    )

    assert metrics.confusion_matrix.shape == (5, 5)
    assert len(metrics.per_category) == 5
    assert Path(report_paths.summary_json).is_file()
    assert Path(report_paths.confusion_matrix_csv).is_file()
    assert Path(report_paths.per_category_accuracy_csv).is_file()


def test_collect_evaluation_outputs_returns_predictions() -> None:
    model = ExactUpstreamTransformerClassifier(
        input_dim=6,
        num_classes=5,
        model_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
    )
    windows = np.random.default_rng(2).normal(size=(5, 20, 6)).astype(np.float32)
    labels = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    loader = build_dataloader(windows, labels, batch_size=5, shuffle=False, num_workers=0)
    outputs = collect_evaluation_outputs(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        class_names=("alpha", "beta", "gamma", "delta", "epsilon"),
        category_mapping={
            "alpha": "nuts",
            "beta": "spices",
            "gamma": "herbs",
            "delta": "fruits",
            "epsilon": "vegetables",
        },
    )

    assert outputs.true_labels.shape == (5,)
    assert outputs.predicted_labels.shape == (5,)
    assert outputs.topk_indices.shape == (5, 5)
    assert outputs.logits.shape == (5, 5)


def test_maybe_shuffle_train_labels_is_deterministic_when_enabled() -> None:
    labels = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    config = ExactUpstreamRunConfig(
        track="exact-upstream",
        experiment_name="shuffle-smoke",
        config_path="config.yaml",
        data_root="data",
        output_root="results/runs",
        category_map_path="category_map.json",
        preprocessing_summary_path="preprocessing.json",
        class_vocab_manifest_path="class_vocab.json",
        gcms_class_map_manifest_path="gcms_map.json",
        seed=7,
        device="cpu",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        diff_period=25,
        window_size=100,
        stride=None,
        num_workers=0,
        shuffle_train_labels=True,
        model_name="transformer",
        model=TransformerModelConfig(),
    )

    shuffled = maybe_shuffle_train_labels(labels, config)
    reshuffled = maybe_shuffle_train_labels(labels, config)

    assert not np.array_equal(shuffled, labels)
    assert np.array_equal(shuffled, reshuffled)


def test_run_config_resolves_exact_upstream_base_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "run.yaml"
    data_root = tmp_path / "data_root"
    refs_root = tmp_path / "refs"
    refs_root.mkdir()
    (refs_root / "preprocessing.json").write_text("{}", encoding="utf-8")
    (refs_root / "class_vocab.json").write_text('{"class_vocab": ["a", "b"]}', encoding="utf-8")
    (refs_root / "gcms_map.json").write_text("{}", encoding="utf-8")
    (refs_root / "category_map.json").write_text('{"a": "nuts", "b": "spices"}', encoding="utf-8")
    monkeypatch.setenv("SMELT_DATA_ROOT", str(data_root))
    config_path.write_text(
        "\n".join(
            [
                "track: exact-upstream",
                "experiment_name: smoke",
                "data_root: ${SMELT_DATA_ROOT}",
                "output_root: results/runs",
                f"category_map_path: {refs_root / 'category_map.json'}",
                f"preprocessing_summary_path: {refs_root / 'preprocessing.json'}",
                f"class_vocab_manifest_path: {refs_root / 'class_vocab.json'}",
                f"gcms_class_map_manifest_path: {refs_root / 'gcms_map.json'}",
                "seed: 1",
                "device: cpu",
                "epochs: 1",
                "batch_size: 2",
                "lr: 0.001",
                "weight_decay: 0.0",
                "grad_clip: 1.0",
                "diff_period: 25",
                "window_size: 100",
                "stride: null",
                "num_workers: 0",
                "model_name: transformer",
                "model:",
                "  model_dim: 32",
                "  num_heads: 4",
                "  num_layers: 1",
                "  dropout: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.data_root == str(data_root)
    assert config.track == "exact-upstream"
    assert config.window_size == 100
    assert resolve_device("cpu").type == "cpu"
