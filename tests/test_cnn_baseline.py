from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from smelt.models import ExactUpstreamCnnClassifier
from smelt.training.run import (
    CnnModelConfig,
    ExactUpstreamRunConfig,
    build_classifier_model,
    build_dataloader,
    load_run_config,
    train_classifier,
)


def test_cnn_forward_pass_has_expected_shape() -> None:
    model = ExactUpstreamCnnClassifier(
        in_channels=6,
        num_classes=4,
        channels=(16, 32),
        kernel_size=5,
        dropout=0.2,
        use_batchnorm=True,
    )
    batch = torch.randn(3, 100, 6)

    logits = model(batch)

    assert logits.shape == (3, 4)


def test_cnn_training_loop_runs_one_batch_without_runtime_errors() -> None:
    config = ExactUpstreamRunConfig(
        track="exact-upstream",
        experiment_name="cnn-smoke",
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
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        diff_period=25,
        window_size=100,
        stride=None,
        num_workers=0,
        shuffle_train_labels=False,
        model_name="cnn",
        model=CnnModelConfig(channels=(16, 32), kernel_size=5, dropout=0.2),
    )
    model = build_classifier_model(config=config, input_dim=6, num_classes=3)
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


def test_cnn_run_config_loads_explicit_model_name(tmp_path: Path) -> None:
    config_path = tmp_path / "cnn.yaml"
    config_path.write_text(
        "\n".join(
            [
                "track: exact-upstream",
                "experiment_name: cnn_smoke",
                "data_root: data",
                "output_root: results/runs",
                "category_map_path: category_map.json",
                "preprocessing_summary_path: preprocessing.json",
                "class_vocab_manifest_path: class_vocab.json",
                "gcms_class_map_manifest_path: gcms_map.json",
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
                "model_name: cnn",
                "model:",
                "  channels: [16, 32]",
                "  kernel_size: 5",
                "  dropout: 0.2",
                "  use_batchnorm: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_run_config(config_path)

    assert config.model_name == "cnn"
    assert isinstance(config.model, CnnModelConfig)
    assert config.model.channels == (16, 32)
