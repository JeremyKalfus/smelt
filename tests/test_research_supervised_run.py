from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from smelt.datasets import FUSED_RAW_DIFF_VIEW, load_base_sensor_dataset
from smelt.models import ExactResearchInceptionClassifier
from smelt.training.run import build_dataloader, train_classifier
from smelt.training.run_research import (
    ResearchInceptionModelConfig,
    ResearchRunConfig,
    load_research_run_config,
    prepare_research_window_tensors,
)


def test_research_model_forward_pass_has_expected_shape() -> None:
    model = ExactResearchInceptionClassifier(
        input_dim=12,
        num_classes=50,
        stem_channels=32,
        branch_channels=16,
        bottleneck_channels=16,
        num_blocks=3,
        residual_interval=3,
        activation_name="gelu",
        dropout=0.1,
        head_hidden_dim=64,
    )
    batch = torch.randn(4, 100, 12)

    logits = model(batch)

    assert logits.shape == (4, 50)
    assert model.output_dim == 50


def test_research_model_can_train_one_batch_without_runtime_errors() -> None:
    config = ResearchRunConfig(
        track="research-extension",
        experiment_name="research-smoke",
        config_path="config.yaml",
        data_root="data",
        output_root="results/runs",
        category_map_path="category_map.json",
        class_vocab_manifest_path="class_vocab.json",
        exact_upstream_regression_path="regression.json",
        seed=7,
        device="cpu",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        diff_period=25,
        window_size=100,
        stride=50,
        num_workers=0,
        view_mode=FUSED_RAW_DIFF_VIEW,
        shuffle_train_labels=False,
        model=ResearchInceptionModelConfig(
            stem_channels=32,
            branch_channels=16,
            bottleneck_channels=16,
            num_blocks=3,
            residual_interval=3,
            activation_name="gelu",
            dropout=0.1,
            head_hidden_dim=64,
        ),
    )
    model = ExactResearchInceptionClassifier(
        input_dim=12,
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
    windows = np.random.default_rng(0).normal(size=(4, 100, 12)).astype(np.float32)
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    loader = build_dataloader(windows, labels, batch_size=4, shuffle=False, num_workers=0)

    history = train_classifier(
        model=model,
        train_loader=loader,
        device=torch.device("cpu"),
        epochs=1,
        lr=config.lr,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
    )

    assert len(history) == 1
    assert history[0].epoch == 1


def test_research_run_config_resolves_fused_view_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "research.yaml"
    data_root = tmp_path / "data_root"
    refs_root = tmp_path / "refs"
    refs_root.mkdir()
    (refs_root / "class_vocab.json").write_text('{"class_vocab": ["a", "b"]}', encoding="utf-8")
    (refs_root / "regression.json").write_text("{}", encoding="utf-8")
    (refs_root / "category_map.json").write_text('{"a": "nuts", "b": "spices"}', encoding="utf-8")
    monkeypatch.setenv("SMELT_DATA_ROOT", str(data_root))
    config_path.write_text(
        "\n".join(
            [
                "track: research-extension",
                "experiment_name: research_smoke",
                "data_root: ${SMELT_DATA_ROOT}",
                "output_root: results/runs",
                f"category_map_path: {refs_root / 'category_map.json'}",
                f"class_vocab_manifest_path: {refs_root / 'class_vocab.json'}",
                f"exact_upstream_regression_path: {refs_root / 'regression.json'}",
                "seed: 1",
                "device: cpu",
                "epochs: 1",
                "batch_size: 2",
                "lr: 0.001",
                "weight_decay: 0.0",
                "grad_clip: 1.0",
                "diff_period: 25",
                "window_size: 100",
                "stride: 50",
                "num_workers: 0",
                "view_mode: fused_raw_diff",
                "model:",
                "  stem_channels: 32",
                "  branch_channels: 16",
                "  bottleneck_channels: 16",
                "  num_blocks: 3",
                "  residual_interval: 3",
                "  activation_name: gelu",
                "  dropout: 0.1",
                "  head_hidden_dim: 64",
            ]
        ),
        encoding="utf-8",
    )

    config = load_research_run_config(config_path)

    assert config.data_root == str(data_root)
    assert config.track == "research-extension"
    assert config.view_mode == FUSED_RAW_DIFF_VIEW


def test_prepare_research_window_tensors_builds_twelve_feature_windows() -> None:
    data_root = Path("tests/fixtures/smellnet_base_valid")
    dataset = load_base_sensor_dataset(data_root)
    config = ResearchRunConfig(
        track="research-extension",
        experiment_name="research-prepare",
        config_path="config.yaml",
        data_root=str(data_root.resolve()),
        output_root="results/runs",
        category_map_path="configs/exact-upstream/category_map.json",
        class_vocab_manifest_path="artifacts/manifests/base_class_vocab.json",
        exact_upstream_regression_path="artifacts/methods/t09_exact_upstream_regression_smoke.json",
        seed=42,
        device="cpu",
        epochs=1,
        batch_size=8,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        diff_period=1,
        window_size=2,
        stride=1,
        num_workers=0,
        view_mode=FUSED_RAW_DIFF_VIEW,
        shuffle_train_labels=False,
        model=ResearchInceptionModelConfig(
            stem_channels=32,
            branch_channels=16,
            bottleneck_channels=16,
            num_blocks=3,
            residual_interval=3,
            activation_name="gelu",
            dropout=0.1,
            head_hidden_dim=64,
        ),
    )

    prepared = prepare_research_window_tensors(dataset, config)

    assert prepared.train_windows.shape[2] == 12
    assert prepared.test_windows.shape[2] == 12
    assert len(prepared.class_names) == 50
    assert prepared.train_window_count > 0
