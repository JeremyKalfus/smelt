"""gc-ms pretrain coverage for t11."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from smelt.datasets import load_research_gcms_anchor_set
from smelt.evaluation.diagnostics import compare_research_supervised_recipe_compatibility
from smelt.models import (
    ExactResearchInceptionClassifier,
    ResearchGcmsPretrainModel,
    extract_inception_encoder_state_dict,
)
from smelt.training.run_research_gcms import load_pretrained_sensor_encoder


def test_load_research_gcms_anchor_set_resolves_explicit_mapping(tmp_path: Path) -> None:
    gcms_csv = tmp_path / "gcms.csv"
    gcms_csv.write_text(
        "food_name,C,H\napple,1.0,2.0\nbanana,3.0,4.0\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "gcms_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "resolved_gcms_source_path": str(gcms_csv.resolve()),
                "gcms_feature_columns": ["C", "H"],
                "class_vocab": ["apple", "banana"],
                "mapping_entries": [
                    {"class_name": "apple", "anchor_label": "apple", "source_row_index": 0},
                    {"class_name": "banana", "anchor_label": "banana", "source_row_index": 1},
                ],
                "validation": {"passed": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    anchor_set = load_research_gcms_anchor_set(manifest, class_names=("apple", "banana"))
    assert anchor_set.anchor_labels == ("apple", "banana")
    assert anchor_set.feature_columns == ("C", "H")
    assert anchor_set.feature_matrix.shape == (2, 2)
    assert np.allclose(anchor_set.feature_matrix[1], np.asarray([3.0, 4.0], dtype=np.float32))


def test_gcms_pretrain_batch_forward_backward_runs() -> None:
    sensor_backbone = ExactResearchInceptionClassifier(
        input_dim=6,
        num_classes=2,
        stem_channels=16,
        branch_channels=8,
        bottleneck_channels=8,
        num_blocks=3,
        residual_interval=3,
        head_hidden_dim=16,
    )
    model = ResearchGcmsPretrainModel(
        sensor_backbone=sensor_backbone,
        gcms_feature_count=3,
        projection_dim=8,
        gcms_hidden_dim=8,
        temperature=0.1,
    )
    model.set_anchor_features(torch.randn(2, 3))
    batch_x = torch.randn(4, 100, 6)
    batch_y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    logits = model(batch_x)
    assert logits.shape == (4, 2)
    loss = torch.nn.CrossEntropyLoss()(logits, batch_y)
    loss.backward()
    grad_sum = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert grad_sum > 0


def test_finetune_path_loads_pretrained_encoder_weights(tmp_path: Path) -> None:
    source_model = ExactResearchInceptionClassifier(
        input_dim=6,
        num_classes=2,
        stem_channels=16,
        branch_channels=8,
        bottleneck_channels=8,
        num_blocks=3,
        residual_interval=3,
        head_hidden_dim=16,
    )
    with torch.no_grad():
        first_weight = next(source_model.stem.parameters())
        first_weight.fill_(0.25)
    checkpoint_path = tmp_path / "pretrain.pt"
    torch.save(
        {"sensor_encoder_state_dict": extract_inception_encoder_state_dict(source_model)},
        checkpoint_path,
    )
    target_model = ExactResearchInceptionClassifier(
        input_dim=6,
        num_classes=2,
        stem_channels=16,
        branch_channels=8,
        bottleneck_channels=8,
        num_blocks=3,
        residual_interval=3,
        head_hidden_dim=16,
    )
    load_pretrained_sensor_encoder(target_model, checkpoint_path)
    target_first_weight = next(target_model.stem.parameters())
    assert torch.allclose(target_first_weight, torch.full_like(target_first_weight, 0.25))


def test_t11_can_fairly_reuse_d2_diff_baseline() -> None:
    compatibility = compare_research_supervised_recipe_compatibility(
        Path("configs/research-extension/t10b_inception_diff_supervised_w100_g25.yaml"),
        Path("configs/research-extension/t11_f1_inception_diff_gcms_finetune_w100_g25.yaml"),
    )
    assert compatibility["compatible"] is True
    assert compatibility["material_mismatches"] == {}
