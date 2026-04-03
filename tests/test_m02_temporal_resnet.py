from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from smelt.datasets import build_base_class_vocab_manifest, load_base_sensor_dataset
from smelt.evaluation.diagnostics import build_m02_architecture_summary
from smelt.models import DeepTemporalResNet1D
from smelt.training.run_moonshot import run_moonshot_cnn, run_moonshot_device_smoke


def test_deep_temporal_resnet_forward_has_expected_shape() -> None:
    model = DeepTemporalResNet1D(
        in_channels=12,
        num_classes=50,
        stage_depths=(1, 1, 1, 1),
        stage_widths=(16, 32, 64, 96),
        stem_width=16,
        kernel_size=5,
        normalization="groupnorm",
        groupnorm_groups=8,
        se_reduction=8,
        head_dropout=0.1,
        stochastic_depth_probability=0.0,
    )
    batch = torch.randn(4, 100, 12)

    logits = model(batch)

    assert logits.shape == (4, 50)


def test_deep_temporal_resnet_one_batch_training_step_works() -> None:
    model = DeepTemporalResNet1D(
        in_channels=12,
        num_classes=50,
        stage_depths=(1, 1, 1, 1),
        stage_widths=(16, 32, 64, 96),
        stem_width=16,
        kernel_size=5,
        normalization="groupnorm",
        groupnorm_groups=8,
        se_reduction=8,
        head_dropout=0.1,
        stochastic_depth_probability=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    batch = torch.randn(2, 100, 12)
    labels = torch.tensor([0, 1], dtype=torch.long)

    optimizer.zero_grad()
    logits = model(batch)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    assert float(loss.item()) > 0.0


def test_m02_device_smoke_reports_expected_summary(tmp_path: Path) -> None:
    data_root = Path("tests/fixtures/smellnet_base_valid")
    dataset = load_base_sensor_dataset(data_root)
    category_map_path = tmp_path / "category_map.json"
    category_map_path.write_text(
        json.dumps(
            {
                name: ("nuts", "spices", "herbs", "fruits", "vegetables")[index % 5]
                for index, name in enumerate(dataset.class_vocab)
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    class_vocab_path = tmp_path / "class_vocab.json"
    class_vocab_path.write_text(
        json.dumps(build_base_class_vocab_manifest(dataset).to_dict(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    regression_path = tmp_path / "regression.json"
    regression_path.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "m02_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "track": "moonshot-enhanced-setting",
                "experiment_name": "m02_smoke",
                "data_root": str(data_root.resolve()),
                "output_root": str((tmp_path / "runs").resolve()),
                "category_map_path": str(category_map_path.resolve()),
                "class_vocab_manifest_path": str(class_vocab_path.resolve()),
                "exact_upstream_regression_path": str(regression_path.resolve()),
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 8,
                "gradient_accumulation_steps": 2,
                "lr": 0.001,
                "weight_decay": 0.0005,
                "grad_clip": 1.0,
                "label_smoothing": 0.05,
                "diff_period": 1,
                "window_size": 2,
                "stride": 1,
                "num_workers": 0,
                "validation_files_per_class": 1,
                "locked_protocol": True,
                "candidate_file_aggregators": [
                    "mean_logits",
                    "mean_probabilities",
                    "majority_vote",
                ],
                "channel_set": "all12",
                "scheduler_name": "cosine",
                "scheduler_t_max": 1,
                "scheduler_eta_min": 0.0001,
                "model_name": "deep_temporal_resnet",
                "model": {
                    "stage_depths": [1, 1, 1, 1],
                    "stage_widths": [16, 32, 64, 96],
                    "stem_width": 16,
                    "kernel_size": 5,
                    "normalization": "groupnorm",
                    "groupnorm_groups": 8,
                    "se_reduction": 8,
                    "head_dropout": 0.1,
                    "stochastic_depth_probability": 0.0,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    smoke = run_moonshot_device_smoke(config_path)

    assert smoke.device == "cpu"
    assert smoke.feature_count == 12
    assert smoke.effective_batch_size == 16
    assert smoke.parameter_count > 0
    assert smoke.model_family == "deep_temporal_resnet"


def test_m02_run_writes_architecture_summary_deterministically(tmp_path: Path) -> None:
    data_root = Path("tests/fixtures/smellnet_base_valid")
    dataset = load_base_sensor_dataset(data_root)
    category_map_path = tmp_path / "category_map.json"
    category_map_path.write_text(
        json.dumps(
            {
                name: ("nuts", "spices", "herbs", "fruits", "vegetables")[index % 5]
                for index, name in enumerate(dataset.class_vocab)
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    class_vocab_path = tmp_path / "class_vocab.json"
    class_vocab_path.write_text(
        json.dumps(build_base_class_vocab_manifest(dataset).to_dict(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    regression_path = tmp_path / "regression.json"
    regression_path.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "m02_run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "track": "moonshot-enhanced-setting",
                "experiment_name": "m02_run",
                "data_root": str(data_root.resolve()),
                "output_root": str((tmp_path / "runs").resolve()),
                "category_map_path": str(category_map_path.resolve()),
                "class_vocab_manifest_path": str(class_vocab_path.resolve()),
                "exact_upstream_regression_path": str(regression_path.resolve()),
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 64,
                "gradient_accumulation_steps": 1,
                "lr": 0.001,
                "weight_decay": 0.0005,
                "grad_clip": 1.0,
                "label_smoothing": 0.05,
                "diff_period": 1,
                "window_size": 2,
                "stride": 1,
                "num_workers": 0,
                "validation_files_per_class": 1,
                "locked_protocol": True,
                "candidate_file_aggregators": [
                    "mean_logits",
                    "mean_probabilities",
                    "majority_vote",
                ],
                "channel_set": "all12",
                "scheduler_name": "cosine",
                "scheduler_t_max": 1,
                "scheduler_eta_min": 0.0001,
                "model_name": "deep_temporal_resnet",
                "model": {
                    "stage_depths": [1, 1, 1, 1],
                    "stage_widths": [16, 32, 64, 96],
                    "stem_width": 16,
                    "kernel_size": 5,
                    "normalization": "groupnorm",
                    "groupnorm_groups": 8,
                    "se_reduction": 8,
                    "head_dropout": 0.1,
                    "stochastic_depth_probability": 0.0,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    run_dir, _window_metrics, _file_metrics, _prepared = run_moonshot_cnn(config_path)
    architecture_summary = build_m02_architecture_summary(run_dir)

    assert architecture_summary["model_family"] == "deep_temporal_resnet"
    assert architecture_summary["stage_depths"] == [1, 1, 1, 1]
    assert architecture_summary["stage_widths"] == [16, 32, 64, 96]
    assert architecture_summary["parameter_count"] > 0
