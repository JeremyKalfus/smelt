from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from smelt.datasets import build_base_class_vocab_manifest, load_base_sensor_dataset
from smelt.evaluation.diagnostics import build_m03_seed_summary
from smelt.models import AttentionDeepSetsClassifier
from smelt.training.export_m03_embeddings import main as export_m03_embeddings_main
from smelt.training.m03 import (
    WINDOW_FEATURE_BUNDLE_NAME,
    load_window_feature_bundle,
    select_ensemble_method,
    select_primary_encoder_run,
)
from smelt.training.run_m03_file_model import (
    FileCheckpointSelection,
    is_better_file_checkpoint,
)
from smelt.training.run_moonshot import run_moonshot_cnn


def test_embedding_export_is_deterministic_and_file_aware(tmp_path: Path) -> None:
    run_dir = build_locked_moonshot_fixture_run(tmp_path)
    export_root_a = tmp_path / "embeddings_a"
    export_root_b = tmp_path / "embeddings_b"
    selection_a = tmp_path / "selection_a.json"
    selection_b = tmp_path / "selection_b.json"

    export_m03_embeddings_main(
        [
            "--run-dir",
            str(run_dir),
            "--output-root",
            str(export_root_a),
            "--primary-selection-json",
            str(selection_a),
        ]
    )
    export_m03_embeddings_main(
        [
            "--run-dir",
            str(run_dir),
            "--output-root",
            str(export_root_b),
            "--primary-selection-json",
            str(selection_b),
        ]
    )

    bundle_a = load_window_feature_bundle(export_root_a / run_dir.name / WINDOW_FEATURE_BUNDLE_NAME)
    bundle_b = load_window_feature_bundle(export_root_b / run_dir.name / WINDOW_FEATURE_BUNDLE_NAME)

    assert np.array_equal(bundle_a.embeddings, bundle_b.embeddings)
    assert np.array_equal(bundle_a.logits, bundle_b.logits)
    assert bundle_a.relative_paths == bundle_b.relative_paths
    assert bundle_a.absolute_paths == bundle_b.absolute_paths
    assert bundle_a.splits == bundle_b.splits
    assert selection_a.read_text(encoding="utf-8") == selection_b.read_text(encoding="utf-8")


def test_primary_encoder_seed_selection_uses_validation_only() -> None:
    run_dirs = (
        Path("/tmp/m01c_seed7"),
        Path("/tmp/m01c_seed42"),
        Path("/tmp/m01c_seed13"),
    )
    payloads = {
        run_dirs[0]: {
            "locked_primary_aggregator": "mean_logits",
            "best_validation_summary": {"file_acc@1": 90.0, "file_macro_f1": 87.33333333333333},
        },
        run_dirs[1]: {
            "locked_primary_aggregator": "majority_vote",
            "best_validation_summary": {"file_acc@1": 90.0, "file_macro_f1": 87.33333333333333},
        },
        run_dirs[2]: {
            "locked_primary_aggregator": "mean_probabilities",
            "best_validation_summary": {"file_acc@1": 86.0, "file_macro_f1": 82.33333333333333},
        },
    }
    for run_dir, payload in payloads.items():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_metadata.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    selection = select_primary_encoder_run(run_dirs)

    assert selection.selected_run_id == "m01c_seed42"
    assert selection.validation_file_acc_at_1 == 90.0
    assert selection.validation_file_macro_f1 == 87.33333333333333


def test_ensemble_method_selection_uses_validation_only() -> None:
    validation_results = {
        "mean_logits": SimpleNamespace(metrics=SimpleNamespace(acc_at_1=88.0, f1_macro=83.0)),
        "mean_probabilities": SimpleNamespace(
            metrics=SimpleNamespace(acc_at_1=88.0, f1_macro=83.0)
        ),
        "vote": SimpleNamespace(metrics=SimpleNamespace(acc_at_1=88.0, f1_macro=82.0)),
    }

    selected = select_ensemble_method(validation_results=validation_results)

    assert selected == "mean_probabilities"


def test_attention_deepsets_forward_supports_variable_length_batches() -> None:
    model = AttentionDeepSetsClassifier(input_dim=16, hidden_dim=32, num_classes=50, dropout=0.1)
    batch = torch.randn(3, 7, 16)
    mask = torch.tensor(
        [
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, True, False],
            [True, True, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )

    logits = model(batch, mask)

    assert logits.shape == (3, 50)


def test_learned_file_level_checkpoint_selection_uses_validation_only() -> None:
    incumbent = FileCheckpointSelection(
        epoch=2,
        checkpoint_path=Path("/tmp/incumbent.pt"),
        validation_file_acc_at_1=84.0,
        validation_file_acc_at_5=98.0,
        validation_file_f1=79.0,
    )
    better_f1 = FileCheckpointSelection(
        epoch=3,
        checkpoint_path=Path("/tmp/better.pt"),
        validation_file_acc_at_1=84.0,
        validation_file_acc_at_5=98.0,
        validation_file_f1=80.0,
    )
    worse = FileCheckpointSelection(
        epoch=4,
        checkpoint_path=Path("/tmp/worse.pt"),
        validation_file_acc_at_1=83.0,
        validation_file_acc_at_5=98.0,
        validation_file_f1=90.0,
    )

    assert is_better_file_checkpoint(better_f1, incumbent) is True
    assert is_better_file_checkpoint(worse, incumbent) is False


def test_m03_seed_summary_is_deterministic() -> None:
    summary = build_m03_seed_summary(
        [
            {
                "run_id": "m03_a",
                "encoder_source_run_id": "m01c_seed42",
                "file_level_model_family": "attention_deepsets",
                "file_acc@1": "86.0",
                "file_acc@5": "98.0",
                "file_macro_f1": "82.0",
            },
            {
                "run_id": "m03_b",
                "encoder_source_run_id": "m01c_seed42",
                "file_level_model_family": "attention_deepsets",
                "file_acc@1": "88.0",
                "file_acc@5": "100.0",
                "file_macro_f1": "84.0",
            },
        ]
    )

    assert summary["n_runs"] == 2
    assert summary["file_acc@1"]["mean"] == 87.0
    assert summary["file_acc@1"]["std"] == 1.0


def build_locked_moonshot_fixture_run(tmp_path: Path) -> Path:
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
    config_path = tmp_path / "m03_locked_fixture.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "track": "moonshot-enhanced-setting",
                "experiment_name": "m03_locked_fixture",
                "data_root": str(data_root.resolve()),
                "output_root": str((tmp_path / "runs").resolve()),
                "category_map_path": str(category_map_path.resolve()),
                "class_vocab_manifest_path": str(class_vocab_path.resolve()),
                "exact_upstream_regression_path": str(regression_path.resolve()),
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 128,
                "lr": 0.003,
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
                "model_name": "cnn",
                "model": {
                    "channels": [8, 16],
                    "kernel_size": 3,
                    "dropout": 0.1,
                    "use_batchnorm": True,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    run_dir, _window_metrics, _file_metrics, _prepared = run_moonshot_cnn(config_path)
    return run_dir
