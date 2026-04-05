from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from smelt.evaluation import FileScoreBundle
from smelt.evaluation.diagnostics import build_m04_model_bank_row
from smelt.training.m04 import (
    build_diversity_matrix_rows,
    decide_optional_transformer_family,
    evaluate_m04_ensemble_candidates,
)
from smelt.training.run_moonshot import load_moonshot_run_config, run_moonshot_cnn


def test_m04_family_configs_resolve_under_locked_protocol(tmp_path: Path) -> None:
    for model_name, model_payload in (
        (
            "hinception",
            {
                "stem_channels": 32,
                "branch_channels": 16,
                "bottleneck_channels": 16,
                "num_blocks": 4,
                "residual_interval": 2,
                "activation_name": "gelu",
                "dropout": 0.1,
                "head_hidden_dim": 64,
            },
        ),
        (
            "patch_transformer",
            {
                "patch_size": 4,
                "patch_stride": 2,
                "model_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.1,
            },
        ),
    ):
        config_path = tmp_path / f"{model_name}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "track": "moonshot-enhanced-setting",
                    "experiment_name": f"m04_{model_name}_fixture",
                    "data_root": "/tmp/data",
                    "output_root": "/tmp/runs",
                    "category_map_path": "/tmp/category_map.json",
                    "class_vocab_manifest_path": "/tmp/class_vocab.json",
                    "exact_upstream_regression_path": "/tmp/regression.json",
                    "seed": 42,
                    "device": "cpu",
                    "epochs": 1,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 1,
                    "lr": 0.001,
                    "weight_decay": 0.0005,
                    "grad_clip": 1.0,
                    "label_smoothing": 0.05,
                    "diff_period": 25,
                    "window_size": 100,
                    "stride": 50,
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
                    "scheduler_eta_min": 0.0,
                    "shuffle_train_labels": False,
                    "model_name": model_name,
                    "model": model_payload,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        config = load_moonshot_run_config(config_path)

        assert config.locked_protocol is True
        assert config.channel_set == "all12"
        assert config.model_name == model_name


def test_m04_model_bank_row_is_deterministic(tmp_path: Path) -> None:
    run_dir = build_m04_fixture_run(tmp_path)
    export_dir = tmp_path / "embeddings" / run_dir.name
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "window_features.npz").write_bytes(b"fixture")

    first = build_m04_model_bank_row(run_dir=run_dir, export_dir=export_dir)
    second = build_m04_model_bank_row(run_dir=run_dir, export_dir=export_dir)

    assert first == second
    assert first["family_name"] == "cnn"


def test_m04_diversity_matrix_is_deterministic() -> None:
    bundles = {
        "run_a": build_file_score_bundle(
            scores=np.asarray([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64),
            predicted=np.asarray([0, 1], dtype=np.int64),
        ),
        "run_b": build_file_score_bundle(
            scores=np.asarray([[0.7, 0.3], [0.3, 0.7]], dtype=np.float64),
            predicted=np.asarray([0, 1], dtype=np.int64),
        ),
    }

    first = build_diversity_matrix_rows(validation_bundles=bundles)
    second = build_diversity_matrix_rows(validation_bundles=bundles)

    assert first == second
    assert len(first) == 4


def test_m04_ensemble_selection_uses_validation_only() -> None:
    validation_bundles = {
        "run_a": build_file_score_bundle(
            scores=np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64),
            predicted=np.asarray([0, 1], dtype=np.int64),
        ),
        "run_b": build_file_score_bundle(
            scores=np.asarray([[0.85, 0.15], [0.15, 0.85]], dtype=np.float64),
            predicted=np.asarray([0, 1], dtype=np.int64),
        ),
        "run_c": build_file_score_bundle(
            scores=np.asarray([[0.4, 0.6], [0.6, 0.4]], dtype=np.float64),
            predicted=np.asarray([1, 0], dtype=np.int64),
        ),
    }
    test_bundles = {
        "run_a": build_file_score_bundle(
            scores=np.asarray([[0.4, 0.6], [0.6, 0.4]], dtype=np.float64),
            predicted=np.asarray([1, 0], dtype=np.int64),
        ),
        "run_b": build_file_score_bundle(
            scores=np.asarray([[0.45, 0.55], [0.55, 0.45]], dtype=np.float64),
            predicted=np.asarray([1, 0], dtype=np.int64),
        ),
        "run_c": build_file_score_bundle(
            scores=np.asarray([[0.95, 0.05], [0.05, 0.95]], dtype=np.float64),
            predicted=np.asarray([0, 1], dtype=np.int64),
        ),
    }

    selection, _candidates = evaluate_m04_ensemble_candidates(
        validation_bundles=validation_bundles,
        test_bundles=test_bundles,
        category_mapping={"class0": "nuts", "class1": "spices"},
        max_members=2,
    )

    assert selection["selected_method"] != ""
    assert selection["selected_member_run_ids"] != ["run_c"]


def test_optional_transformer_skip_path_is_explicit() -> None:
    decision = decide_optional_transformer_family(
        smoke_succeeded=False,
        smoke_payload={
            "device": "mps",
            "parameter_count": 12_000_000,
            "batch_size": 16,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 64,
        },
    )

    assert decision.status == "skipped"
    assert decision.reason == "device_smoke_failed"


def build_file_score_bundle(
    *,
    scores: np.ndarray,
    predicted: np.ndarray,
) -> FileScoreBundle:
    topk = np.argsort(-scores, axis=1, kind="mergesort")
    return FileScoreBundle(
        aggregator="mean_probabilities",
        class_names=("class0", "class1"),
        scores=np.asarray(scores, dtype=np.float64),
        true_labels=np.asarray([0, 1], dtype=np.int64),
        predicted_labels=np.asarray(predicted, dtype=np.int64),
        topk_indices=np.asarray(topk, dtype=np.int64),
        split_names=("validation", "validation"),
        relative_paths=("a.csv", "b.csv"),
        absolute_paths=("/tmp/a.csv", "/tmp/b.csv"),
        num_windows=np.asarray([3, 3], dtype=np.int64),
    )


def build_m04_fixture_run(tmp_path: Path) -> Path:
    from smelt.datasets import build_base_class_vocab_manifest, load_base_sensor_dataset

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
    config_path = tmp_path / "m04_fixture.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "track": "moonshot-enhanced-setting",
                "experiment_name": "m04_fixture",
                "data_root": str(data_root.resolve()),
                "output_root": str((tmp_path / "runs").resolve()),
                "category_map_path": str(category_map_path.resolve()),
                "class_vocab_manifest_path": str(class_vocab_path.resolve()),
                "exact_upstream_regression_path": str(regression_path.resolve()),
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 16,
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
                "scheduler_eta_min": 0.0,
                "shuffle_train_labels": False,
                "model_name": "cnn",
                "model": {
                    "channels": [16, 32],
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
