from __future__ import annotations

import json
from pathlib import Path

import yaml

from smelt.datasets import (
    MOONSHOT_TRACK,
    build_base_class_vocab_manifest,
    deterministic_grouped_validation_split,
    load_base_sensor_dataset,
    prepare_moonshot_window_splits,
)
from smelt.training.run_moonshot import load_moonshot_run_config, run_moonshot_cnn


def test_grouped_validation_split_is_deterministic_and_leak_free() -> None:
    dataset = load_base_sensor_dataset(Path("tests/fixtures/smellnet_base_valid"))

    first = deterministic_grouped_validation_split(
        dataset.train_records,
        validation_files_per_class=1,
    )
    second = deterministic_grouped_validation_split(
        dataset.train_records,
        validation_files_per_class=1,
    )

    assert [record.relative_path for record in first.train_records] == [
        record.relative_path for record in second.train_records
    ]
    assert [record.relative_path for record in first.validation_records] == [
        record.relative_path for record in second.validation_records
    ]
    assert not (
        {record.relative_path for record in first.train_records}
        & {record.relative_path for record in first.validation_records}
    )


def test_all12_diff_path_resolves_twelve_channels() -> None:
    dataset = load_base_sensor_dataset(Path("tests/fixtures/smellnet_base_valid"))
    prepared = prepare_moonshot_window_splits(
        dataset,
        diff_period=1,
        window_size=2,
        stride=1,
        validation_files_per_class=1,
    )

    assert prepared.standardized_train_split.column_names == dataset.raw_column_names
    assert len(prepared.standardized_train_split.column_names) == 12


def test_moonshot_run_config_resolves_track_and_aggregators(tmp_path: Path) -> None:
    config_path = tmp_path / "moonshot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "track": "moonshot-enhanced-setting",
                "experiment_name": "m01_smoke",
                "data_root": "tests/fixtures/smellnet_base_valid",
                "output_root": str((tmp_path / "runs").resolve()),
                "category_map_path": "configs/exact-upstream/category_map.json",
                "class_vocab_manifest_path": str((tmp_path / "class_vocab.json").resolve()),
                "exact_upstream_regression_path": str((tmp_path / "regression.json").resolve()),
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 16,
                "lr": 0.003,
                "weight_decay": 0.0005,
                "grad_clip": 1.0,
                "label_smoothing": 0.05,
                "diff_period": 1,
                "window_size": 2,
                "stride": 1,
                "num_workers": 0,
                "validation_files_per_class": 1,
                "primary_file_aggregator": "mean_logits",
                "validation_file_aggregator": "mean_logits",
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
    (tmp_path / "class_vocab.json").write_text('{"class_vocab": []}', encoding="utf-8")
    (tmp_path / "regression.json").write_text("{}", encoding="utf-8")

    config = load_moonshot_run_config(config_path)

    assert config.track == MOONSHOT_TRACK
    assert config.primary_file_aggregator == "mean_logits"
    assert config.validation_file_aggregator == "mean_logits"


def test_moonshot_run_writes_window_and_file_level_artifacts(tmp_path: Path) -> None:
    data_root = Path("tests/fixtures/smellnet_base_valid")
    dataset = load_base_sensor_dataset(data_root)
    category_cycle = ("nuts", "spices", "herbs", "fruits", "vegetables")
    category_map_path = tmp_path / "category_map.json"
    category_map_path.write_text(
        json.dumps(
            {
                class_name: category_cycle[index % len(category_cycle)]
                for index, class_name in enumerate(dataset.class_vocab)
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
    config_path = tmp_path / "moonshot.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "track": "moonshot-enhanced-setting",
                "experiment_name": "m01_smoke",
                "data_root": str(data_root.resolve()),
                "output_root": str((tmp_path / "runs").resolve()),
                "category_map_path": str(category_map_path),
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
                "primary_file_aggregator": "mean_logits",
                "validation_file_aggregator": "mean_logits",
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

    run_dir, metrics, file_metrics, prepared = run_moonshot_cnn(config_path)

    assert metrics.acc_at_1 >= 0.0
    assert file_metrics.acc_at_1 >= 0.0
    assert prepared.train_windows.shape[2] == 12
    assert (run_dir / "summary_metrics.json").is_file()
    assert (run_dir / "predictions.npz").is_file()
    assert (run_dir / "file_level_metrics_comparison.json").is_file()
    assert (run_dir / "file_level" / "mean_logits" / "summary_metrics.json").is_file()
    assert (run_dir / "validation_split.json").is_file()
