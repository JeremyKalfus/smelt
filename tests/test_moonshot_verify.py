from __future__ import annotations

import json
from pathlib import Path

import yaml

from smelt.datasets import build_base_class_vocab_manifest, load_base_sensor_dataset
from smelt.training.run_moonshot import run_moonshot_cnn
from smelt.training.verify_moonshot import verify_moonshot_run


def test_verify_moonshot_run_is_deterministic_on_fixture(tmp_path: Path) -> None:
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
                "experiment_name": "m01b_verify_smoke",
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
                "shuffle_train_labels": False,
                "channel_set": "all12",
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
    result = verify_moonshot_run(run_dir)

    assert result.a1_pass is True
    assert result.a2_pass is True
    assert result.a3_pass is True
    assert result.summary_metrics_path.is_file()
    assert result.predictions_path.is_file()
    assert result.verification_summary_path.is_file()
