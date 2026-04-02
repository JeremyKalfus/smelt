from __future__ import annotations

import json
from pathlib import Path

import yaml

from smelt.datasets import build_base_class_vocab_manifest, load_base_sensor_dataset
from smelt.evaluation.diagnostics import (
    build_moonshot_locked_run_row,
    build_moonshot_protocol_definition,
    build_moonshot_seed_summary,
)
from smelt.evaluation.file_level import (
    AggregatorSelectionCandidate,
    select_validation_locked_aggregator,
)
from smelt.training.run_moonshot import (
    AggregatorCheckpointSelection,
    is_better_validation_checkpoint,
    run_moonshot_cnn,
)


def test_validation_locked_aggregator_selection_is_deterministic() -> None:
    candidates = [
        AggregatorSelectionCandidate("mean_logits", acc_at_1=80.0, f1_macro=75.0),
        AggregatorSelectionCandidate("mean_probabilities", acc_at_1=80.0, f1_macro=75.0),
        AggregatorSelectionCandidate("majority_vote", acc_at_1=79.0, f1_macro=76.0),
    ]

    assert select_validation_locked_aggregator(candidates) == "mean_probabilities"


def test_checkpoint_selection_uses_validation_acc_then_f1() -> None:
    incumbent = AggregatorCheckpointSelection(
        aggregator="mean_logits",
        epoch=2,
        checkpoint_path=Path("/tmp/incumbent.pt"),
        validation_file_acc_at_1=80.0,
        validation_file_acc_at_5=98.0,
        validation_file_f1=75.0,
        validation_window_acc_at_1=60.0,
        validation_window_acc_at_5=90.0,
        validation_window_f1=58.0,
    )
    better_f1 = AggregatorCheckpointSelection(
        aggregator="mean_logits",
        epoch=3,
        checkpoint_path=Path("/tmp/better.pt"),
        validation_file_acc_at_1=80.0,
        validation_file_acc_at_5=98.0,
        validation_file_f1=76.0,
        validation_window_acc_at_1=59.0,
        validation_window_acc_at_5=89.0,
        validation_window_f1=57.0,
    )
    worse = AggregatorCheckpointSelection(
        aggregator="mean_logits",
        epoch=4,
        checkpoint_path=Path("/tmp/worse.pt"),
        validation_file_acc_at_1=79.0,
        validation_file_acc_at_5=98.0,
        validation_file_f1=80.0,
        validation_window_acc_at_1=65.0,
        validation_window_acc_at_5=92.0,
        validation_window_f1=62.0,
    )

    assert is_better_validation_checkpoint(better_f1, incumbent) is True
    assert is_better_validation_checkpoint(worse, incumbent) is False


def test_protocol_definition_and_seed_summary_are_deterministic(tmp_path: Path) -> None:
    config_path = tmp_path / "locked.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "track": "moonshot-enhanced-setting",
                "model_name": "cnn",
                "channel_set": "all12",
                "diff_period": 25,
                "window_size": 100,
                "stride": 50,
                "validation_files_per_class": 1,
                "candidate_file_aggregators": [
                    "mean_logits",
                    "mean_probabilities",
                    "majority_vote",
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    protocol = build_moonshot_protocol_definition(config_path)
    summary = build_moonshot_seed_summary(
        [
            {
                "run_id": "a",
                "locked_primary_aggregator": "mean_probabilities",
                "acc@1": "67.0",
                "acc@5": "94.0",
                "macro_f1": "66.0",
                "file_acc@1_locked": "82.0",
                "file_acc@5_locked": "98.0",
                "file_macro_f1_locked": "78.0",
            },
            {
                "run_id": "b",
                "locked_primary_aggregator": "mean_probabilities",
                "acc@1": "69.0",
                "acc@5": "95.0",
                "macro_f1": "67.0",
                "file_acc@1_locked": "88.0",
                "file_acc@5_locked": "98.0",
                "file_macro_f1_locked": "85.0",
            },
        ]
    )

    assert protocol["locked_aggregator_rule"]["source"] == "validation_only"
    assert protocol["checkpoint_selection_rule"]["source"] == "validation_only"
    assert summary["n_runs"] == 2
    assert summary["acc@1"]["mean"] == 68.0
    assert summary["file_acc@1_locked"]["std"] == 3.0


def test_locked_protocol_run_writes_validation_locked_artifacts(tmp_path: Path) -> None:
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
    config_path = tmp_path / "locked.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "track": "moonshot-enhanced-setting",
                "experiment_name": "m01c_smoke",
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
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    locked_row = build_moonshot_locked_run_row(run_dir)

    assert metadata["locked_primary_aggregator"] in {
        "mean_logits",
        "mean_probabilities",
        "majority_vote",
    }
    assert metadata["primary_checkpoint_selection_metric"] == (
        "validation_file_acc@1_then_validation_file_macro_f1"
    )
    assert (run_dir / "locked_aggregator_selection.json").is_file()
    assert (run_dir / "validation_file_level_metrics_comparison.json").is_file()
    assert locked_row["locked_primary_aggregator"] == metadata["locked_primary_aggregator"]
