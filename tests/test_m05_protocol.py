from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from smelt.datasets import (
    BaseSensorDataset,
    SensorFileRecord,
    build_base_class_vocab_manifest,
    build_grouped_cv_fold_manifest,
    load_base_sensor_dataset,
)
from smelt.evaluation import load_category_mapping
from smelt.training.m05 import (
    FrozenRefitMember,
    M05BankEntry,
    build_m05_duplicate_audit,
    build_m05_ensemble_rows,
    run_full_refit_member,
    run_m05_cv_search,
)
from smelt.training.run_moonshot import load_moonshot_run_config


def test_grouped_cv_fold_manifest_covers_each_training_file_once_in_validation() -> None:
    dataset = load_base_sensor_dataset(Path("tests/fixtures/smellnet_base_valid"))
    manifest = build_grouped_cv_fold_manifest(dataset.train_records, fold_count=5)

    validation_counts = Counter(
        record.relative_path for fold in manifest.folds for record in fold.validation_records
    )

    assert manifest.fold_count == 5
    assert all(count == 1 for count in validation_counts.values())
    assert set(validation_counts) == {record.relative_path for record in dataset.train_records}
    for fold in manifest.folds:
        assert not (
            {record.relative_path for record in fold.train_records}
            & {record.relative_path for record in fold.validation_records}
        )


def test_m05_duplicate_audit_flags_cross_boundary_exact_duplicates() -> None:
    dataset = load_base_sensor_dataset(Path("tests/fixtures/smellnet_base_valid"))
    manifest = build_grouped_cv_fold_manifest(dataset.train_records, fold_count=5)
    train_record = dataset.train_records[0]
    duplicate_test_record = SensorFileRecord(
        split="offline_testing",
        class_name=train_record.class_name,
        relative_path=f"offline_testing/{train_record.class_name}/duplicate.csv",
        absolute_path="/tmp/duplicate.csv",
        column_names=train_record.column_names,
        rows=train_record.rows,
    )
    duplicate_dataset = BaseSensorDataset(
        resolved_data_root=dataset.resolved_data_root,
        raw_column_names=dataset.raw_column_names,
        train_records=dataset.train_records,
        test_records=dataset.test_records + (duplicate_test_record,),
    )

    audit = build_m05_duplicate_audit(dataset=duplicate_dataset, fold_manifest=manifest)

    assert audit["passed"] is False
    assert audit["collision_count"] >= 1


def test_m05_cv_selection_payload_contains_no_official_test_metrics(tmp_path: Path) -> None:
    dataset, category_mapping, entries = build_fixture_bank_entries(tmp_path, seeds=(7, 42))
    fold_manifest = build_grouped_cv_fold_manifest(dataset.train_records, fold_count=5)

    summaries, selection_payload, candidates = run_m05_cv_search(
        entries=entries,
        dataset=dataset,
        fold_manifest=fold_manifest,
        category_mapping=category_mapping,
        output_root=tmp_path / "runs",
    )

    assert summaries
    assert "test_file" not in json.dumps(selection_payload, sort_keys=True)
    search_rows = build_m05_ensemble_rows(
        candidates=candidates,
        selected_method=str(selection_payload["selected_method"]),
    )
    assert all("test" not in key for row in search_rows for key in row)


def test_m05_cv_search_does_not_invoke_full_refit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset, category_mapping, entries = build_fixture_bank_entries(tmp_path, seeds=(13, 42))
    fold_manifest = build_grouped_cv_fold_manifest(dataset.train_records, fold_count=5)
    called = False

    def fake_run_full_refit_member(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("cv search must not invoke the final refit path")

    monkeypatch.setattr("smelt.training.m05.run_full_refit_member", fake_run_full_refit_member)

    run_m05_cv_search(
        entries=entries,
        dataset=dataset,
        fold_manifest=fold_manifest,
        category_mapping=category_mapping,
        output_root=tmp_path / "runs",
    )

    assert called is False


def test_m05_full_refit_uses_frozen_epoch_budget_and_aggregator(tmp_path: Path) -> None:
    dataset, category_mapping, entries = build_fixture_bank_entries(tmp_path, seeds=(7,))
    entry = entries[0]
    plan = FrozenRefitMember(
        member_id=entry.member_id,
        config_path=entry.config_path,
        config=entry.config,
        stable_order=entry.stable_order,
        selected_aggregator="mean_probabilities",
        epoch_budget=2,
        selected_weight=1.0,
        oof_file_acc_at_1=0.0,
        oof_file_macro_f1=0.0,
    )

    result = run_full_refit_member(
        plan=plan,
        dataset=dataset,
        category_mapping=category_mapping,
        output_root=tmp_path / "full_refit",
    )

    metadata = json.loads((result.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    history = json.loads((result.run_dir / "training_history.json").read_text(encoding="utf-8"))

    assert metadata["locked_primary_aggregator"] == "mean_probabilities"
    assert metadata["frozen_epoch_budget"] == 2
    assert metadata["official_test_metrics_persisted"] is False
    assert len(history["rows"]) == 2


def build_fixture_bank_entries(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...],
) -> tuple[object, dict[str, str], tuple[M05BankEntry, ...]]:
    data_root = Path("tests/fixtures/smellnet_base_valid")
    dataset = load_base_sensor_dataset(data_root)
    category_map_path = tmp_path / "category_map.json"
    category_map_path.write_text(
        json.dumps(
            {
                class_name: ("nuts", "spices", "herbs", "fruits", "vegetables")[index % 5]
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
    entries: list[M05BankEntry] = []
    for stable_order, seed in enumerate(seeds):
        config_path = tmp_path / f"m05_fixture_seed{seed}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "track": "moonshot-enhanced-setting",
                    "experiment_name": f"m05_fixture_seed{seed}",
                    "data_root": str(data_root.resolve()),
                    "output_root": str((tmp_path / "runs").resolve()),
                    "category_map_path": str(category_map_path.resolve()),
                    "class_vocab_manifest_path": str(class_vocab_path.resolve()),
                    "exact_upstream_regression_path": str(regression_path.resolve()),
                    "seed": seed,
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
                    "shuffle_train_labels": False,
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
        config = load_moonshot_run_config(config_path)
        entries.append(
            M05BankEntry(
                member_id=config.experiment_name,
                config_path=config_path.resolve(),
                config=config,
                stable_order=stable_order,
            )
        )
    return dataset, load_category_mapping(category_map_path), tuple(entries)
