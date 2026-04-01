"""diagnostic coverage for t10b exports and single-view configs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from smelt.evaluation.diagnostics import build_recipe_snapshot, export_run_registry_artifacts
from smelt.training.run_research import load_research_run_config


def test_research_run_config_resolves_raw_aligned_view(tmp_path: Path) -> None:
    regression_path = "artifacts/methods/t10b_exact_upstream_regression_smoke.json"
    config_path = tmp_path / "raw.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "track": "research-extension",
                "experiment_name": "t10b_raw",
                "data_root": "tests/fixtures/smellnet_base_valid",
                "output_root": "results/runs",
                "category_map_path": "configs/exact-upstream/category_map.json",
                "class_vocab_manifest_path": "artifacts/manifests/base_class_vocab.json",
                "exact_upstream_regression_path": regression_path,
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 2,
                "lr": 0.001,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "diff_period": 25,
                "window_size": 100,
                "stride": 50,
                "num_workers": 0,
                "view_mode": "raw_aligned",
                "model": {
                    "stem_channels": 16,
                    "branch_channels": 8,
                    "bottleneck_channels": 8,
                    "num_blocks": 3,
                    "residual_interval": 3,
                    "activation_name": "gelu",
                    "dropout": 0.1,
                    "head_hidden_dim": 32,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = load_research_run_config(config_path)
    assert config.view_mode == "raw_aligned"


def test_research_run_config_resolves_diff_view(tmp_path: Path) -> None:
    regression_path = "artifacts/methods/t10b_exact_upstream_regression_smoke.json"
    config_path = tmp_path / "diff.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "track": "research-extension",
                "experiment_name": "t10b_diff",
                "data_root": "tests/fixtures/smellnet_base_valid",
                "output_root": "results/runs",
                "category_map_path": "configs/exact-upstream/category_map.json",
                "class_vocab_manifest_path": "artifacts/manifests/base_class_vocab.json",
                "exact_upstream_regression_path": regression_path,
                "seed": 42,
                "device": "cpu",
                "epochs": 1,
                "batch_size": 2,
                "lr": 0.001,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "diff_period": 25,
                "window_size": 100,
                "stride": 50,
                "num_workers": 0,
                "view_mode": "diff",
                "model": {
                    "stem_channels": 16,
                    "branch_channels": 8,
                    "bottleneck_channels": 8,
                    "num_blocks": 3,
                    "residual_interval": 3,
                    "activation_name": "gelu",
                    "dropout": 0.1,
                    "head_hidden_dim": 32,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = load_research_run_config(config_path)
    assert config.view_mode == "diff"


def test_export_run_registry_artifacts_writes_expected_files(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    run_root.mkdir()
    exact_run = run_root / "e0_transformer_cls_w100_g25-test"
    research_run = run_root / "t10b_inception_raw_aligned_supervised_w100_g25-test"
    exact_run.mkdir()
    research_run.mkdir()
    write_fake_run_dir(
        exact_run,
        track="exact-upstream",
        experiment_name="e0_transformer_cls_w100_g25",
        diff_period=25,
        window_size=100,
        stride=None,
        view_mode=None,
        metrics={
            "acc@1": 50.0,
            "acc@5": 80.0,
            "precision_macro": 51.0,
            "recall_macro": 49.0,
            "f1_macro": 48.0,
        },
    )
    write_fake_run_dir(
        research_run,
        track="research-extension",
        experiment_name="t10b_inception_raw_aligned_supervised_w100_g25",
        diff_period=25,
        window_size=100,
        stride=50,
        view_mode="raw_aligned",
        metrics={
            "acc@1": 45.0,
            "acc@5": 78.0,
            "precision_macro": 46.0,
            "recall_macro": 44.0,
            "f1_macro": 43.0,
        },
    )

    paths = export_run_registry_artifacts(
        run_root=run_root,
        table_root=tmp_path / "tables",
        figdata_root=tmp_path / "figdata",
        existing_run_ids=(exact_run.name, research_run.name),
    )
    assert Path(paths.run_registry_csv).exists()
    assert Path(paths.run_registry_json).exists()
    assert Path(paths.existing_run_summary_csv).exists()
    assert Path(paths.metrics_long_csv).exists()
    assert Path(paths.training_histories_long_csv).exists()
    assert Path(paths.file_level_metrics_long_csv).exists()

    with Path(paths.run_registry_json).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    run_ids = [row["run_id"] for row in payload["runs"]]
    assert exact_run.name in run_ids
    assert research_run.name in run_ids

    with Path(paths.training_histories_long_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4


def test_build_recipe_snapshot_is_deterministic(monkeypatch: object) -> None:
    monkeypatch.setenv("SMELT_DATA_ROOT", "tests/fixtures/smellnet_base_valid")
    exact_path = Path("configs/exact-upstream/f1_cnn_cls_w100_g25.yaml")
    research_path = Path("configs/research-extension/e1_inception_fused_supervised_w100_g25.yaml")
    exact_first = build_recipe_snapshot(exact_path)
    exact_second = build_recipe_snapshot(exact_path)
    research_first = build_recipe_snapshot(research_path)
    research_second = build_recipe_snapshot(research_path)
    assert exact_first == exact_second
    assert research_first == research_second
    assert exact_first["model_family"] == "cnn"
    assert research_first["model_family"] == "inception"


def write_fake_run_dir(
    run_dir: Path,
    *,
    track: str,
    experiment_name: str,
    diff_period: int,
    window_size: int,
    stride: int | None,
    view_mode: str | None,
    metrics: dict[str, float],
) -> None:
    (run_dir / "checkpoint_final.pt").write_bytes(b"checkpoint")
    (run_dir / "confusion_matrix.csv").write_text("true_class,a,b\n", encoding="utf-8")
    (run_dir / "per_category_accuracy.csv").write_text("category,n,acc@1,acc@5\n", encoding="utf-8")
    (run_dir / "predictions.npz").write_bytes(b"predictions")
    (run_dir / "summary_metrics.json").write_text(
        json.dumps(
            {
                **metrics,
                "class_names": ["a", "b"],
                "methods": {
                    "train_window_count": 10,
                    "test_window_count": 4,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata: dict[str, object] = {
        "track": track,
        "window_counts": {"train": 10, "test": 4},
        "checkpoint_path": str((run_dir / "checkpoint_final.pt").resolve()),
    }
    if view_mode is not None:
        metadata["view_mode"] = view_mode
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config_payload = {
        "track": track,
        "experiment_name": experiment_name,
        "diff_period": diff_period,
        "window_size": window_size,
        "stride": stride,
    }
    if track == "exact-upstream":
        config_payload["model_name"] = "transformer"
    else:
        config_payload["view_mode"] = view_mode
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "training_history.csv").write_text(
        "epoch,train_loss,train_acc@1\n1,1.0,50.0\n2,0.8,60.0\n",
        encoding="utf-8",
    )
    if view_mode is not None:
        (run_dir / "research_view_manifest.json").write_text(
            json.dumps({"feature_count": 12 if view_mode == "fused_raw_diff" else 6}) + "\n",
            encoding="utf-8",
        )
