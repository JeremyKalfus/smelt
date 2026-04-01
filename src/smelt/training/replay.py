"""checkpoint replay helpers for eval-only exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from smelt.datasets import (
    MOONSHOT_TRACK,
    build_base_class_vocab_manifest,
    load_base_sensor_dataset,
    preprocess_split_records_for_view,
)
from smelt.evaluation import load_category_mapping
from smelt.models import ExactResearchInceptionClassifier
from smelt.preprocessing import (
    EXACT_UPSTREAM_DROPPED_COLUMNS,
    apply_window_standardizer,
    fit_window_standardizer,
    generate_split_windows,
    preprocess_split_records,
    stack_window_values,
)
from smelt.preprocessing.windows import WindowedSplit

from .run import (
    build_classifier_model,
    build_dataloader,
    load_run_config,
    resolve_device,
)
from .run_moonshot import load_moonshot_run_config, prepare_moonshot_tensors
from .run_research import (
    load_research_run_config,
)


class ReplayError(Exception):
    """raised when a saved run cannot be replayed safely."""


@dataclass(slots=True)
class ReplayContext:
    run_dir: Path
    track: str
    class_names: tuple[str, ...]
    category_mapping: dict[str, str]
    test_windows: WindowedSplit
    test_loader: DataLoader[Any]
    model: nn.Module
    device: torch.device


def load_replay_context(run_dir: Path) -> ReplayContext:
    resolved_run_dir = run_dir.expanduser().resolve()
    config_path = resolved_run_dir / "resolved_config.yaml"
    raw_config = load_yaml_file(config_path)
    track = str(raw_config.get("track", ""))
    if track == "exact-upstream":
        return load_exact_upstream_replay_context(resolved_run_dir)
    if track == "research-extension":
        return load_research_replay_context(resolved_run_dir)
    if track == MOONSHOT_TRACK:
        return load_moonshot_replay_context(resolved_run_dir)
    raise ReplayError(f"unsupported replay track for {resolved_run_dir}: {track!r}")


def load_exact_upstream_replay_context(run_dir: Path) -> ReplayContext:
    config = load_run_config(run_dir / "resolved_config.yaml")
    dataset = load_base_sensor_dataset(Path(config.data_root))
    class_names = build_base_class_vocab_manifest(dataset).class_vocab
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    category_mapping = load_category_mapping(Path(config.category_map_path))

    train_records = preprocess_split_records(
        dataset.train_records,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=config.diff_period,
    )
    test_records = preprocess_split_records(
        dataset.test_records,
        dropped_columns=EXACT_UPSTREAM_DROPPED_COLUMNS,
        diff_period=config.diff_period,
    )
    train_windows = generate_split_windows(
        train_records,
        window_size=config.window_size,
        stride=config.stride,
    )
    test_windows = generate_split_windows(
        test_records,
        window_size=config.window_size,
        stride=config.stride,
    )
    standardizer = fit_window_standardizer(train_windows)
    standardized_test = apply_window_standardizer(test_windows, standardizer)
    test_loader = build_dataloader(
        stack_window_values(standardized_test.windows).astype(np.float32, copy=False),
        np.asarray(
            [class_to_index[window.class_name] for window in standardized_test.windows],
            dtype=np.int64,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    device = resolve_device(config.device)
    model = build_classifier_model(
        config=config,
        input_dim=len(standardized_test.column_names),
        num_classes=len(class_names),
    ).to(device)
    load_checkpoint_weights(model, run_dir / "checkpoint_final.pt")
    return ReplayContext(
        run_dir=run_dir,
        track=config.track,
        class_names=class_names,
        category_mapping=category_mapping,
        test_windows=standardized_test,
        test_loader=test_loader,
        model=model,
        device=device,
    )


def load_research_replay_context(run_dir: Path) -> ReplayContext:
    config = load_research_run_config(run_dir / "resolved_config.yaml")
    dataset = load_base_sensor_dataset(Path(config.data_root))
    class_names = build_base_class_vocab_manifest(dataset).class_vocab
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    category_mapping = load_category_mapping(Path(config.category_map_path))

    train_records = preprocess_split_records_for_view(
        dataset.train_records,
        view_mode=config.view_mode,
        diff_period=config.diff_period,
    )
    test_records = preprocess_split_records_for_view(
        dataset.test_records,
        view_mode=config.view_mode,
        diff_period=config.diff_period,
    )
    train_windows = generate_split_windows(
        train_records,
        window_size=config.window_size,
        stride=config.stride,
    )
    test_windows = generate_split_windows(
        test_records,
        window_size=config.window_size,
        stride=config.stride,
    )
    standardizer = fit_window_standardizer(train_windows)
    standardized_test = apply_window_standardizer(test_windows, standardizer)
    test_loader = build_dataloader(
        stack_window_values(standardized_test.windows).astype(np.float32, copy=False),
        np.asarray(
            [class_to_index[window.class_name] for window in standardized_test.windows],
            dtype=np.int64,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    device = resolve_device(config.device)
    model = ExactResearchInceptionClassifier(
        input_dim=len(standardized_test.column_names),
        num_classes=len(class_names),
        stem_channels=config.model.stem_channels,
        branch_channels=config.model.branch_channels,
        bottleneck_channels=config.model.bottleneck_channels,
        num_blocks=config.model.num_blocks,
        residual_interval=config.model.residual_interval,
        activation_name=config.model.activation_name,
        dropout=config.model.dropout,
        head_hidden_dim=config.model.head_hidden_dim,
    ).to(device)
    load_checkpoint_weights(model, run_dir / "checkpoint_final.pt")
    return ReplayContext(
        run_dir=run_dir,
        track=config.track,
        class_names=class_names,
        category_mapping=category_mapping,
        test_windows=standardized_test,
        test_loader=test_loader,
        model=model,
        device=device,
    )


def load_moonshot_replay_context(run_dir: Path) -> ReplayContext:
    config = load_moonshot_run_config(run_dir / "resolved_config.yaml")
    dataset = load_base_sensor_dataset(Path(config.data_root))
    class_names = build_base_class_vocab_manifest(dataset).class_vocab
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    category_mapping = load_category_mapping(Path(config.category_map_path))

    prepared = prepare_moonshot_tensors(dataset, config)
    test_loader = build_dataloader(
        prepared.test_windows,
        np.asarray(
            [
                class_to_index[window.class_name]
                for window in prepared.standardized_test_split.windows
            ],
            dtype=np.int64,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    device = resolve_device(config.device)
    model = build_classifier_model(
        config=config,
        input_dim=prepared.test_windows.shape[2],
        num_classes=len(class_names),
    ).to(device)
    load_checkpoint_weights(model, run_dir / "checkpoint_final.pt")
    return ReplayContext(
        run_dir=run_dir,
        track=config.track,
        class_names=class_names,
        category_mapping=category_mapping,
        test_windows=prepared.standardized_test_split,
        test_loader=test_loader,
        model=model,
        device=device,
    )


def load_checkpoint_weights(model: nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise ReplayError(f"checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ReplayError(f"checkpoint missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(state_dict)


def load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReplayError(f"unable to read yaml file: {path}") from exc
    if not isinstance(payload, dict):
        raise ReplayError(f"yaml config must deserialize to a mapping: {path}")
    return payload
