from __future__ import annotations

import csv
from pathlib import Path

import pytest

from smelt.datasets import (
    build_base_class_vocab_manifest,
    build_gcms_class_map,
    extract_base_class_vocab,
    load_base_sensor_dataset,
    load_gcms_source_table,
    resolve_exact_upstream_gcms_csv,
    write_base_class_vocab_manifest,
    write_gcms_class_map_manifest,
)
from smelt.datasets.gcms_map import GcmsMapError

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def test_extract_base_class_vocab_is_deterministic() -> None:
    dataset = load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_valid")

    vocab = extract_base_class_vocab(dataset)

    assert vocab == tuple(f"odor_{index:02d}" for index in range(50))


def test_build_gcms_class_map_succeeds_on_valid_fixture(tmp_path: Path) -> None:
    gcms_csv = tmp_path / "gcms.csv"
    write_gcms_fixture(
        gcms_csv,
        rows=[
            ("alpha", [1.0, 2.0]),
            ("beta", [3.0, 4.0]),
            ("gamma", [5.0, 6.0]),
        ],
    )

    source_table = load_gcms_source_table(gcms_csv)
    manifest = build_gcms_class_map(
        resolved_data_root="/tmp/base",
        class_vocab=("alpha", "beta", "gamma"),
        source_table=source_table,
    )

    assert manifest.class_count == 3
    assert manifest.anchor_count == 3
    assert manifest.validation.bijective_for_scope is True
    assert [entry.class_name for entry in manifest.mapping_entries] == ["alpha", "beta", "gamma"]


def test_build_gcms_class_map_fails_on_missing_class(tmp_path: Path) -> None:
    gcms_csv = tmp_path / "gcms.csv"
    write_gcms_fixture(
        gcms_csv,
        rows=[
            ("alpha", [1.0, 2.0]),
            ("beta", [3.0, 4.0]),
        ],
    )

    source_table = load_gcms_source_table(gcms_csv)

    with pytest.raises(GcmsMapError, match="missing gc-ms anchors"):
        build_gcms_class_map(
            resolved_data_root="/tmp/base",
            class_vocab=("alpha", "beta", "gamma"),
            source_table=source_table,
        )


def test_load_gcms_source_table_fails_on_duplicate_anchor_labels(tmp_path: Path) -> None:
    gcms_csv = tmp_path / "gcms.csv"
    write_gcms_fixture(
        gcms_csv,
        rows=[
            ("alpha", [1.0, 2.0]),
            ("alpha", [3.0, 4.0]),
        ],
    )

    with pytest.raises(GcmsMapError, match="duplicate anchor labels"):
        load_gcms_source_table(gcms_csv)


def test_build_gcms_class_map_fails_on_incomplete_vocab_agreement(tmp_path: Path) -> None:
    gcms_csv = tmp_path / "gcms.csv"
    write_gcms_fixture(
        gcms_csv,
        rows=[
            ("alpha", [1.0, 2.0]),
            ("beta", [3.0, 4.0]),
            ("extra", [5.0, 6.0]),
        ],
    )

    source_table = load_gcms_source_table(gcms_csv)

    with pytest.raises(GcmsMapError, match="outside base vocab"):
        build_gcms_class_map(
            resolved_data_root="/tmp/base",
            class_vocab=("alpha", "beta"),
            source_table=source_table,
        )


def test_manifest_serialization_is_deterministic(tmp_path: Path) -> None:
    dataset = load_base_sensor_dataset(FIXTURE_ROOT / "smellnet_base_valid")
    vocab_manifest = build_base_class_vocab_manifest(dataset)
    gcms_csv = tmp_path / "gcms.csv"
    write_gcms_fixture(
        gcms_csv,
        rows=[
            (class_name, [float(index), float(index + 1)])
            for index, class_name in enumerate(vocab_manifest.class_vocab)
        ],
    )
    source_table = load_gcms_source_table(gcms_csv)
    gcms_manifest = build_gcms_class_map(
        resolved_data_root=dataset.resolved_data_root,
        class_vocab=vocab_manifest.class_vocab,
        source_table=source_table,
    )

    vocab_json_a = tmp_path / "a_vocab.json"
    vocab_json_b = tmp_path / "b_vocab.json"
    map_json_a = tmp_path / "a_map.json"
    map_json_b = tmp_path / "b_map.json"
    write_base_class_vocab_manifest(vocab_json_a, vocab_manifest)
    write_base_class_vocab_manifest(vocab_json_b, vocab_manifest)
    write_gcms_class_map_manifest(map_json_a, gcms_manifest)
    write_gcms_class_map_manifest(map_json_b, gcms_manifest)

    assert vocab_json_a.read_text(encoding="utf-8") == vocab_json_b.read_text(encoding="utf-8")
    assert map_json_a.read_text(encoding="utf-8") == map_json_b.read_text(encoding="utf-8")


def test_real_reference_assets_build_bijective_map_when_available() -> None:
    base_root = Path(".reference/smellnet_upstream/data")
    gcms_csv = resolve_exact_upstream_gcms_csv(Path.cwd())
    if not base_root.is_dir() or not gcms_csv.is_file():
        pytest.skip("reference base data or gc-ms source is unavailable")

    dataset = load_base_sensor_dataset(base_root)
    vocab_manifest = build_base_class_vocab_manifest(dataset)
    source_table = load_gcms_source_table(gcms_csv)
    gcms_manifest = build_gcms_class_map(
        resolved_data_root=dataset.resolved_data_root,
        class_vocab=vocab_manifest.class_vocab,
        source_table=source_table,
    )

    assert vocab_manifest.class_count == 50
    assert source_table.row_count == 50
    assert gcms_manifest.validation.bijective_for_scope is True


def write_gcms_fixture(output_path: Path, *, rows: list[tuple[str, list[float]]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["food_name", "C", "H"])
        for anchor_label, feature_values in rows:
            writer.writerow([anchor_label, *feature_values])
