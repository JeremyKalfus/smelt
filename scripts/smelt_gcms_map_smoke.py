"""build and verify the exact-upstream base gc-ms mapping artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from smelt.datasets import (
    build_base_class_vocab_manifest,
    build_gcms_class_map,
    load_base_sensor_dataset,
    load_gcms_source_table,
    resolve_exact_upstream_gcms_csv,
    write_base_class_vocab_manifest,
    write_gcms_class_map_csv,
    write_gcms_class_map_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--gcms-csv", type=Path, default=None)
    parser.add_argument("--emit-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_base_sensor_dataset(args.data_root)
    class_vocab_manifest = build_base_class_vocab_manifest(dataset)
    gcms_csv_path = (
        args.gcms_csv.expanduser().resolve()
        if args.gcms_csv is not None
        else resolve_exact_upstream_gcms_csv(Path.cwd())
    )
    source_table = load_gcms_source_table(gcms_csv_path)
    gcms_manifest = build_gcms_class_map(
        resolved_data_root=dataset.resolved_data_root,
        class_vocab=class_vocab_manifest.class_vocab,
        source_table=source_table,
    )

    args.emit_dir.mkdir(parents=True, exist_ok=True)
    vocab_json = args.emit_dir / "base_class_vocab.json"
    gcms_json = args.emit_dir / "gcms_class_map.json"
    gcms_csv = args.emit_dir / "gcms_class_map.csv"
    write_base_class_vocab_manifest(vocab_json, class_vocab_manifest)
    write_gcms_class_map_manifest(gcms_json, gcms_manifest)
    write_gcms_class_map_csv(gcms_csv, gcms_manifest)

    print(f"resolved_base_data_root: {dataset.resolved_data_root}")
    print(f"resolved_gcms_source_path: {source_table.resolved_source_path}")
    print(f"class_count: {class_vocab_manifest.class_count}")
    print(f"anchor_count: {source_table.row_count}")
    print(f"bijective_for_scope: {gcms_manifest.validation.bijective_for_scope}")
    print(f"base_class_vocab_json: {vocab_json.resolve()}")
    print(f"gcms_class_map_json: {gcms_json.resolve()}")
    print(f"gcms_class_map_csv: {gcms_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
