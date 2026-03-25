"""audit the public smellnet-base split contract."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .contracts import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    AuditManifest,
    BenchmarkContractResult,
    FileMetadata,
    SchemaMetadata,
    SchemaSummary,
)

STRICT_CLASS_COUNT = 50
STRICT_TRAIN_FILES = 5
STRICT_TEST_FILES = 1


class AuditError(Exception):
    """raised when the base split contract is violated."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m smelt.datasets.audit_base",
        description="audit the exact-upstream smellnet-base split contract",
    )
    parser.add_argument("--data-root", required=True, help="path to the dataset root")
    parser.add_argument("--emit", required=True, help="path to write the manifest json")
    parser.add_argument(
        "--strict-upstream",
        action="store_true",
        help="enforce the public upstream class and file-count expectations",
    )
    return parser


def audit_base_dataset(data_root: Path, strict_upstream: bool = False) -> AuditManifest:
    resolved_root = data_root.expanduser().resolve()
    split_paths = {
        TRAIN_SPLIT: str(resolved_root / TRAIN_SPLIT),
        TEST_SPLIT: str(resolved_root / TEST_SPLIT),
    }
    violations: list[str] = []
    warnings: list[str] = []

    if not resolved_root.exists():
        violations.append(f"data root does not exist: {resolved_root}")
    elif not resolved_root.is_dir():
        violations.append(f"data root is not a directory: {resolved_root}")

    split_dirs: dict[str, Path] = {
        TRAIN_SPLIT: resolved_root / TRAIN_SPLIT,
        TEST_SPLIT: resolved_root / TEST_SPLIT,
    }
    per_split_class_lists: dict[str, list[str]] = {TRAIN_SPLIT: [], TEST_SPLIT: []}
    per_class_file_counts: dict[str, dict[str, int]] = {TRAIN_SPLIT: {}, TEST_SPLIT: {}}
    files: list[FileMetadata] = []
    schema_counter: Counter[tuple[str, ...]] = Counter()

    for split_name, split_dir in split_dirs.items():
        if not split_dir.exists():
            violations.append(f"required split directory is missing: {split_dir}")
            continue
        if not split_dir.is_dir():
            violations.append(f"split path is not a directory: {split_dir}")
            continue

        class_names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        per_split_class_lists[split_name] = class_names
        if not class_names:
            violations.append(f"split directory has no class subdirectories: {split_dir}")
            continue

        for class_name in class_names:
            class_dir = split_dir / class_name
            csv_paths = sorted(path for path in class_dir.glob("*.csv") if path.is_file())
            per_class_file_counts[split_name][class_name] = len(csv_paths)

            if not csv_paths:
                violations.append(f"class directory has no csv files: {class_dir}")
                continue

            expected_count = None
            if strict_upstream and split_name == TRAIN_SPLIT:
                expected_count = STRICT_TRAIN_FILES
            if strict_upstream and split_name == TEST_SPLIT:
                expected_count = STRICT_TEST_FILES
            if expected_count is not None and len(csv_paths) != expected_count:
                violations.append(
                    f"strict-upstream file count mismatch for {split_name}/{class_name}: "
                    f"expected {expected_count}, found {len(csv_paths)}"
                )

            for csv_path in csv_paths:
                try:
                    file_metadata = read_csv_metadata(
                        csv_path,
                        resolved_root,
                        split_name,
                        class_name,
                    )
                except AuditError as exc:
                    violations.append(str(exc))
                    continue

                files.append(file_metadata)
                schema_counter[tuple(file_metadata.column_names)] += 1

    train_classes = set(per_split_class_lists[TRAIN_SPLIT])
    test_classes = set(per_split_class_lists[TEST_SPLIT])
    if train_classes != test_classes:
        missing_in_test = sorted(train_classes - test_classes)
        missing_in_train = sorted(test_classes - train_classes)
        violations.append(
            "class vocab mismatch across train/test: "
            f"missing_in_test={missing_in_test}, missing_in_train={missing_in_train}"
        )

    class_vocab = sorted(train_classes | test_classes)
    if strict_upstream and len(class_vocab) != STRICT_CLASS_COUNT:
        violations.append(
            "strict-upstream class count mismatch: "
            f"expected {STRICT_CLASS_COUNT}, found {len(class_vocab)}"
        )

    distinct_schemas = [
        SchemaMetadata(column_count=len(schema), column_names=list(schema), file_count=file_count)
        for schema, file_count in sorted(schema_counter.items())
    ]
    globally_consistent = len(distinct_schemas) == 1 and bool(files)
    if files and not globally_consistent:
        violations.append(
            "inconsistent csv schema detected across files: "
            f"found {len(distinct_schemas)} distinct schemas"
        )

    benchmark_contract = BenchmarkContractResult(
        passed=not violations,
        violations=violations,
        warnings=warnings,
    )
    schema_summary = SchemaSummary(
        distinct_schemas=distinct_schemas,
        globally_consistent=globally_consistent,
    )
    return AuditManifest(
        resolved_data_root=str(resolved_root),
        split_paths=split_paths,
        strict_upstream=strict_upstream,
        class_vocab=class_vocab,
        class_count=len(class_vocab),
        per_split_class_lists=per_split_class_lists,
        per_class_file_counts=per_class_file_counts,
        files=files,
        schema_summary=schema_summary,
        benchmark_contract=benchmark_contract,
    )


def read_csv_metadata(
    csv_path: Path,
    data_root: Path,
    split_name: str,
    class_name: str,
) -> FileMetadata:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise AuditError(f"csv file is empty: {csv_path}") from exc

            if not header:
                raise AuditError(f"csv header is empty: {csv_path}")

            row_count = 0
            expected_width = len(header)
            for row_index, row in enumerate(reader, start=2):
                if len(row) != expected_width:
                    raise AuditError(
                        f"inconsistent csv schema in {csv_path}: row {row_index} has "
                        f"{len(row)} columns, expected {expected_width}"
                    )
                row_count += 1
    except OSError as exc:
        raise AuditError(f"unable to read csv file {csv_path}: {exc}") from exc
    except UnicodeError as exc:
        raise AuditError(f"unable to decode csv file {csv_path}: {exc}") from exc
    except csv.Error as exc:
        raise AuditError(f"unable to parse csv file {csv_path}: {exc}") from exc

    return FileMetadata(
        relative_path=str(csv_path.relative_to(data_root)),
        split=split_name,
        class_name=class_name,
        row_count=row_count,
        column_count=len(header),
        column_names=header,
    )


def write_manifest(emit_path: Path, manifest: AuditManifest) -> None:
    emit_path.parent.mkdir(parents=True, exist_ok=True)
    emit_path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")


def format_failure(violations: list[str]) -> str:
    lines = ["audit failed:"]
    lines.extend(f"- {violation}" for violation in violations)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    manifest = audit_base_dataset(Path(args.data_root), strict_upstream=args.strict_upstream)
    write_manifest(Path(args.emit), manifest)

    if not manifest.benchmark_contract.passed:
        print(format_failure(manifest.benchmark_contract.violations), file=sys.stderr)
        return 1

    print(
        "audit passed: "
        f"{manifest.class_count} classes, {len(manifest.files)} files, "
        f"schema_consistent={manifest.schema_summary.globally_consistent}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
