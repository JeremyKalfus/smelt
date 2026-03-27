"""strict gc-ms anchor mapping for the exact-upstream base benchmark."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


class GcmsMapError(Exception):
    """raised when the exact-upstream gc-ms mapping contract is violated."""


@dataclass(slots=True)
class GcmsSourceRow:
    source_row_index: int
    anchor_label: str
    feature_values: tuple[float, ...]


@dataclass(slots=True)
class GcmsSourceTable:
    resolved_source_path: str
    label_column: str
    feature_columns: tuple[str, ...]
    rows: tuple[GcmsSourceRow, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def anchor_labels(self) -> tuple[str, ...]:
        return tuple(row.anchor_label for row in self.rows)


@dataclass(slots=True)
class GcmsClassMapEntry:
    class_name: str
    anchor_label: str
    source_row_index: int

    def to_dict(self) -> dict[str, object]:
        return {
            "class_name": self.class_name,
            "anchor_label": self.anchor_label,
            "source_row_index": self.source_row_index,
        }


@dataclass(slots=True)
class GcmsValidationResult:
    passed: bool
    bijective_for_scope: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "bijective_for_scope": self.bijective_for_scope,
            "violations": self.violations,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class GcmsClassMapManifest:
    track: str
    usage_contract: str
    resolved_data_root: str
    resolved_gcms_source_path: str
    gcms_label_column: str
    gcms_feature_columns: tuple[str, ...]
    class_count: int
    anchor_count: int
    class_vocab: tuple[str, ...]
    mapping_entries: tuple[GcmsClassMapEntry, ...]
    validation: GcmsValidationResult

    def to_dict(self) -> dict[str, object]:
        return {
            "track": self.track,
            "usage_contract": self.usage_contract,
            "resolved_data_root": self.resolved_data_root,
            "resolved_gcms_source_path": self.resolved_gcms_source_path,
            "gcms_label_column": self.gcms_label_column,
            "gcms_feature_columns": list(self.gcms_feature_columns),
            "class_count": self.class_count,
            "anchor_count": self.anchor_count,
            "class_vocab": list(self.class_vocab),
            "mapping_entries": [entry.to_dict() for entry in self.mapping_entries],
            "validation": self.validation.to_dict(),
        }


def load_gcms_source_table(
    gcms_csv_path: Path,
    *,
    label_column: str = "food_name",
) -> GcmsSourceTable:
    resolved_path = gcms_csv_path.expanduser().resolve()
    if not resolved_path.exists():
        raise GcmsMapError(f"gc-ms source asset is missing: {resolved_path}")
    if not resolved_path.is_file():
        raise GcmsMapError(f"gc-ms source path is not a file: {resolved_path}")

    try:
        with resolved_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise GcmsMapError(f"gc-ms source has no header: {resolved_path}")
            header = tuple(reader.fieldnames)
            if label_column not in header:
                raise GcmsMapError(
                    "gc-ms source is missing required label column "
                    f"{label_column!r}: {resolved_path}"
                )
            feature_columns = tuple(
                column_name for column_name in header if column_name != label_column
            )
            if not feature_columns:
                raise GcmsMapError(f"gc-ms source has no feature columns: {resolved_path}")

            rows: list[GcmsSourceRow] = []
            labels: list[str] = []
            for source_row_index, row in enumerate(reader):
                anchor_label = (row.get(label_column) or "").strip()
                if not anchor_label:
                    raise GcmsMapError(
                        f"gc-ms row {source_row_index} has an empty anchor label in {resolved_path}"
                    )
                feature_values: list[float] = []
                for feature_name in feature_columns:
                    raw_value = row.get(feature_name)
                    if raw_value is None:
                        raise GcmsMapError(
                            f"gc-ms row {source_row_index} is missing feature {feature_name!r}"
                        )
                    try:
                        feature_values.append(float(raw_value))
                    except ValueError as exc:
                        raise GcmsMapError(
                            f"gc-ms row {source_row_index} has non-numeric feature "
                            f"{feature_name!r}={raw_value!r}"
                        ) from exc
                rows.append(
                    GcmsSourceRow(
                        source_row_index=source_row_index,
                        anchor_label=anchor_label,
                        feature_values=tuple(feature_values),
                    )
                )
                labels.append(anchor_label)
    except OSError as exc:
        raise GcmsMapError(f"unable to read gc-ms source asset {resolved_path}: {exc}") from exc
    except csv.Error as exc:
        raise GcmsMapError(f"unable to parse gc-ms source asset {resolved_path}: {exc}") from exc

    if not rows:
        raise GcmsMapError(f"gc-ms source asset has no data rows: {resolved_path}")

    duplicate_labels = sorted(label for label, count in Counter(labels).items() if count > 1)
    if duplicate_labels:
        raise GcmsMapError(f"gc-ms source contains duplicate anchor labels: {duplicate_labels}")

    return GcmsSourceTable(
        resolved_source_path=str(resolved_path),
        label_column=label_column,
        feature_columns=feature_columns,
        rows=tuple(rows),
    )


def build_gcms_class_map(
    *,
    resolved_data_root: str,
    class_vocab: Sequence[str],
    source_table: GcmsSourceTable,
) -> GcmsClassMapManifest:
    class_vocab_tuple = tuple(class_vocab)
    if not class_vocab_tuple:
        raise GcmsMapError("class vocab must be non-empty")
    if len(set(class_vocab_tuple)) != len(class_vocab_tuple):
        raise GcmsMapError(f"class vocab contains duplicates: {list(class_vocab_tuple)}")

    source_rows_by_label = {row.anchor_label: row for row in source_table.rows}
    missing_classes = sorted(set(class_vocab_tuple) - set(source_rows_by_label))
    extra_anchors = sorted(set(source_rows_by_label) - set(class_vocab_tuple))
    violations: list[str] = []
    if missing_classes:
        violations.append(f"missing gc-ms anchors for classes: {missing_classes}")
    if extra_anchors:
        violations.append(f"gc-ms source contains anchors outside base vocab: {extra_anchors}")
    if violations:
        raise GcmsMapError("; ".join(violations))

    mapping_entries = tuple(
        GcmsClassMapEntry(
            class_name=class_name,
            anchor_label=class_name,
            source_row_index=source_rows_by_label[class_name].source_row_index,
        )
        for class_name in class_vocab_tuple
    )
    anchor_labels = [entry.anchor_label for entry in mapping_entries]
    if len(set(anchor_labels)) != len(anchor_labels):
        duplicates = sorted(label for label, count in Counter(anchor_labels).items() if count > 1)
        raise GcmsMapError(f"multiple classes map to the same gc-ms anchor: {duplicates}")

    validation = GcmsValidationResult(
        passed=True,
        bijective_for_scope=True,
        violations=[],
        warnings=[],
    )
    return GcmsClassMapManifest(
        track="exact-upstream",
        usage_contract="benchmark-compatible gc-ms anchor/retrieval contract only",
        resolved_data_root=resolved_data_root,
        resolved_gcms_source_path=source_table.resolved_source_path,
        gcms_label_column=source_table.label_column,
        gcms_feature_columns=source_table.feature_columns,
        class_count=len(class_vocab_tuple),
        anchor_count=source_table.row_count,
        class_vocab=tuple(class_vocab_tuple),
        mapping_entries=mapping_entries,
        validation=validation,
    )


def write_gcms_class_map_manifest(output_path: Path, manifest: GcmsClassMapManifest) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_gcms_class_map_csv(output_path: Path, manifest: GcmsClassMapManifest) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_name", "anchor_label", "source_row_index"])
        for entry in manifest.mapping_entries:
            writer.writerow([entry.class_name, entry.anchor_label, entry.source_row_index])


def resolve_exact_upstream_gcms_csv(repo_root: Path) -> Path:
    candidate = repo_root / ".reference" / "smellnet_upstream" / "data" / "gcms_dataframe.csv"
    if not candidate.is_file():
        raise GcmsMapError(f"expected exact-upstream gc-ms source asset is missing: {candidate}")
    return candidate
