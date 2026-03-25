"""contracts for the exact-upstream base audit."""

from __future__ import annotations

from dataclasses import dataclass, field

TRAIN_SPLIT = "offline_training"
TEST_SPLIT = "offline_testing"
SPLIT_NAMES = (TRAIN_SPLIT, TEST_SPLIT)
RAW_SENSOR_COLUMNS = (
    "NO2",
    "C2H5OH",
    "VOC",
    "CO",
    "Alcohol",
    "LPG",
    "Benzene",
    "Temperature",
    "Pressure",
    "Humidity",
    "Gas_Resistance",
    "Altitude",
)
BENCHMARK_SENSOR_COLUMNS = (
    "NO2",
    "C2H5OH",
    "VOC",
    "CO",
    "Alcohol",
    "LPG",
)


@dataclass(slots=True)
class SensorFileRecord:
    split: str
    class_name: str
    relative_path: str
    absolute_path: str
    column_names: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.column_names)


@dataclass(slots=True)
class BaseSensorDataset:
    resolved_data_root: str
    raw_column_names: tuple[str, ...]
    train_records: tuple[SensorFileRecord, ...]
    test_records: tuple[SensorFileRecord, ...]

    @property
    def class_vocab(self) -> tuple[str, ...]:
        class_names = {record.class_name for record in self.train_records}
        class_names.update(record.class_name for record in self.test_records)
        return tuple(sorted(class_names))

    def records_for_split(self, split_name: str) -> tuple[SensorFileRecord, ...]:
        if split_name == TRAIN_SPLIT:
            return self.train_records
        if split_name == TEST_SPLIT:
            return self.test_records
        raise ValueError(f"unknown split: {split_name}")


@dataclass(slots=True)
class FileMetadata:
    relative_path: str
    split: str
    class_name: str
    row_count: int
    column_count: int
    column_names: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "split": self.split,
            "class_name": self.class_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_names": self.column_names,
        }


@dataclass(slots=True)
class SchemaMetadata:
    column_count: int
    column_names: list[str]
    file_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "column_count": self.column_count,
            "column_names": self.column_names,
            "file_count": self.file_count,
        }


@dataclass(slots=True)
class SchemaSummary:
    distinct_schemas: list[SchemaMetadata] = field(default_factory=list)
    globally_consistent: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "distinct_schemas": [schema.to_dict() for schema in self.distinct_schemas],
            "globally_consistent": self.globally_consistent,
        }


@dataclass(slots=True)
class BenchmarkContractResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class AuditManifest:
    resolved_data_root: str
    split_paths: dict[str, str]
    strict_upstream: bool
    class_vocab: list[str]
    class_count: int
    per_split_class_lists: dict[str, list[str]]
    per_class_file_counts: dict[str, dict[str, int]]
    files: list[FileMetadata]
    schema_summary: SchemaSummary
    benchmark_contract: BenchmarkContractResult

    def to_dict(self) -> dict[str, object]:
        return {
            "resolved_data_root": self.resolved_data_root,
            "split_paths": self.split_paths,
            "strict_upstream": self.strict_upstream,
            "class_vocab": self.class_vocab,
            "class_count": self.class_count,
            "per_split_class_lists": self.per_split_class_lists,
            "per_class_file_counts": self.per_class_file_counts,
            "files": [file_metadata.to_dict() for file_metadata in self.files],
            "schema_summary": self.schema_summary.to_dict(),
            "benchmark_contract": self.benchmark_contract.to_dict(),
        }
