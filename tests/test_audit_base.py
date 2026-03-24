from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures"
EXPECTED_COLUMNS = [
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
]
EXPECTED_CLASSES = [f"odor_{index:02d}" for index in range(50)]


def run_audit_cli(fixture_name: str, emit_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "smelt.datasets.audit_base",
            "--data-root",
            str(FIXTURE_ROOT / fixture_name),
            "--emit",
            str(emit_path),
            "--strict-upstream",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_audit_base_valid_fixture_emits_expected_manifest(tmp_path: Path) -> None:
    emit_path = tmp_path / "smellnet_base.json"
    result = run_audit_cli("smellnet_base_valid", emit_path)
    train_path = FIXTURE_ROOT / "smellnet_base_valid" / "offline_training"
    test_path = FIXTURE_ROOT / "smellnet_base_valid" / "offline_testing"

    assert result.returncode == 0, result.stderr
    manifest = json.loads(emit_path.read_text(encoding="utf-8"))

    assert manifest["resolved_data_root"] == str((FIXTURE_ROOT / "smellnet_base_valid").resolve())
    assert manifest["split_paths"] == {
        "offline_training": str(train_path.resolve()),
        "offline_testing": str(test_path.resolve()),
    }
    assert manifest["strict_upstream"] is True
    assert manifest["class_vocab"] == EXPECTED_CLASSES
    assert manifest["class_count"] == 50
    assert manifest["per_split_class_lists"]["offline_training"] == EXPECTED_CLASSES
    assert manifest["per_split_class_lists"]["offline_testing"] == EXPECTED_CLASSES
    assert manifest["per_class_file_counts"]["offline_training"]["odor_00"] == 5
    assert manifest["per_class_file_counts"]["offline_testing"]["odor_00"] == 1
    assert len(manifest["files"]) == 300
    file_index = {file_entry["relative_path"]: file_entry for file_entry in manifest["files"]}
    assert file_index["offline_testing/odor_00/test_00.csv"] == {
        "relative_path": "offline_testing/odor_00/test_00.csv",
        "split": "offline_testing",
        "class_name": "odor_00",
        "row_count": 3,
        "column_count": 12,
        "column_names": EXPECTED_COLUMNS,
    }
    assert file_index["offline_training/odor_00/train_00.csv"] == {
        "relative_path": "offline_training/odor_00/train_00.csv",
        "split": "offline_training",
        "class_name": "odor_00",
        "row_count": 3,
        "column_count": 12,
        "column_names": EXPECTED_COLUMNS,
    }
    assert manifest["schema_summary"]["globally_consistent"] is True
    assert manifest["schema_summary"]["distinct_schemas"] == [
        {
            "column_count": 12,
            "column_names": EXPECTED_COLUMNS,
            "file_count": 300,
        }
    ]
    assert manifest["benchmark_contract"] == {
        "passed": True,
        "violations": [],
        "warnings": [],
    }


@pytest.mark.parametrize(
    ("fixture_name", "expected_error"),
    [
        ("smellnet_base_invalid_missing_split", "required split directory is missing"),
        ("smellnet_base_invalid_vocab_mismatch", "class vocab mismatch across train/test"),
        ("smellnet_base_invalid_wrong_counts", "strict-upstream file count mismatch"),
        ("smellnet_base_invalid_schema", "inconsistent csv schema"),
    ],
)
def test_audit_base_invalid_fixtures_fail_with_clear_error(
    fixture_name: str,
    expected_error: str,
    tmp_path: Path,
) -> None:
    emit_path = tmp_path / f"{fixture_name}.json"
    result = run_audit_cli(fixture_name, emit_path)

    assert result.returncode == 1
    assert expected_error in result.stderr

    manifest = json.loads(emit_path.read_text(encoding="utf-8"))
    assert manifest["benchmark_contract"]["passed"] is False
    assert any(
        expected_error in violation for violation in manifest["benchmark_contract"]["violations"]
    )
