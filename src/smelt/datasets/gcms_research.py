"""research-extension helpers for explicit gc-ms anchor usage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .gcms_map import GcmsClassMapEntry, load_gcms_source_table


class ResearchGcmsError(Exception):
    """raised when research gc-ms assets cannot be loaded safely."""


@dataclass(slots=True)
class ResearchGcmsManifest:
    resolved_manifest_path: str
    resolved_source_path: str
    feature_columns: tuple[str, ...]
    class_vocab: tuple[str, ...]
    mapping_entries: tuple[GcmsClassMapEntry, ...]


@dataclass(slots=True)
class ResearchGcmsAnchorSet:
    class_names: tuple[str, ...]
    anchor_labels: tuple[str, ...]
    feature_columns: tuple[str, ...]
    feature_matrix: NDArray[np.float32]
    source_row_indices: tuple[int, ...]
    resolved_manifest_path: str
    resolved_source_path: str

    @property
    def anchor_count(self) -> int:
        return int(self.feature_matrix.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.feature_matrix.shape[1])

    def to_artifact_dict(self) -> dict[str, object]:
        return {
            "class_names": list(self.class_names),
            "anchor_labels": list(self.anchor_labels),
            "feature_columns": list(self.feature_columns),
            "anchor_count": self.anchor_count,
            "feature_count": self.feature_count,
            "source_row_indices": list(self.source_row_indices),
            "resolved_manifest_path": self.resolved_manifest_path,
            "resolved_source_path": self.resolved_source_path,
        }


def load_research_gcms_anchor_set(
    manifest_path: Path,
    *,
    class_names: tuple[str, ...],
) -> ResearchGcmsAnchorSet:
    manifest = load_research_gcms_manifest(manifest_path)
    if tuple(class_names) != manifest.class_vocab:
        raise ResearchGcmsError(
            "class_names do not match gc-ms manifest vocab: "
            f"{list(class_names)} vs {list(manifest.class_vocab)}"
        )
    source_table = load_gcms_source_table(Path(manifest.resolved_source_path))
    rows_by_index = {row.source_row_index: row for row in source_table.rows}
    entries_by_class = {entry.class_name: entry for entry in manifest.mapping_entries}

    try:
        ordered_entries = [entries_by_class[class_name] for class_name in class_names]
    except KeyError as exc:
        raise ResearchGcmsError(f"gc-ms manifest is missing class {exc.args[0]!r}") from exc

    try:
        feature_matrix = np.asarray(
            [rows_by_index[entry.source_row_index].feature_values for entry in ordered_entries],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise ResearchGcmsError(
            f"gc-ms source is missing mapped row index {exc.args[0]!r}"
        ) from exc
    if feature_matrix.ndim != 2:
        raise ResearchGcmsError(
            f"expected 2d gc-ms feature matrix, found shape {feature_matrix.shape}"
        )
    return ResearchGcmsAnchorSet(
        class_names=tuple(class_names),
        anchor_labels=tuple(entry.anchor_label for entry in ordered_entries),
        feature_columns=manifest.feature_columns,
        feature_matrix=feature_matrix,
        source_row_indices=tuple(entry.source_row_index for entry in ordered_entries),
        resolved_manifest_path=manifest.resolved_manifest_path,
        resolved_source_path=manifest.resolved_source_path,
    )


def write_research_gcms_anchor_usage(output_path: Path, anchor_set: ResearchGcmsAnchorSet) -> None:
    output_path.write_text(
        json.dumps(anchor_set.to_artifact_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_research_gcms_manifest(manifest_path: Path) -> ResearchGcmsManifest:
    resolved_path = manifest_path.expanduser().resolve()
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ResearchGcmsError(f"unable to read gc-ms manifest: {resolved_path}") from exc
    except json.JSONDecodeError as exc:
        raise ResearchGcmsError(f"invalid gc-ms manifest json: {resolved_path}") from exc
    if not isinstance(payload, dict):
        raise ResearchGcmsError("gc-ms manifest payload must be a mapping")
    validation = payload.get("validation", {})
    if not isinstance(validation, dict) or not bool(validation.get("passed")):
        raise ResearchGcmsError("gc-ms manifest validation must pass before research pretraining")
    try:
        entries = tuple(
            GcmsClassMapEntry(
                class_name=str(entry["class_name"]),
                anchor_label=str(entry["anchor_label"]),
                source_row_index=int(entry["source_row_index"]),
            )
            for entry in payload["mapping_entries"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ResearchGcmsError("gc-ms manifest mapping_entries are malformed") from exc
    try:
        return ResearchGcmsManifest(
            resolved_manifest_path=str(resolved_path),
            resolved_source_path=str(payload["resolved_gcms_source_path"]),
            feature_columns=tuple(str(name) for name in payload["gcms_feature_columns"]),
            class_vocab=tuple(str(name) for name in payload["class_vocab"]),
            mapping_entries=entries,
        )
    except KeyError as exc:
        raise ResearchGcmsError(f"gc-ms manifest is missing key {exc.args[0]!r}") from exc
