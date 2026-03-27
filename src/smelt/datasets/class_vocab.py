"""deterministic class vocabulary helpers for the exact-upstream base split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import TEST_SPLIT, TRAIN_SPLIT, BaseSensorDataset


@dataclass(slots=True)
class BaseClassVocabManifest:
    resolved_data_root: str
    split_names: tuple[str, str]
    class_count: int
    class_vocab: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "resolved_data_root": self.resolved_data_root,
            "split_names": list(self.split_names),
            "class_count": self.class_count,
            "class_vocab": list(self.class_vocab),
        }


def extract_base_class_vocab(dataset: BaseSensorDataset) -> tuple[str, ...]:
    train_vocab = tuple(sorted({record.class_name for record in dataset.train_records}))
    test_vocab = tuple(sorted({record.class_name for record in dataset.test_records}))
    if train_vocab != test_vocab:
        raise ValueError(
            "train/test class vocab mismatch while extracting base vocab: "
            f"{train_vocab} vs {test_vocab}"
        )
    return train_vocab


def build_base_class_vocab_manifest(dataset: BaseSensorDataset) -> BaseClassVocabManifest:
    class_vocab = extract_base_class_vocab(dataset)
    return BaseClassVocabManifest(
        resolved_data_root=dataset.resolved_data_root,
        split_names=(TRAIN_SPLIT, TEST_SPLIT),
        class_count=len(class_vocab),
        class_vocab=class_vocab,
    )


def write_base_class_vocab_manifest(output_path: Path, manifest: BaseClassVocabManifest) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
