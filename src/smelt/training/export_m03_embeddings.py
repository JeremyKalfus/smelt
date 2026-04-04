"""export frozen moonshot embeddings for m03 without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

from smelt.evaluation import write_json

from .m03 import export_locked_moonshot_run_embeddings, select_primary_encoder_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--primary-selection-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dirs = tuple(sorted(path.expanduser().resolve() for path in args.run_dir))
    for run_dir in run_dirs:
        export_dir = export_locked_moonshot_run_embeddings(
            run_dir=run_dir,
            output_root=args.output_root.expanduser().resolve(),
        )
        print(f"export_dir[{run_dir.name}]: {export_dir}")
    selection = select_primary_encoder_run(run_dirs)
    selection_path = args.primary_selection_json.expanduser().resolve()
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(selection_path, selection.to_dict())
    print(f"primary_selection_json: {selection_path}")
    print(f"selected_run_id: {selection.selected_run_id}")
    print(f"locked_primary_aggregator: {selection.locked_primary_aggregator}")
    print(f"validation_file_acc@1: {selection.validation_file_acc_at_1}")
    print(f"validation_file_macro_f1: {selection.validation_file_macro_f1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
