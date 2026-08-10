"""run the m05b benchmark6 grouped-cv moonshot protocol.

m05b mirrors the m05 post-audit protocol exactly, with one change: the bank is
restricted to the benchmark-retained 6 channels. This is the apples-to-apples
channel ablation for the m05 all12 research-extension result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .m05 import build_default_m05b_bank_config_paths, run_m05_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        help="repeat to override the default preserved m05b bank",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/runs"),
    )
    parser.add_argument(
        "--table-root",
        type=Path,
        default=Path("results/tables"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_paths = (
        tuple(path.expanduser().resolve() for path in args.config)
        if args.config
        else build_default_m05b_bank_config_paths()
    )
    result = run_m05_protocol(
        config_paths=config_paths,
        output_root=args.output_root.expanduser().resolve(),
        table_root=args.table_root.expanduser().resolve(),
        protocol_id="m05b",
        required_channel_set="benchmark6",
    )
    final_metrics = result["final_result"].metrics
    selection_payload = result["selection_payload"]
    print(f"run_dir: {result['run_dir']}")
    print(f"selected_method: {selection_payload['selected_method']}")
    print(
        "selected_member_ids: "
        + ",".join(str(value) for value in selection_payload["selected_member_ids"])
    )
    print(f"file_acc@1: {final_metrics.acc_at_1}")
    print(f"file_acc@5: {final_metrics.acc_at_5}")
    print(f"file_precision_macro: {final_metrics.precision_macro}")
    print(f"file_recall_macro: {final_metrics.recall_macro}")
    print(f"file_f1_macro: {final_metrics.f1_macro}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
