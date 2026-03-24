"""minimal command line entrypoint for smelt."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ._version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smelt", description="smelt bootstrap cli")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    print("smelt bootstrap ready")
    return 0
