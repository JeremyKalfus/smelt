"""minimal environment check script for smelt."""

from smelt import __version__


def main() -> int:
    print(f"smelt {__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
