"""Convenience wrapper; run with uv run python scripts/check_environment.py."""

import sys

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["check-environment", *sys.argv[1:]]))
