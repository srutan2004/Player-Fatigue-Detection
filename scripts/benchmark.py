"""Convenience wrapper for a configured real-video feasibility pilot."""

import sys

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["pilot", *sys.argv[1:]]))
