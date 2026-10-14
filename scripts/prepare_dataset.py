"""Convenience wrapper for the bounded SoccerTrack intake commands."""

import sys

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["dataset-probe"]))
