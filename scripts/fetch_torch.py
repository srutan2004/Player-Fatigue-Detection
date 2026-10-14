"""Fetch the pinned official Windows wheel with exact hash verification."""

from pathlib import Path

from src.download import fetch_ranges

if __name__ == "__main__":
    fetch_ranges(
        "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl",
        Path(".cache/wheels/torch-2.8.0+cu128-cp312-cp312-win_amd64.whl"),
        3461384651,
        "0ad925202387f4e7314302a1b4f8860fa824357f9b1466d7992bf276370ebcff",
    )
