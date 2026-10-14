"""Refresh the React dashboard from existing evidence without rerunning vision or analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import ROOT
from src.dashboard import write_dashboard


def render_dashboard(
    output: Path = ROOT / "results/phase4", metrics: Path = ROOT / "results/metrics_csv"
) -> dict:
    """Keep empirical results fixed while rebuilding their interactive presentation."""
    report = json.loads((output / "analysis_report.json").read_text("utf-8"))
    write_dashboard(
        pd.read_csv(output / "normalized_windows.csv"),
        pd.read_csv(output / "baselines.csv"),
        pd.read_csv(output / "early_late_comparison.csv"),
        pd.read_csv(metrics / "study_coverage.csv"),
        report,
        metrics,
        output,
    )
    return {
        "status": "completed",
        "dashboard": str(output / "dashboard.html"),
        "inference_rerun": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase4")
    parser.add_argument("--metrics", type=Path, default=ROOT / "results/metrics_csv")
    args = parser.parse_args()
    print(json.dumps(render_dashboard(args.output, args.metrics), indent=2))
