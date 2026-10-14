"""Run the Phase 3 measurement stage without rerunning expensive vision models."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import provenance, write_json
from src.metrics import gait_metrics, movement_metrics, window_metrics
from src.visualize import plot_gait_cycles, plot_measurements


def process_metrics(source: Path, output: Path) -> dict:
    """Persist frame, bout, cycle and window tables with metric-specific availability.

    This stage measures movement and gait proxies. It deliberately does not turn
    them into fatigue diagnoses or the Phase 4 trend/change-point report.
    """
    output.mkdir(parents=True, exist_ok=True)
    frames = pd.read_csv(source)
    frames["player_id"] = frames.player_id.astype(str)
    if "match_seconds" in frames:
        frames["video_timestamp"] = frames["timestamp"]
        frames["timestamp"] = frames["match_seconds"]
    frames = movement_metrics(frames)
    bouts, cycles = gait_metrics(frames)
    frames.to_parquet(output / "frame_metrics.parquet", index=False)
    bouts.to_csv(output / "running_bouts.csv", index=False)
    cycles.to_csv(output / "gait_cycles.csv", index=False)
    plot_gait_cycles(cycles, output / "plots")
    if not cycles.empty:
        cycles["speed_band_mps"] = pd.cut(
            cycles.avg_speed, [2, 3, 4, 5, 6, 9], include_lowest=True
        ).astype(str)
        cycles["source_body_scale_band_px"] = pd.cut(
            cycles.body_scale_source_pixels, [0, 40, 60, 80, 10000]
        ).astype(str)
        cycles["image_motion_axis"] = np.where(
            np.abs(np.cos(np.deg2rad(cycles.image_motion_angle_deg))) >= 0.866,
            "predominantly_horizontal",
            "oblique_or_depth",
        )
        stratified = cycles.groupby(
            [
                "clip_name",
                "player_id",
                "speed_band_mps",
                "source_body_scale_band_px",
                "image_motion_axis",
            ],
            observed=True,
        ).agg(
            cycles=("bout_id", "size"),
            cadence=("stride_frequency", "median"),
            stride_length=("stride_length", "median"),
            knee_excursion=("knee_angle_range", "median"),
            vertical_proxy=("vertical_oscillation", "median"),
        )
        stratified.to_csv(output / "gait_speed_scale_strata.csv")
    window_counts = {}
    for step, label in ((60, "display"), (120, "nonoverlap")):
        windows = window_metrics(frames, bouts, cycles, step=step)
        windows.to_csv(output / f"windowed_metrics_{label}.csv", index=False)
        window_counts[label] = len(windows)
        if not windows.empty:
            for (clip, player), selected in windows.groupby(["clip_name", "player_id"]):
                selected.to_csv(
                    output / f"{clip}_{player}_windowed_metrics_{label}.csv", index=False
                )
        if label == "display" and not windows.empty:
            plot_measurements(frames, windows, output / "plots")
    report = {
        **provenance(),
        "source": str(source),
        "rows": len(frames),
        "movement_valid_fraction": float(frames.movement_valid.mean()),
        "running_bouts": len(bouts),
        "accepted_cycles": len(cycles),
        "window_counts": window_counts,
        "vertical_oscillation_unit": "normalized_image_proxy",
        "minimum_gait_body_height_source_px": 40,
        "gait_view_qualification": "Speed/body-scale/image-motion strata; 3D facing unavailable",
        "fatigue_analysis_performed": False,
    }
    write_json(output / "metrics_report.json", report)
    return report
