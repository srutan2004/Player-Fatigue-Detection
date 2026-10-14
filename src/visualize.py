"""Quality-aware movement plots for the Phase 3 measurement study."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_measurements(frames: pd.DataFrame, windows: pd.DataFrame, output: Path) -> list[str]:
    """Plot measured windows and missingness without inventing fatigue boundaries.

    Metric availability is part of the result. NaNs break lines, and ground paths
    are broken at invalid observations so a disappearance cannot look like travel.
    """
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    metrics = {
        "avg_speed": "Average movement speed (m/s)",
        "peak_speed": "Sustained 1-second peak speed (m/s)",
        "stride_frequency": "Same-leg cadence (strides/s)",
        "stride_length": "Ground-path stride-length proxy (m)",
        "vertical_oscillation": "Vertical image-motion proxy (body-height fraction)",
        "knee_angle_range": "Projected knee excursion (degrees)",
    }
    for metric, label in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        for (clip, player), part in windows.groupby(["clip_name", "player_id"]):
            ax.plot(
                (part.window_start_time + part.window_end_time) / 120,
                part[metric],
                "o-",
                markersize=3,
                label=f"{player} / {clip}",
            )
        if windows.empty or windows[metric].notna().sum() == 0:
            ax.text(
                0.5, 0.5, "Insufficient eligible observations", ha="center", transform=ax.transAxes
            )
        ax.set(
            xlabel="Configured match time (minutes)",
            ylabel=label,
            title="Phase 3 measurements — quality-gated observations",
        )
        ax.grid(alpha=0.2)
        if not windows.empty:
            ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        path = output / f"{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))
    for (clip, player), part in frames.groupby(["clip_name", "player_id"]):
        fig, (ax, coverage) = plt.subplots(2, 1, figsize=(10, 8))
        ax.set(
            xlim=(-3, 108),
            ylim=(-3, 71),
            aspect="equal",
            xlabel="Pitch x (m)",
            ylabel="Pitch y (m)",
            title=f"{clip} / {player} — observed ground trajectory",
        )
        ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color="black")
        ax.axvline(52.5, color="gray", lw=0.7)
        xy = part[["world_x", "world_y"]].where(part.movement_valid, np.nan)
        ax.plot(xy.world_x, xy.world_y, lw=0.5, alpha=0.65)
        minute = np.floor((part.timestamp - part.timestamp.min()) / 60)
        quality = part.groupby(minute)[
            ["position_valid", "movement_valid", "gait_pose_valid"]
        ].mean()
        quality.plot(ax=coverage, marker="o")
        coverage.set(
            xlabel="Elapsed segment minutes", ylabel="Observed frame fraction", ylim=(0, 1.05)
        )
        coverage.grid(alpha=0.2)
        fig.tight_layout()
        path = output / f"{clip}_{player}_trajectory_coverage.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))
    return saved


def plot_gait_cycles(cycles: pd.DataFrame, output: Path) -> list[str]:
    """Show descriptive accepted cycles even when window-level exposure is insufficient.

    Scatter points are correlated within bouts and are not independent fatigue
    evidence. Speed-versus-mechanics plots expose a major comparison confound.
    """
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    labels = {
        "stride_frequency": "Same-leg cadence (strides/s)",
        "stride_length": "Ground-path stride proxy (m)",
        "vertical_oscillation": "Vertical image proxy (body-height fraction)",
        "knee_angle_range": "Projected knee excursion (degrees)",
    }
    for metric, label in labels.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        for (player, leg), part in cycles.groupby(["player_id", "leg"]):
            ax.scatter(
                (part.start_time + part.end_time) / 120,
                part[metric],
                s=13,
                alpha=0.65,
                marker="o" if leg == "left" else "^",
                label=f"{player} / {leg}",
            )
        if cycles.empty:
            ax.text(0.5, 0.5, "No accepted cycles", ha="center", transform=ax.transAxes)
        else:
            ax.legend(fontsize=7)
        ax.set(
            xlabel="Configured match time (minutes)",
            ylabel=label,
            title="Descriptive accepted cycles — window eligibility remains separate",
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = output / f"cycle_{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))
    for metric in ("stride_frequency", "knee_angle_range"):
        fig, ax = plt.subplots(figsize=(8, 4))
        for player, part in cycles.groupby("player_id"):
            ax.scatter(part.avg_speed, part[metric], s=12, alpha=0.65, label=player)
        if not cycles.empty:
            ax.legend(fontsize=8)
        ax.set(
            xlabel="Observed cycle-average speed (m/s)",
            ylabel=labels[metric],
            title="Speed is a comparison confound — descriptive cycles",
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        path = output / f"speed_vs_{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path))
    return saved
