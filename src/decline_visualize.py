"""Static and standalone interactive views of qualified movement decline evidence."""
# ruff: noqa: E501

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fatigue_analysis import METRICS, MOVEMENT


def deliver_visuals(
    windows: pd.DataFrame,
    baselines: pd.DataFrame,
    pairs: pd.DataFrame,
    coverage: pd.DataFrame,
    report: dict,
    metrics_dir: Path,
    output: Path,
) -> None:
    """Expose missingness and confounds alongside physical and normalized measurements.

    Missing gait is displayed as insufficient evidence. Boundaries are drawn only
    when they pass sensitivity checks, because a forced timestamp misrepresents fatigue.
    """
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    players = sorted(windows.player_id.unique())
    colors = dict(zip(players, ["#1778ad", "#b46a10", "#8f45a4"], strict=False))
    for metric, (title, unit, _) in METRICS.items():
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        for (player, _), part in windows.groupby(["player_id", "clip_name"]):
            part = part.sort_values("window_start_time")
            t = (part.window_start_time + part.window_end_time) / 120
            label = player if part.half.iloc[0] == 1 else None
            axes[0].plot(t, part[metric], "o-", color=colors[player], label=label)
            axes[1].plot(t, part[f"{metric}_pct"], "o-", color=colors[player])
        for entry in report["metrics"]:
            if entry["metric"] != metric:
                continue
            for candidate in entry["candidates"]:
                if candidate.get("supported"):
                    for ax in axes:
                        ax.axvline(
                            candidate["boundary_time"] / 60,
                            color=colors[entry["player_id"]],
                            linestyle="--",
                        )
        axes[0].set(title=title, ylabel=unit)
        axes[1].set(
            ylabel="Change from own baseline (%)",
            xlabel="Configured match minute (second-half offset approximate)",
        )
        axes[1].axhline(0, color="gray", linewidth=0.8)
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.axvspan(30, 60, color="gray", alpha=0.12)
        if windows[metric].notna().any():
            axes[0].legend(loc="upper right", fontsize=8)
        else:
            axes[0].text(
                0.5,
                0.5,
                "Insufficient accepted running/cycle exposure\nNo eligible window estimates",
                transform=axes[0].transAxes,
                ha="center",
            )
        fig.text(
            0.5,
            0.005,
            "Gaps remain missing. Shaded interval unprocessed. Movement changes are fatigue proxies, not diagnoses.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.025, 1, 1))
        fig.savefig(plots / f"{metric}_decline.png", dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, metric in zip(axes.flat, MOVEMENT, strict=True):
        selected = pairs[pairs.metric == metric].set_index("player_id").reindex(players)
        x = np.arange(len(players))
        for label, delta, color in (("early", -0.18, "#1778ad"), ("late", 0.18, "#db8b23")):
            ax.bar(x + delta, selected[f"{label}_mean"], width=0.35, color=color, label=label)
            for i, value in enumerate(selected[f"{label}_mean"]):
                if not np.isfinite(value):
                    ax.text(
                        i + delta,
                        0,
                        "insufficient",
                        rotation=90,
                        fontsize=8,
                        va="bottom",
                        ha="center",
                    )
        ax.set(title=METRICS[metric][0], ylabel=METRICS[metric][1], xticks=x, xticklabels=players)
        ax.set_xlim(-0.5, len(players) - 0.5)
        ax.legend()
    fig.suptitle("Qualified early/late segment means (excluded periods stay missing)")
    fig.tight_layout()
    fig.savefig(plots / "early_late_comparison.png", dpi=160)
    plt.close(fig)
    from src.dashboard import write_dashboard

    write_dashboard(windows, baselines, pairs, coverage, report, metrics_dir, output)
