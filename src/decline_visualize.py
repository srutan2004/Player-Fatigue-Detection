"""Static and standalone interactive views of qualified movement decline evidence."""
# ruff: noqa: E501

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    figures = []
    for player in players:
        group = windows[windows.player_id == player]
        fig = make_subplots(
            rows=8,
            cols=2,
            subplot_titles=[
                text
                for m in METRICS
                for text in (METRICS[m][0] + " | " + METRICS[m][1], "Own baseline change (%)")
            ],
            vertical_spacing=0.025,
        )
        for row, metric in enumerate(METRICS, 1):
            for clip, part in group.groupby("clip_name"):
                part = part.sort_values("window_start_time")
                t = ((part.window_start_time + part.window_end_time) / 120).to_numpy()
                for col, field in ((1, metric), (2, f"{metric}_pct")):
                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=part[field],
                            name=clip,
                            mode="lines+markers",
                            connectgaps=False,
                            showlegend=row == 1 and col == 1,
                            customdata=part[
                                ["position_coverage", "running_seconds", "valid_cycles"]
                            ],
                            hovertemplate="minute %{x:.1f}<br>value %{y:.3f}<br>coverage %{customdata[0]:.1%}<br>accepted running %{customdata[1]:.1f}s<br>cycles %{customdata[2]}<extra>%{fullData.name}</extra>",
                        ),
                        row=row,
                        col=col,
                    )
            for col, field in ((1, metric), (2, f"{metric}_pct")):
                if not group[field].notna().any():
                    fig.add_annotation(
                        text=(
                            "Unavailable own-player baseline"
                            if col == 2
                            else "Insufficient eligible observations"
                        ),
                        x=0.5,
                        y=0.5,
                        xref=f"x{(row - 1) * 2 + col} domain"
                        if (row - 1) * 2 + col > 1
                        else "x domain",
                        yref=f"y{(row - 1) * 2 + col} domain"
                        if (row - 1) * 2 + col > 1
                        else "y domain",
                        showarrow=False,
                    )
            for entry in report["metrics"]:
                if entry["player_id"] == player and entry["metric"] == metric:
                    for candidate in entry["candidates"]:
                        if candidate.get("supported"):
                            fig.add_vline(
                                x=candidate["boundary_time"] / 60,
                                line_dash="dash",
                                row=row,
                                col="all",
                            )
        fig.update_layout(
            height=2000,
            template="plotly_white",
            title=f"{player}: observed movement and gait availability",
            margin=dict(l=65, r=35, t=100, b=45),
        )
        fig.update_xaxes(title_text="Configured match minute", row=8)
        figures.append((player, fig))
    pitch = go.Figure()
    for c in sorted(windows.clip_name.unique()):
        source = pd.read_parquet(
            metrics_dir / c / "frame_metrics.parquet",
            columns=[
                "player_id",
                "timestamp",
                "world_x",
                "world_y",
                "movement_valid",
                "identity_segment",
            ],
        )
        for player, part in source.groupby("player_id"):
            # Insert separators on actual gaps/switches before thinning, so a map never
            # draws displacement across missing intervals or identity transitions.
            x, y, times = [], [], []
            last = None
            for r in part.iloc[::25].itertuples():
                if not r.movement_valid or (
                    last is not None
                    and (
                        r.identity_segment != last.identity_segment
                        or r.timestamp - last.timestamp > 1.1
                    )
                ):
                    x.append(None)
                    y.append(None)
                    times.append(None)
                if r.movement_valid:
                    # Also check intervening native observations before connecting.
                    interval = (
                        part[(part.timestamp > last.timestamp) & (part.timestamp <= r.timestamp)]
                        if last is not None
                        else part.iloc[:1]
                    )
                    if not interval.movement_valid.all():
                        x.append(None)
                        y.append(None)
                        times.append(None)
                    x.append(r.world_x)
                    y.append(r.world_y)
                    times.append(r.timestamp / 60)
                last = r
            pitch.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=f"{player} / {c}",
                    mode="lines",
                    connectgaps=False,
                    customdata=times,
                    hovertemplate="x %{x:.1f}m; y %{y:.1f}m<br>minute %{customdata:.1f}<extra>%{fullData.name}</extra>",
                )
            )
    pitch.add_shape(type="rect", x0=0, y0=0, x1=105, y1=68, line_color="green")
    pitch.add_shape(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line_color="green")
    pitch.update_layout(
        title="Independent measured ground trajectories (1-second display thinning)",
        height=540,
        template="plotly_white",
    )
    pitch.update_yaxes(scaleanchor="x", scaleratio=1, title="Field y (m)")
    pitch.update_xaxes(title="Field x (m)")
    statuses = pd.DataFrame(
        [
            {
                k: r[k]
                for k in (
                    "player_id",
                    "metric",
                    "status",
                    "baseline_status",
                    "qualified_physical_windows",
                    "eligible_windows",
                    "longest_continuous_windows",
                )
            }
            for r in report["metrics"]
        ]
    )
    statuses.to_csv(output / "analysis_status.csv", index=False)
    options = "".join(
        f'<option value="{html.escape(p)}">{html.escape(p)}</option>' for p in players
    )
    content = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Football movement evidence</title><style>body{font:16px system-ui;margin:24px auto;max-width:1250px;padding:0 18px;color:#173044}table{border-collapse:collapse;font-size:12px;width:100%;margin:15px 0}td,th{padding:7px;border:1px solid #ddd;text-align:left}.panel{display:none}.active{display:block}.notice{background:#edf4f8;padding:18px;border-radius:8px}select{padding:9px;font-size:16px}details{margin:18px 0}</style></head><body>",
        f"<h1>Football movement: evidence and availability</h1><p class='notice'>{report['study_scope']['distinct_minutes']:g} minutes, {report['study_scope']['matches']} matches, {report['study_scope']['players']} players. This is proxy analysis, not a fatigue diagnosis. Early and late are selected segments, not complete halves. No sensor fatigue labels are available. Substitution context: {html.escape(report['substitution_metadata'])}. Missing gait estimates and unobserved match intervals remain missing.</p>",
        f"<label for='player'>Player </label><select id='player'>{options}</select>",
    ]
    for i, (player, fig) in enumerate(figures):
        content.append(f'<section id="{player}" class="panel {"active" if i == 0 else ""}">')
        content.append(statuses[statuses.player_id == player].to_html(index=False))
        content.append(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False))
        content.append("</section>")
    content.extend(
        [
            "<details><summary>Baseline and exposure tables</summary>",
            baselines.to_html(index=False),
            coverage.to_html(index=False),
            "</details>",
            "<h2>Qualified early/late comparison</h2><p>Only eligible periods are averaged. Statistical unit: player. Fewer than three qualifying pairs are insufficient for the configured exploratory test.</p>",
            pairs.to_html(index=False),
            pitch.to_html(full_html=False, include_plotlyjs=False),
            "<script>document.getElementById('player').addEventListener('change',function(){document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===this.value));window.dispatchEvent(new Event('resize'));});</script></body></html>",
        ]
    )
    (output / "dashboard.html").write_text("\n".join(content), "utf-8")
