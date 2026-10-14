"""Export scientific evidence into a self-contained React/Plotly glass dashboard."""
# ruff: noqa: E501

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from src.common import ROOT
from src.fatigue_analysis import METRICS


def records(frame: pd.DataFrame) -> list:
    """Preserve missing measurements as JSON null; they must never appear as zero workload."""
    return json.loads(frame.to_json(orient="records", double_precision=15))


def observed_trajectories(metrics_dir: Path, clips: list[str]) -> list[dict]:
    """Thin to one-second map samples, retaining every intervening gap/identity break.

    Display sampling is independent of the native-rate measurements. Connections
    are allowed only if all intervening source observations are valid and share identity.
    """
    paths = []
    for clip in clips:
        source = metrics_dir / clip / "frame_metrics.parquet"
        if not source.exists():
            continue
        frames = pd.read_parquet(
            source,
            columns=[
                "player_id",
                "timestamp",
                "world_x",
                "world_y",
                "movement_valid",
                "identity_segment",
                "speed",
            ],
        )
        for player, part in frames.groupby("player_id"):
            part = part.sort_values("timestamp").reset_index(drop=True)
            points, previous = [], None
            for index in range(0, len(part), 25):
                row = part.iloc[index]
                start = index if previous is None else previous
                interval = part.iloc[start : index + 1]
                continuous = (
                    interval.movement_valid.all()
                    and interval.identity_segment.nunique() == 1
                    and (len(interval) < 2 or interval.timestamp.diff().max() <= 0.15)
                )
                if not continuous:
                    points.append({"t": None, "x": None, "y": None, "speed": None})
                if row.movement_valid and np.isfinite([row.world_x, row.world_y]).all():
                    points.append(
                        {
                            "t": float(row.timestamp / 60),
                            "x": float(row.world_x),
                            "y": float(row.world_y),
                            "speed": float(row.speed) if np.isfinite(row.speed) else None,
                        }
                    )
                previous = index
            paths.append({"clip_name": clip, "player_id": str(player), "points": points})
    return paths


def write_dashboard(
    windows: pd.DataFrame,
    baselines: pd.DataFrame,
    pairs: pd.DataFrame,
    coverage: pd.DataFrame,
    report: dict,
    metrics_dir: Path,
    output: Path,
) -> None:
    """Embed all interpretation, exposure, evaluation and export details beside the plots.

    React changes presentation only. It never recalculates eligibility or interprets
    confidence availability as fatigue accuracy. The result works offline under file://.
    """
    assets = ROOT / "dashboard"
    bundle = assets / "assets/app.js"
    if not bundle.exists():
        raise FileNotFoundError(
            "Build the dashboard once: npm --prefix dashboard ci; npm --prefix dashboard run build"
        )
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

    def read_json(path):
        """Read optional recorded evaluation; absence stays visible instead of fabricated scores."""
        return json.loads(path.read_text("utf-8")) if path.exists() else None

    def read_table(path):
        """Export recorded measurements, never replace missing measurements with zeros."""
        return records(pd.read_csv(path)) if path.exists() else []

    cycles = read_table(metrics_dir / "all_gait_cycles.csv")
    if not cycles:
        for clip in windows.clip_name.unique():
            cycles.extend(read_table(metrics_dir / clip / "gait_cycles.csv"))
    # Filter a global optional cycle table to the actual analysis scope.
    selected_clips, selected_players = set(windows.clip_name), set(windows.player_id)
    cycles = [
        c for c in cycles if c["clip_name"] in selected_clips and c["player_id"] in selected_players
    ]
    calibrations = {}
    for match in {c.split("_")[0] for c in windows.clip_name}:
        value = read_json(ROOT / f"results/phase2/calibration/{match}.json")
        if value:
            calibrations[match] = {
                "method": value["selected_method"],
                "validation": value["validation"][value["selected_method"]],
            }
    trajectories = observed_trajectories(metrics_dir, sorted(selected_clips))
    existing = output / "dashboard.html"
    prior = {}
    if existing.exists():
        match = re.search(
            r'<script id="study-data" type="application/json">(.*?)</script>',
            existing.read_text("utf-8"),
            re.DOTALL,
        )
        if match:
            saved = json.loads(match.group(1))
            if (
                saved.get("created_utc") == report["created_utc"]
                and saved.get("scope") == report["study_scope"]
            ):
                prior = saved
    if not trajectories:
        trajectories = [
            p
            for p in prior.get("trajectories", [])
            if p["clip_name"] in selected_clips and p["player_id"] in selected_players
        ]
    payload = {
        "scope": report["study_scope"],
        "created_utc": report["created_utc"],
        "metrics": [
            {
                "id": m,
                "label": title,
                "unit": unit,
                "direction": direction,
                "family": "movement" if i < 4 else "gait",
            }
            for i, (m, (title, unit, direction)) in enumerate(METRICS.items())
        ],
        "windows": records(windows),
        "baselines": records(baselines),
        "pairs": records(pairs),
        "coverage": records(coverage),
        "analysis": report,
        "cycles": cycles,
        "trajectories": trajectories,
        "trends": read_table(output / "trends.csv"),
        "sensitivity": read_table(output / "sensitivity.csv"),
        "paired_tests": read_json(output / "paired_tests.json"),
        "controls": read_json(output / "synthetic_controls.json"),
        "training": read_json(ROOT / "results/phase2/training/bounded_run.json"),
        "tracking": (
            read_json(ROOT / "results/phase2/autonomous_tracking/report.json")
            or prior.get("tracking")
        ),
        "calibration": calibrations,
        "broadcast": read_json(output / "broadcast_audit/report.json"),
        "broadcast_clips": read_table(output / "broadcast_audit/clips.csv"),
        "broadcast_frames": read_table(output / "broadcast_audit/frames.csv"),
        "broadcast_poses": read_table(output / "broadcast_audit/poses.csv"),
        "study": read_json(Path(report["study"])),
        "exports": {
            name: (output / name).read_text("utf-8")
            for name in (
                "normalized_windows.csv",
                "baselines.csv",
                "early_late_comparison.csv",
                "trends.csv",
                "sensitivity.csv",
                "analysis_report.json",
            )
            if (output / name).exists()
        },
    }
    # Script-block escaping prevents metadata such as </script> from breaking the page.
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    css = (assets / "styles.css").read_text("utf-8")
    client = bundle.read_text("utf-8").replace("</script", "<\\/script")
    plotly = get_plotlyjs().replace("</script", "<\\/script")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark light">
<meta name="description" content="Interactive soccer movement evidence, gait cycles and model evaluation">
<title>STRIDE — Player Movement Lab</title><style>{css}</style></head>
<body><div id="root"><div class="boot">Opening your movement study…</div></div>
<noscript>This interactive dashboard requires JavaScript. The project also supplies CSV tables and static PNG plots.</noscript>
<script id="study-data" type="application/json">{encoded}</script>
<script>{plotly}</script><script>{client}</script></body></html>"""
    (output / "dashboard.html").write_text(page, "utf-8", newline="\n")
