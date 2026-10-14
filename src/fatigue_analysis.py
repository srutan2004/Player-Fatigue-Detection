"""Conservative movement-decline inference; physiological fatigue is not observable."""

from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.common import ROOT, provenance, write_json

METRICS = {
    "avg_speed": ("Average speed", "m/s", -1),
    "peak_speed": ("Sustained peak speed", "m/s", -1),
    "p90_running_speed": ("Running speed, 90th percentile", "m/s", -1),
    "high_intensity_distance_per_minute": ("High-intensity distance rate", "m/observed min", -1),
    "stride_frequency": ("Same-leg cadence", "cycles/s", 0),
    "stride_length": ("Stride-length proxy", "m/cycle", -1),
    "vertical_oscillation": ("Projected vertical-motion proxy", "body-height fraction", 0),
    "knee_angle_range": ("Knee flexion excursion", "degrees", -1),
}
MOVEMENT = list(METRICS)[:4]


def attach_context(windows: pd.DataFrame, study: dict) -> pd.DataFrame:
    """Attach actual clip intervals so elapsed baselines do not imply match kickoff."""
    context = pd.DataFrame(
        [
            {
                "clip_name": c["clip_name"],
                "half": c["half"],
                "clip_start_time": c["start_seconds"] + c["match_time_offset_seconds"],
                "clip_duration": c["duration_seconds"],
            }
            for c in study["clips"]
        ]
    )
    return windows.merge(context, on="clip_name", validate="many_to_one")


def baseline_normalize(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use qualified observations inside the first 600 elapsed seconds of the early clip.

    Own-player normalization reduces scale differences, but cannot remove tactical
    workload, projection or missingness bias. Zero baselines have no valid percent change.
    """
    data = windows.copy()
    rows = []
    for player, group in data.groupby("player_id"):
        early = group[
            (group.half == 1)
            & (group.window_start_time >= group.clip_start_time)
            & (group.window_end_time <= group.clip_start_time + 600)
        ]
        for metric in METRICS:
            selected = early[early[metric].notna()]
            exposure = "valid_seconds" if metric in MOVEMENT else "running_seconds"
            seconds = float(selected[exposure].sum())
            required = 420 if metric in MOVEMENT else 60
            value = None
            reason = "insufficient_baseline_exposure"
            if len(selected) >= 3 and seconds >= required:
                candidate = float(np.average(selected[metric], weights=selected[exposure]))
                if abs(candidate) > 1e-6:
                    value, reason = candidate, "available"
                else:
                    reason = "near_zero_baseline"
            mask = data.player_id.eq(player)
            data.loc[mask, f"{metric}_pct"] = (
                (data.loc[mask, metric] / value - 1) * 100 if value is not None else np.nan
            )
            rows.append(
                {
                    "player_id": player,
                    "metric": metric,
                    "baseline": value,
                    "baseline_status": reason,
                    "eligible_windows": len(selected),
                    "observed_seconds": seconds,
                    "interval": "first_600_elapsed_seconds_of_selected_early_clip",
                }
            )
    return data, pd.DataFrame(rows)


def continuous_chunks(data: pd.DataFrame, column: str) -> list[pd.DataFrame]:
    """Break at missing windows and clip boundaries; unobserved match time is never filled."""
    chunks = []
    for _, clip in data.groupby("clip_name", sort=False):
        clip = clip.sort_values("window_start_time")
        valid = np.isfinite(clip[column])
        boundary = (
            ~valid
            | ~valid.shift(fill_value=False)
            | ((clip.window_start_time - clip.window_end_time.shift()).abs() > 1e-6)
        )
        chunks.extend(part for _, part in clip[valid].groupby(boundary.cumsum()[valid]))
    return chunks


def analyze_series(data: pd.DataFrame, metric: str, penalty_scale: float = 3) -> dict:
    """Fit HAC descriptive trends and stable, material downward PELT boundaries.

    Six continuous nonoverlapping windows support a descriptive trend; eight support
    change-point search. A candidate needs three observed windows on either side,
    a >=10-baseline-percentage-point drop and agreement across two of three penalties.
    These thresholds are design rules, not clinical sensitivity/specificity estimates.
    """
    column = f"{metric}_pct"
    chunks = continuous_chunks(data, column)
    trends, candidates, searches = [], [], 0
    for part in chunks:
        n = len(part)
        y = part[column].to_numpy(float)
        if n >= 6:
            t = (part.window_start_time.to_numpy() - part.window_start_time.iloc[0]) / 60
            model = sm.OLS(y, sm.add_constant(t)).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
            ci = model.conf_int()[1]
            trends.append(
                {
                    "clip_name": part.clip_name.iloc[0],
                    "windows": n,
                    "start_time": float(part.window_start_time.iloc[0]),
                    "end_time": float(part.window_end_time.iloc[-1]),
                    "slope_pct_points_per_minute": float(model.params[1]),
                    "ci_low": float(ci[0]),
                    "ci_high": float(ci[1]),
                    "p_value": float(model.pvalues[1]) if np.isfinite(model.pvalues[1]) else None,
                }
            )
        if n < 8 or METRICS[metric][2] != -1:
            continue
        searches += 1
        # First differences estimate local noise without inflating it with a genuine step.
        difference = np.diff(y)
        noise = float(1.4826 * np.median(np.abs(difference - np.median(difference))) / np.sqrt(2))
        scale = max(noise, 2.0)
        signal = ((y - np.median(y)) / scale).reshape(-1, 1)
        boundaries = []
        for factor in (penalty_scale * 2 / 3, penalty_scale, penalty_scale * 4 / 3):
            fitted = rpt.Pelt(model="l2", min_size=3, jump=1).fit(signal)
            boundaries.append(fitted.predict(pen=factor * np.log(n))[:-1])
        for boundary in sorted(set(sum(boundaries, []))):
            if boundary < 3 or n - boundary < 3:
                continue
            votes = sum(any(abs(b - boundary) <= 1 for b in values) for values in boundaries)
            # Compare adjacent observations, avoiding averaging over unrelated remote changes.
            before, after = y[boundary - 3 : boundary], y[boundary : boundary + 3]
            change = float(np.median(after) - np.median(before))
            if votes >= 2 and change <= -max(10.0, 2 * noise):
                candidates.append(
                    {
                        "clip_name": part.clip_name.iloc[0],
                        "boundary_time": float(part.window_start_time.iloc[boundary]),
                        "uncertainty_seconds": float(
                            part.window_end_time.iloc[boundary]
                            - part.window_start_time.iloc[boundary]
                        ),
                        "drop_pct_points": change,
                        "penalty_votes": votes,
                        "local_noise_pct_points": noise,
                        "sequence_windows": n,
                    }
                )
    return {
        "metric": metric,
        "eligible_windows": sum(map(len, chunks)),
        "longest_continuous_windows": max(map(len, chunks), default=0),
        "trends": trends,
        "candidates": candidates,
        "searched_sequences": searches,
        "status": "candidate_pending_sensitivity"
        if candidates
        else "no_supported_change"
        if searches
        else "insufficient_data",
    }


def paired_comparison(windows: pd.DataFrame, coverage: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Compare early/late periods per player; windows are not independent subjects.

    Require four eligible windows and 70% full-clip movement exposure in each period.
    Very small paired cohorts cannot establish a population fatigue effect.
    """
    rows, statistics = [], []
    for player, group in windows.groupby("player_id"):
        for metric in METRICS:
            row = {
                "player_id": player,
                "metric": metric,
                "early_mean": np.nan,
                "late_mean": np.nan,
                "change_pct": np.nan,
                "paired_eligible": False,
            }
            eligible = []
            for half, label in ((1, "early"), (2, "late")):
                part = group[group.half == half]
                selected = part[part[metric].notna()]
                cov = coverage[(coverage.player_id == player) & (coverage.half == half)]
                exposure = "valid_seconds" if metric in MOVEMENT else "running_seconds"
                fraction = (
                    float(cov.valid_movement_seconds.sum() / part.clip_duration.max())
                    if len(part)
                    else 0
                )
                ok = len(selected) >= 4 and (
                    fraction >= 0.7 if metric in MOVEMENT else selected.running_seconds.sum() >= 80
                )
                row[f"{label}_qualified_windows"] = len(selected)
                row[f"{label}_movement_coverage"] = fraction
                row[f"{label}_observed_seconds"] = float(selected[exposure].sum())
                if ok:
                    row[f"{label}_mean"] = float(
                        np.average(selected[metric], weights=selected[exposure])
                    )
                eligible.append(ok)
            row["paired_eligible"] = all(eligible)
            row["reason"] = (
                "qualified_pair"
                if all(eligible)
                else "insufficient_eligible_windows_or_period_coverage"
            )
            if all(eligible) and abs(row["early_mean"]) > 1e-6:
                row["change_pct"] = (row["late_mean"] / row["early_mean"] - 1) * 100
            rows.append(row)
    table = pd.DataFrame(rows)
    for metric, part in table.groupby("metric", sort=False):
        selected = part[part.paired_eligible]
        result = {
            "metric": metric,
            "paired_players": len(selected),
            "test": "Wilcoxon signed-rank",
            "p_value": None,
            "status": "insufficient_paired_players",
            "unit": "player",
        }
        if len(selected) >= 3:
            differences = selected.late_mean - selected.early_mean
            p = (
                1.0
                if np.allclose(differences, 0)
                else float(wilcoxon(differences, method="exact").pvalue)
            )
            result.update(p_value=p, status="exploratory_only")
        statistics.append(result)
    valid = [s for s in statistics if s["p_value"] is not None]
    if valid:
        adjusted = multipletests([s["p_value"] for s in valid], method="holm")[1]
        for entry, p in zip(valid, adjusted, strict=True):
            entry["holm_p_value"] = float(p)
    return table, statistics


def synthetic_controls() -> list[dict]:
    """Check known steps and nulls as software controls, not real fatigue labels."""
    rng = np.random.default_rng(42)
    results = []
    for name, values in (
        ("known_downward_step", np.r_[np.zeros(12), np.full(12, -25)] + rng.normal(0, 1, 24)),
        ("stationary_noise", rng.normal(0, 1, 24)),
        ("constant", np.zeros(24)),
    ):
        series = pd.DataFrame(
            {
                "clip_name": "synthetic",
                "window_start_time": np.arange(24) * 120,
                "window_end_time": (np.arange(24) + 1) * 120,
                "avg_speed_pct": values,
            }
        )
        output = analyze_series(series, "avg_speed")
        expected = name == "known_downward_step"
        passed = bool(output["candidates"]) == expected
        if expected:
            passed &= any(abs(c["boundary_time"] - 1440) <= 120 for c in output["candidates"])
        results.append(
            {
                "name": name,
                "passed": passed,
                "expected_change": expected,
                "candidates": output["candidates"],
            }
        )
    return results


def run_analysis(
    study_path: Path = ROOT / "data/manifests/study.json",
    metrics_dir: Path = ROOT / "results/metrics_csv",
    output: Path = ROOT / "results/phase4",
    sensitivity: bool = True,
) -> dict:
    """Run validated, reproducible proxy-decline analysis on completed real measurements."""
    import json

    from src.decline_visualize import deliver_visuals
    from src.metrics import movement_metrics, window_metrics

    output.mkdir(parents=True, exist_ok=True)
    study = json.loads(study_path.read_text("utf-8"))
    windows = attach_context(
        pd.read_csv(metrics_dir / "all_windowed_metrics_nonoverlap.csv"), study
    )
    windows, baselines = baseline_normalize(windows)
    coverage = pd.read_csv(metrics_dir / "study_coverage.csv")
    results = []
    for player, group in windows.groupby("player_id"):
        for metric in METRICS:
            result = analyze_series(group, metric)
            result["player_id"] = player
            base = baselines[(baselines.player_id == player) & (baselines.metric == metric)].iloc[0]
            result["qualified_physical_windows"] = int(group[metric].notna().sum())
            result["baseline_status"] = base.baseline_status
            if base.baseline_status != "available":
                result["status"] = (
                    "unavailable_baseline"
                    if result["qualified_physical_windows"]
                    else "insufficient_data"
                )
            if METRICS[metric][2] == 0 and result["eligible_windows"]:
                result["status"] = "descriptive_only"
            results.append(result)
    # Recompute smoothing from measured positions only; no new poses, filling or training.
    sensitivity_rows = []
    if sensitivity:
        for c in study["clips"]:
            folder = metrics_dir / c["clip_name"]
            frames = pd.read_parquet(folder / "frame_metrics.parquet")
            bouts, cycles = (
                pd.read_csv(folder / "running_bouts.csv"),
                pd.read_csv(folder / "gait_cycles.csv"),
            )
            for smoothing in (0.6, 0.8, 1.2):
                measured = (
                    frames
                    if smoothing == 0.8
                    else movement_metrics(frames, smoothing_seconds=smoothing)
                )
                for duration in (120, 180, 240):
                    alternative = attach_context(
                        window_metrics(measured, bouts, cycles, duration=duration, step=duration),
                        study,
                    )
                    for player, group in alternative.groupby("player_id"):
                        for metric in MOVEMENT:
                            base = baselines[
                                (baselines.player_id == player) & (baselines.metric == metric)
                            ].iloc[0]
                            group = group.copy()
                            group[f"{metric}_pct"] = (
                                (group[metric] / base.baseline - 1) * 100
                                if base.baseline_status == "available"
                                else np.nan
                            )
                            result = analyze_series(group, metric)
                            sensitivity_rows.append(
                                {
                                    "clip_name": c["clip_name"],
                                    "player_id": player,
                                    "metric": metric,
                                    "smoothing_seconds": smoothing,
                                    "window_seconds": duration,
                                    "eligible_windows": result["eligible_windows"],
                                    "search_available": result["searched_sequences"] > 0,
                                    "candidate_times": json.dumps(
                                        [r["boundary_time"] for r in result["candidates"]]
                                    ),
                                    "status": result["status"],
                                }
                            )
    sensitivity_table = pd.DataFrame(sensitivity_rows)
    for result in results:
        for candidate in result["candidates"]:
            candidates = (
                sensitivity_table[
                    (sensitivity_table.player_id == result["player_id"])
                    & (sensitivity_table.metric == result["metric"])
                    & (sensitivity_table.clip_name == candidate["clip_name"])
                ]
                if len(sensitivity_table)
                else pd.DataFrame()
            )
            assessed = candidates[candidates.search_available] if len(candidates) else candidates
            votes = sum(
                any(
                    abs(t - candidate["boundary_time"]) <= row.window_seconds
                    for t in json.loads(row.candidate_times)
                )
                for row in assessed.itertuples()
            )
            candidate["sensitivity_assessed_settings"] = len(assessed)
            candidate["sensitivity_supporting_settings"] = votes
            # Need at least two smoothing values AND two window widths with corroboration.
            corroborated = [
                row
                for row in assessed.itertuples()
                if any(
                    abs(t - candidate["boundary_time"]) <= row.window_seconds
                    for t in json.loads(row.candidate_times)
                )
            ]
            stable = (
                len({r.smoothing_seconds for r in corroborated}) >= 2
                and len({r.window_seconds for r in corroborated}) >= 2
                and votes >= 0.5 * len(assessed)
            )
            candidate["supported"] = stable
        if result["candidates"]:
            result["status"] = (
                "supported_candidate"
                if any(c["supported"] for c in result["candidates"])
                else "quality_confounded"
            )
        # Insufficient data with substantial observed losses is explicitly qualified.
        cov = coverage[coverage.player_id == result["player_id"]]
        if (
            result["metric"] in MOVEMENT
            and result["status"] == "insufficient_data"
            and (cov.movement_frame_coverage < 0.7).any()
        ):
            result["status"] = "quality_confounded"
    trends = [
        {"player_id": r["player_id"], "metric": r["metric"], **t}
        for r in results
        for t in r["trends"]
    ]
    if trends:
        trend_table = pd.DataFrame(trends)
        trend_table["holm_p_value"] = np.nan
        for _, group in trend_table.groupby("player_id"):
            valid = group[group.p_value.notna()]
            if len(valid):
                trend_table.loc[valid.index, "holm_p_value"] = multipletests(
                    valid.p_value, method="holm"
                )[1]
    else:
        trend_table = pd.DataFrame(columns=["player_id", "metric", "p_value", "holm_p_value"])
    pairs, tests = paired_comparison(windows, coverage)
    controls = synthetic_controls()
    if not all(c["passed"] for c in controls):
        raise RuntimeError("Synthetic software controls failed")
    windows.to_csv(output / "normalized_windows.csv", index=False)
    baselines.to_csv(output / "baselines.csv", index=False)
    trend_table.to_csv(output / "trends.csv", index=False)
    sensitivity_table.to_csv(output / "sensitivity.csv", index=False)
    pairs.to_csv(output / "early_late_comparison.csv", index=False)
    write_json(output / "paired_tests.json", {"results": tests})
    write_json(
        output / "synthetic_controls.json", {"controls": controls, "not_real_match_data": True}
    )
    substitution = study.get("verified_substitutions", {})
    context = []
    for r in results:
        verified = substitution.get(r["player_id"], {})
        for candidate in r["candidates"]:
            valid = (
                verified.get("verified") is True
                and bool(verified.get("source"))
                and isinstance(verified.get("match_seconds"), (int, float))
            )
            context.append(
                {
                    "player_id": r["player_id"],
                    "metric": r["metric"],
                    "boundary_time": candidate["boundary_time"],
                    "substitution_status": "verified_context_only" if valid else "unknown",
                    "occurs_before_substitution": candidate["boundary_time"]
                    < verified["match_seconds"]
                    if valid
                    else None,
                }
            )
    report = {
        **provenance(),
        "status": "completed",
        "study": str(study_path),
        "study_scope": {
            "distinct_minutes": sum(c["duration_seconds"] for c in study["clips"]) / 60,
            "players": int(windows.player_id.nunique()),
            "matches": len({c["match_id"] for c in study["clips"]}),
        },
        "metrics": results,
        "substitution_context": context,
        "substitution_metadata": "none verified"
        if not substitution
        else "provided; validated per candidate",
        "units": {m: list(v) for m, v in METRICS.items()},
        "sensitivity_settings": (
            "0.6/0.8/1.2-second smoothing x 120/180/240-second nonoverlap windows; "
            "2/3/4-log(n) penalties"
        ),
        "gait_sensitivity": (
            "unavailable: no eligible primary gait windows; gait sampling kept fixed"
        ),
        "inference": (
            "descriptive HAC lag-1 trends; Holm per player; "
            "stable material downward PELT changes only"
        ),
        "not_a_physiological_fatigue_diagnosis": True,
        "comparison": "selected early/late 25-minute segments; not complete half averages",
        "window_boundary": (
            "Incomplete final windows excluded; full-clip coverage gate still applies"
        ),
    }
    write_json(output / "analysis_report.json", report)
    deliver_visuals(windows, baselines, pairs, coverage, report, metrics_dir, output)
    return report
