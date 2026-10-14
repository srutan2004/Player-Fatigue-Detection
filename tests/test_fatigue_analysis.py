"""Controls for decline inference, missingness, normalization and subject-level validation."""

import numpy as np
import pandas as pd

from src.fatigue_analysis import (
    METRICS,
    analyze_series,
    baseline_normalize,
    continuous_chunks,
    paired_comparison,
    synthetic_controls,
)


def test_known_change_and_null_controls():
    assert all(r["passed"] for r in synthetic_controls())


def test_missing_windows_and_halves_never_bridge():
    data = pd.DataFrame(
        {
            "clip_name": ["early"] * 5 + ["late"] * 5,
            "window_start_time": list(np.arange(5) * 120) * 2,
            "window_end_time": list((np.arange(5) + 1) * 120) * 2,
            "avg_speed_pct": [0, 0, np.nan, 0, 0] + [-30] * 5,
        }
    )
    chunks = continuous_chunks(data, "avg_speed_pct")
    assert sorted(map(len, chunks)) == [2, 2, 5]
    result = analyze_series(data, "avg_speed")
    assert not result["candidates"] and not result["trends"]


def test_elapsed_baseline_zero_and_missing_exposure():
    data = pd.DataFrame(
        {
            "player_id": ["p"] * 6,
            "half": [1] * 5 + [2],
            "clip_start_time": [300] * 5 + [3600],
            "window_start_time": [300, 420, 540, 660, 780, 3600],
            "window_end_time": [420, 540, 660, 780, 900, 3720],
            "valid_seconds": [100] * 6,
            "running_seconds": [0] * 6,
        }
    )
    for metric in METRICS:
        data[metric] = 0 if metric == "peak_speed" else 2
    data.loc[5, "avg_speed"] = 1
    normalized, baseline = baseline_normalize(data)
    assert normalized.avg_speed_pct.iloc[5] == -50
    assert normalized.peak_speed_pct.isna().all()
    assert baseline[baseline.metric == "peak_speed"].baseline_status.iloc[0] == "near_zero_baseline"
    assert normalized.stride_frequency_pct.isna().all()


def test_paired_unit_is_player_and_low_coverage_period_excluded():
    data = pd.DataFrame(
        {
            "player_id": ["p"] * 10,
            "half": [1] * 5 + [2] * 5,
            "clip_duration": [1500] * 10,
            "valid_seconds": [100] * 10,
            "running_seconds": [0] * 10,
        }
    )
    for metric in METRICS:
        data[metric] = 2
    coverage = pd.DataFrame(
        {"player_id": ["p", "p"], "half": [1, 2], "valid_movement_seconds": [1200, 600]}
    )
    pairs, statistics = paired_comparison(data, coverage)
    assert not pairs.paired_eligible.any()
    coverage.loc[1, "valid_movement_seconds"] = 1200
    pairs, statistics = paired_comparison(data, coverage)
    assert pairs[pairs.metric == "avg_speed"].paired_eligible.all()
    assert all(s["p_value"] is None for s in statistics)
    assert statistics[0]["paired_players"] == 1
