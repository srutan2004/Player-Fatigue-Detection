"""Movement and projected-gait measurements with explicit temporal/quality boundaries."""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, lombscargle, savgol_filter


def knee_flexion(hip, knee, ankle) -> np.ndarray:
    """Return 180 minus the internal 2D knee angle, degrees, with invalid joints missing.

    Reduced flexion excursion can accompany altered mechanics, but projection,
    running speed and viewpoint confound its interpretation as fatigue.
    """
    a, b = np.asarray(hip) - np.asarray(knee), np.asarray(ankle) - np.asarray(knee)
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.sum(a * b, axis=-1) / denominator
        angles = 180 - np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return np.where(denominator > 1e-8, angles, np.nan)


def movement_metrics(
    frames: pd.DataFrame, smoothing_seconds: float = 0.8, maximum_gap_seconds: float = 0.15
) -> pd.DataFrame:
    """Differentiate smoothed ground positions only inside valid uninterrupted segments.

    A track switch, calibration loss, gap, or clip boundary must never become a
    displacement interval. Implausible speeds are flagged as invalid, not clipped.
    """
    data = frames.sort_values(["clip_name", "player_id", "timestamp"]).copy()
    for column in ("speed", "distance_increment", "valid_dt", "heading"):
        data[column] = np.nan
    data["movement_valid"] = False
    data["movement_rejection"] = "missing_position_or_gap"
    for _, player in data.groupby(["clip_name", "player_id"], sort=False):
        dt = player["timestamp"].diff()
        valid = np.isfinite(player[["world_x", "world_y"]]).all(axis=1)
        if "position_valid" in player:
            valid &= player["position_valid"].astype(bool)
        boundaries = (
            ~valid | ~valid.shift(fill_value=False) | (dt > maximum_gap_seconds) | (dt <= 0)
        )
        if "identity_segment" in player:
            boundaries |= player["identity_segment"].ne(player["identity_segment"].shift())
        segments = boundaries.cumsum()
        for _, part in player[valid].groupby(segments[valid]):
            if len(part) < 5:
                continue
            t = part["timestamp"].to_numpy(float)
            median_dt = float(np.median(np.diff(t)))
            if median_dt <= 0:
                continue
            window = max(3, round(smoothing_seconds / median_dt) | 1)
            window = min(window, len(part) if len(part) % 2 else len(part) - 1)
            xy = part[["world_x", "world_y"]].rolling(3, center=True, min_periods=1).median()
            smoothed = savgol_filter(xy.to_numpy(), window, min(2, window - 1), axis=0)
            velocity = np.gradient(smoothed, t, axis=0)
            speed = np.linalg.norm(velocity, axis=1)
            distance = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
            # An interval is valid only if both bounding samples have plausible speed.
            accepted = np.isfinite(speed) & (speed <= 12.0)
            interval_valid = accepted[1:] & accepted[:-1]
            idx = part.index
            data.loc[idx, "speed"] = np.where(accepted, speed, np.nan)
            data.loc[idx, "heading"] = np.arctan2(velocity[:, 1], velocity[:, 0])
            data.loc[idx, "movement_valid"] = accepted
            data.loc[idx, "movement_rejection"] = np.where(accepted, "", "implausible_speed")
            data.loc[idx[1:], "distance_increment"] = np.where(interval_valid, distance, np.nan)
            data.loc[idx[1:], "valid_dt"] = np.where(interval_valid, np.diff(t), np.nan)
    return data


def _xy(part: pd.DataFrame, joint: str) -> np.ndarray:
    """Read confidence-gated image joints; missing anatomy stays NaN."""
    return part[[f"{joint}_x", f"{joint}_y"]].to_numpy(float)


def straight_running_mask(data: pd.DataFrame) -> pd.Series:
    """Require at most 30 degrees of heading spread over two observed seconds.

    Gradual turns can pass a per-frame angle threshold yet distort projected gait.
    Unwrap directions within uninterrupted motion before checking local spread.
    """
    result = pd.Series(False, index=data.index)
    valid = data.movement_valid & data.speed.between(2.0, 9.0)
    boundary = (
        ~valid
        | ~valid.shift(fill_value=False)
        | (data.timestamp.diff() > 0.075)
        | (data.timestamp.diff() <= 0)
    )
    if "identity_segment" in data:
        boundary |= data.identity_segment.ne(data.identity_segment.shift())
    for _, part in data[valid].groupby(boundary.cumsum()[valid]):
        angles = pd.Series(
            np.unwrap(part.heading.to_numpy()), index=pd.to_timedelta(part.timestamp, unit="s")
        )
        local = angles.rolling("2s", center=True, min_periods=2)
        result.loc[part.index] = ((local.max() - local.min()) <= np.deg2rad(30)).to_numpy()
    return result


def cadence_events(signal: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, dict]:
    """Accept observed same-leg peaks only when periodicity supports their period.

    This counts strides, not alternating steps. Directional turns, low sample rates
    and weak projected oscillations are explicit reasons to reject a bout.
    Small confidence holes use Lomb–Scargle on observed samples; joint coordinates
    are never filled. At least 80% observations and no hole >0.16s are required.
    """
    finite = np.isfinite(signal)
    if len(signal) < 20 or finite.mean() < 0.8:
        return np.array([], dtype=int), {"status": "missing_signal"}
    dt = float(np.median(np.diff(timestamps)))
    fps = 1 / dt
    if fps < 20 or np.max(np.abs(np.diff(timestamps) - dt)) > dt * 0.2:
        return np.array([], dtype=int), {"status": "insufficient_or_irregular_sample_rate"}
    observed_indices = np.flatnonzero(finite)
    observed_t = timestamps[finite]
    if len(observed_t) < 20 or np.diff(observed_t).max() > 0.16 + 1e-7:
        return observed_indices[:0], {"status": "pose_gap_too_long"}
    centered = signal[finite] - np.polyval(np.polyfit(observed_t, signal[finite], 1), observed_t)
    if np.std(centered) < 0.01:
        return np.array([], dtype=int), {"status": "insufficient_projected_amplitude"}
    if finite.all():
        centered = savgol_filter(centered, 5, 2)
    peaks, _ = find_peaks(
        centered, distance=max(2, round(fps / 3.5)), prominence=max(0.01, np.std(centered) * 0.6)
    )
    if len(peaks) < 4:
        return peaks[:0], {"status": "too_few_cycles"}
    peaks = observed_indices[peaks]
    periods = np.diff(timestamps[peaks])
    period = float(np.median(periods))
    if not 0.28 <= period <= 2 or np.std(periods) / period > 0.25:
        return peaks[:0], {"status": "irregular_cycles"}
    if finite.all():
        correlation = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
        correlation /= correlation[0]
        lag = round(period * fps)
        support = float(np.max(correlation[max(1, lag - 2) : lag + 3]))
        method = "autocorrelation"
    else:
        frequencies = np.linspace(0.5, 3.5, 400)
        spectrum = lombscargle(
            observed_t - observed_t[0], centered, 2 * np.pi * frequencies, normalize=True
        )
        best = int(np.argmax(spectrum))
        support = float(spectrum[best])
        if abs(frequencies[best] - 1 / period) / frequencies[best] > 0.15:
            return peaks[:0], {"status": "observed_peaks_disagree_with_periodicity"}
        method = "lomb_scargle_observed_samples"
    if support < 0.3:
        return peaks[:0], {"status": "weak_periodicity", "support": support}
    return peaks, {
        "status": "accepted",
        "strides_per_second": 1 / period,
        "autocorrelation_support": support if method == "autocorrelation" else None,
        "periodicity_support": support,
        "support_method": method,
        "signal_observed_fraction": float(finite.mean()),
        "period_cv": float(np.std(periods) / period),
    }


def gait_metrics(frames: pd.DataFrame, minimum_bout_seconds: float = 4) -> tuple:
    """Measure cycles only in continuous, straight, confidence-qualified running bouts.

    Outputs are 2D projection proxies. Compare similar speed/view conditions before
    interpreting any change as altered mechanics or fatigue-related decline.
    """
    bouts, cycles = [], []
    for (clip, player_id), data in frames.groupby(["clip_name", "player_id"], sort=False):
        data = data.sort_values("timestamp")
        required = [
            f"{joint}_{axis}"
            for joint in (
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            )
            for axis in ("x", "y")
        ]
        if not np.isfinite(data[required]).to_numpy().any():
            continue
        dt = data["timestamp"].diff()
        eligible = straight_running_mask(data) & (data["box_height"] >= 40)
        boundaries = ~eligible | ~eligible.shift(fill_value=False) | (dt > 0.075) | (dt <= 0)
        if "identity_segment" in data:
            boundaries |= data["identity_segment"].ne(data["identity_segment"].shift())
        for _, part in data[eligible].groupby(boundaries.cumsum()[eligible]):
            t = part["timestamp"].to_numpy(float)
            if len(t) < 2 or t[-1] - t[0] < minimum_bout_seconds:
                continue
            hip = (_xy(part, "left_hip") + _xy(part, "right_hip")) / 2
            visible_hip = hip[np.isfinite(hip).all(axis=1)]
            image_direction = (
                np.degrees(np.arctan2(*(visible_hip[-1] - visible_hip[0])[::-1]))
                if len(visible_hip) >= 2
                else np.nan
            )
            height = np.maximum(1.0, part["box_height"].to_numpy(float))
            bout_id = f"{clip}_{player_id}_{int(t[0] * 1000)}"
            bout = {
                "clip_name": clip,
                "player_id": player_id,
                "bout_id": bout_id,
                "start_time": float(t[0]),
                "end_time": float(t[-1]),
                "duration_seconds": float(t[-1] - t[0]),
                "avg_speed": float(part["speed"].mean()),
                "accepted_cycles": 0,
            }
            accepted_legs = []
            for leg in ("left", "right"):
                ankle = _xy(part, f"{leg}_ankle")
                signal = (ankle[:, 0] - _xy(part, f"{leg}_hip")[:, 0]) / height
                peaks, assessment = cadence_events(signal, t)
                bout[f"{leg}_status"] = assessment["status"]
                if assessment["status"] != "accepted":
                    continue
                accepted_legs.append((leg, peaks, assessment))
            if len(accepted_legs) == 2:
                frequencies = [item[2]["strides_per_second"] for item in accepted_legs]
                if abs(frequencies[0] - frequencies[1]) / np.mean(frequencies) > 0.25:
                    bout["left_status"] = bout["right_status"] = "left_right_period_disagreement"
                    accepted_legs = []
            for leg, peaks, assessment in accepted_legs:
                knee = knee_flexion(
                    _xy(part, f"{leg}_hip"), _xy(part, f"{leg}_knee"), _xy(part, f"{leg}_ankle")
                )
                anchor_y = part["ground_pixel_y"].to_numpy(float)
                vertical = (hip[:, 1] - anchor_y) / height
                present = np.isfinite(vertical)
                if present.sum() >= 3:
                    vertical -= np.polyval(np.polyfit(t[present], vertical[present], 1), t)
                for first, last in zip(peaks[:-1], peaks[1:], strict=True):
                    cycle = part.iloc[first : last + 1]
                    duration = t[last] - t[first]
                    if cycle["valid_dt"].isna().any():
                        continue
                    if (
                        abs(1 / duration - assessment["strides_per_second"])
                        / (assessment["strides_per_second"])
                        > 0.25
                    ):
                        continue
                    knee_cycle, vertical_cycle = knee[first : last + 1], vertical[first : last + 1]
                    cycles.append(
                        {
                            "clip_name": clip,
                            "player_id": player_id,
                            "bout_id": bout_id,
                            "leg": leg,
                            "start_time": float(t[first]),
                            "end_time": float(t[last]),
                            "stride_frequency": float(1 / duration),
                            "stride_length": float(cycle["distance_increment"].iloc[1:].sum()),
                            "vertical_oscillation": float(np.nanstd(vertical_cycle))
                            if np.isfinite(vertical_cycle).mean() >= 0.8
                            else np.nan,
                            "knee_angle_range": float(
                                np.nanpercentile(knee_cycle, 95) - np.nanpercentile(knee_cycle, 5)
                            )
                            if np.isfinite(knee_cycle).mean() >= 0.8
                            else np.nan,
                            "avg_speed": float(cycle["speed"].mean()),
                            "body_scale_source_pixels": float(np.median(height[first : last + 1])),
                            "image_motion_angle_deg": float(image_direction),
                            "autocorrelation_support": assessment["autocorrelation_support"],
                            "periodicity_support": assessment["periodicity_support"],
                            "support_method": assessment["support_method"],
                            "signal_observed_fraction": assessment["signal_observed_fraction"],
                        }
                    )
                    bout["accepted_cycles"] += 1
            bouts.append(bout)
    bout_columns = [
        "clip_name",
        "player_id",
        "bout_id",
        "start_time",
        "end_time",
        "duration_seconds",
        "avg_speed",
        "accepted_cycles",
        "left_status",
        "right_status",
    ]
    cycle_columns = [
        "clip_name",
        "player_id",
        "bout_id",
        "leg",
        "start_time",
        "end_time",
        "stride_frequency",
        "stride_length",
        "vertical_oscillation",
        "knee_angle_range",
        "avg_speed",
        "body_scale_source_pixels",
        "image_motion_angle_deg",
        "autocorrelation_support",
        "periodicity_support",
        "support_method",
        "signal_observed_fraction",
    ]
    return pd.DataFrame(bouts, columns=bout_columns), pd.DataFrame(cycles, columns=cycle_columns)


def window_metrics(
    frames: pd.DataFrame,
    bouts: pd.DataFrame,
    cycles: pd.DataFrame,
    duration: float = 120,
    step: float = 60,
) -> pd.DataFrame:
    """Aggregate measured exposure; insufficient coverage produces missing metrics.

    Average work rate is observed distance / valid time. Stride length is coupled
    to speed/cadence, and vertical oscillation remains a dimensionless image proxy.
    """
    rows = []
    for (clip, player_id), data in frames.groupby(["clip_name", "player_id"], sort=False):
        minimum, maximum = float(data.timestamp.min()), float(data.timestamp.max())
        for start in np.arange(minimum, maximum, step):
            if start + duration > maximum + 0.1:
                break
            part = data[data.timestamp.between(start, start + duration, inclusive="left")]
            clipped_dt = np.minimum(part.valid_dt, (part.timestamp - start).clip(lower=0))
            clipped_distance = part.distance_increment * (clipped_dt / part.valid_dt)
            valid_seconds = float(clipped_dt.sum())
            coverage = min(1.0, valid_seconds / duration)
            movement_valid = coverage >= 0.7
            speed = part["speed"].dropna()
            sample_dt = float(part.timestamp.diff().median())
            required_samples = max(2, round(0.9 / sample_dt))
            sustained = (
                part.set_index(pd.to_timedelta(part.timestamp, unit="s"))["speed"]
                .rolling("1s", min_periods=required_samples)
                .mean()
            )
            chosen = (
                cycles[
                    (cycles.clip_name == clip)
                    & (cycles.player_id == player_id)
                    & (cycles.start_time >= start)
                    & (cycles.end_time < start + duration)
                ]
                if not cycles.empty
                else cycles
            )
            # Union the measured cycles so left/right legs do not double exposure.
            running_seconds, interval_end = 0.0, None
            for interval in chosen.sort_values("start_time").itertuples():
                left = (
                    interval.start_time
                    if interval_end is None
                    else max(interval.start_time, interval_end)
                )
                running_seconds += max(0.0, interval.end_time - left)
                interval_end = max(interval_end or interval.end_time, interval.end_time)
            same_leg_cycles = int(chosen.groupby("leg").size().max()) if len(chosen) else 0
            gait_valid = running_seconds >= 20 and same_leg_cycles >= 10
            row = {
                "clip_name": clip,
                "player_id": player_id,
                "window_start_time": float(start),
                "window_end_time": float(start + duration),
                "valid_seconds": valid_seconds,
                "position_coverage": coverage,
                "running_seconds": running_seconds,
                "valid_cycles": len(chosen),
                "movement_eligible": movement_valid,
                "gait_eligible": gait_valid,
                "avg_speed": float(clipped_distance.sum() / valid_seconds)
                if movement_valid and valid_seconds
                else np.nan,
                "peak_speed": float(sustained.max()) if movement_valid else np.nan,
                "p90_running_speed": float(speed[speed >= 2].quantile(0.9))
                if movement_valid and (speed >= 2).any()
                else np.nan,
                "high_intensity_distance_per_minute": float(
                    clipped_distance[part.speed >= 5.5].sum() / valid_seconds * 60
                )
                if movement_valid and valid_seconds
                else np.nan,
                "vertical_oscillation_unit": "normalized_image_proxy",
                "gait_rejection": "" if gait_valid else "insufficient_accepted_running_or_cycles",
            }
            for metric in (
                "stride_frequency",
                "stride_length",
                "vertical_oscillation",
                "knee_angle_range",
            ):
                eligible = gait_valid and chosen[metric].notna().sum() >= 10
                row[f"{metric}_eligible"] = eligible
                row[metric] = float(chosen[metric].median()) if eligible else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
