"""Source-coordinate pose tables with explicit confidence/missingness semantics."""

from pathlib import Path

import pandas as pd

JOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def flatten_pose(pose: dict | None) -> dict:
    """Export every joint consistently; low-confidence coordinates stay empty.

    A noisy knee can dominate apparent flexion range, so visibility gates must be
    retained alongside coordinates rather than replacing missing joints with zero.
    """
    result = {f"{joint}_{axis}": None for joint in JOINTS for axis in ("x", "y", "confidence")}
    for joint in (pose or {}).get("joints", []):
        result[f"{joint['name']}_x"] = joint["x"]
        result[f"{joint['name']}_y"] = joint["y"]
        result[f"{joint['name']}_confidence"] = joint["confidence"]
    return result


def save_pose_gaps(frames: pd.DataFrame, output: Path) -> None:
    """Compress failed requested anatomy into explicit intervals, preserving exclusions.

    Not-requested walking frames are selection, not sensor dropouts. Low joint
    confidence inside measured running bouts must remain visible in methodology.
    """
    rows = []
    for player, part in frames.groupby("player_id"):
        part = part.sort_values("timestamp")
        status = part.pose_status.copy()
        status.loc[status.eq("pose_returned") & ~part.gait_pose_valid] = "low_joint_confidence"
        failed = ~status.isin(["pose_returned", "not_requested_outside_running_bout"])
        boundary = status.ne(status.shift()) | (part.timestamp.diff() > 0.075)
        for _, gap in part[failed].groupby(boundary.cumsum()[failed]):
            dt = float(part.timestamp.diff().median())
            rows.append(
                {
                    "player_id": player,
                    "reason": status.loc[gap.index[0]],
                    "start_frame": int(gap.source_frame_index.iloc[0]),
                    "end_frame_exclusive": int(gap.source_frame_index.iloc[-1]) + 1,
                    "start_time": float(gap.timestamp.iloc[0]),
                    "end_time_exclusive": float(gap.timestamp.iloc[-1] + dt),
                }
            )
    pd.DataFrame(
        rows,
        columns=[
            "player_id",
            "reason",
            "start_frame",
            "end_frame_exclusive",
            "start_time",
            "end_time_exclusive",
        ],
    ).to_csv(output, index=False)
