"""Render chronological native-source crops for manual cycle/anatomy review."""

import argparse

import cv2
import numpy as np
import pandas as pd

from src.common import ROOT, load_config, write_json
from src.video_io import iter_video_frames
from src.vision import padded_crop


def make_bout_audit(clip_name: str, player: str) -> dict:
    """Select a measured bout by quality/exposure, never by an apparent decline.

    Sequential source crops permit anatomy and repeated-cycle inspection without
    interpolation or hiding poor frames behind a confidence score.
    """
    config = load_config(ROOT / f"configs/study_{clip_name}.yaml")
    metrics = ROOT / "results/metrics_csv" / clip_name
    bouts = pd.read_csv(metrics / "running_bouts.csv")
    selected = bouts[(bouts.player_id == player) & (bouts.accepted_cycles > 0)].sort_values(
        "accepted_cycles", ascending=False
    )
    if selected.empty:
        return {"status": "no_accepted_bout", "clip": clip_name, "player": player}
    bout = selected.iloc[0]
    offset = config["video"]["match_time_offset_seconds"]
    start = bout.start_time - offset
    duration = min(6.0, bout.duration_seconds)
    pose_dir = ROOT / config["pipeline"].get("pose_output_dir", f"results/phase3/study/{clip_name}")
    poses = pd.read_csv(pose_dir / "keypoints_positions.csv")
    poses = poses[poses.player_id == player].set_index("source_frame_index")
    output = ROOT / "results/phase3/manual_bout_audit"
    output.mkdir(parents=True, exist_ok=True)
    canvas = np.full((6 * 220, 8 * 160, 3), 240, dtype=np.uint8)
    position = 0
    source_rows = []
    first_index = None
    for frame in iter_video_frames(config["video"]["path"], start, duration, 25):
        if first_index is None:
            first_index = frame.index
        if (frame.index - first_index) % 4 or position >= 48:
            continue
        row = poses.loc[frame.index]
        if not np.isfinite(
            row[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(dtype=float)
        ).all():
            continue
        crop, bounds = padded_crop(
            frame.image, [row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2], 0.1
        )
        for leg, color in [("left", (0, 0, 255)), ("right", (0, 255, 255))]:
            points = []
            for joint in ("hip", "knee", "ankle"):
                x, y = row[f"{leg}_{joint}_x"], row[f"{leg}_{joint}_y"]
                if np.isfinite(x) and np.isfinite(y):
                    points.append((round(x - bounds[0]), round(y - bounds[1])))
            if len(points) == 3:
                cv2.polylines(crop, [np.array(points)], False, color, 1)
        y0, x0 = (position // 8) * 220, (position % 8) * 160
        canvas[y0 : y0 + 190, x0 : x0 + 160] = cv2.resize(crop, (160, 190))
        cv2.putText(
            canvas,
            f"{frame.seconds:.2f}s",
            (x0 + 4, y0 + 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        source_rows.append(
            {
                "source_frame": frame.index,
                "video_seconds": frame.seconds,
                "body_height": float(row.box_height),
            }
        )
        position += 1
    path = output / f"{clip_name}_{player}.jpg"
    cv2.imwrite(str(path), canvas)
    report = {
        "status": "awaiting_manual_review",
        "clip": clip_name,
        "player": player,
        "bout_id": bout.bout_id,
        "start_video_seconds": float(start),
        "duration_seconds": float(duration),
        "sheet": str(path),
        "sampled_source_rows": source_rows,
        "manual_identity_consistent": None,
        "manual_anatomy_quality": None,
        "manual_same_leg_cycles": None,
        "note": "Audit images sampled every four native frames; metric extraction remains 25fps.",
    }
    write_json(path.with_suffix(".json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--player", required=True)
    args = parser.parse_args()
    print(make_bout_audit(args.clip, args.player))
