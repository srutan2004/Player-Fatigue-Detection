"""A reviewable initial target-track selection for permitted local footage."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

from src.video_io import iter_video_frames
from src.vision import detect_people, load_model


def selection_frame(config: dict, output: Path) -> dict:
    """Save initial ByteTrack IDs for manual player selection, not jersey recognition.

    These IDs match a fresh extractor's first tracker update with identical settings.
    Later ambiguous reappearances must be reviewed rather than assigned by kit alone.
    """
    output.mkdir(parents=True, exist_ok=True)
    settings = config["pilot"]
    frame = next(iter_video_frames(config["video"]["path"], settings["start_seconds"], 0.2, 25))
    people = detect_people(
        load_model(config["models_dir"], settings["detector"]), frame.image, settings
    )
    rate = config["pipeline"].get("tracking_fps", 12.5)
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        frame_rate=rate,
    )
    detections = tracker.update_with_detections(
        sv.Detections(
            xyxy=np.asarray([p["bbox_xyxy"] for p in people]).reshape(-1, 4),
            confidence=np.asarray([p["confidence"] for p in people]),
            class_id=np.zeros(len(people), dtype=int),
        )
    )
    rows = []
    for track, box in zip(detections.tracker_id, detections.xyxy, strict=True):
        a, b, c, d = map(round, box)
        cv2.rectangle(frame.image, (a, b), (c, d), (0, 255, 255), 2)
        cv2.putText(
            frame.image, f"ID {track}", (a, b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        rows.append(
            {
                "initial_tracklet_id": int(track),
                "x1": a,
                "y1": b,
                "x2": c,
                "y2": d,
                "source_frame": frame.index,
            }
        )
    path = output / "initial_tracks.jpg"
    cv2.imwrite(str(path), frame.image)
    pd.DataFrame(rows).to_csv(output / "initial_tracks.csv", index=False)
    return {
        "image": str(path),
        "track_count": len(rows),
        "instruction": "Set reference_assisted_identity: false; set each initial_tracklet_id.",
    }
