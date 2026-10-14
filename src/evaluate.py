"""Held-out autonomous tracking checks against curated anatomical MOT boxes."""

import sqlite3

import numpy as np
import pandas as pd
import supervision as sv
from scipy.optimize import linear_sum_assignment

from src.common import ROOT, provenance, write_json
from src.reference import reference_people
from src.video_io import iter_video_frames
from src.vision import box_iou, detect_people, load_model


def evaluate_mot(seconds: float = 60) -> dict:
    """Measure known-player recall and observed ID switches with an untouched tracker.

    Referees/staff are incomplete in these source annotations, so unmatched detector
    boxes are not automatically false positives. Switch counts are conditional on
    consecutive annotated matches and are not a complete IDF1/HOTA evaluation.
    """
    if not 0 < seconds <= 240:
        raise ValueError("Evaluation is bounded to the curated four-minute MOT clip")
    output = ROOT / "results/phase2/autonomous_tracking"
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(ROOT / "models", "yolov8n.pt")
    settings = {"detection_confidence": 0.15, "imgsz": 1920, "device": "cuda:0"}
    tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, frame_rate=12.5)
    rows, prior, switches, observations = [], {}, 0, 0
    with sqlite3.connect(ROOT / "data/processed/128057_mot.sqlite") as reference:
        for sampled, frame in enumerate(
            iter_video_frames(ROOT / "data/hub/mot/clips/128057.mp4", 0, seconds, 12.5)
        ):
            people = detect_people(model, frame.image, settings)
            found = tracker.update_with_detections(
                sv.Detections(
                    xyxy=np.asarray([p["bbox_xyxy"] for p in people]).reshape(-1, 4),
                    confidence=np.asarray([p["confidence"] for p in people]),
                    class_id=np.zeros(len(people), dtype=int),
                )
            )
            gt = reference_people(reference, frame.index)
            boxes = np.asarray([[g[k] for k in ("x1", "y1", "x2", "y2")] for g in gt])
            if len(found) and len(gt):
                overlaps = np.array([[box_iou(a, b) for b in found.xyxy] for a in boxes])
                left, right = linear_sum_assignment(1 - overlaps)
                associations = {
                    i: j for i, j in zip(left, right, strict=True) if overlaps[i, j] >= 0.5
                }
            else:
                associations = {}
            for i, g in enumerate(gt):
                matched = i in associations
                track = int(found.tracker_id[associations[i]]) if matched else None
                last = prior.get(g["player"])
                changed = bool(matched and last and last[0] == sampled - 1 and last[1] != track)
                switches += changed
                observations += bool(matched and last and last[0] == sampled - 1)
                if matched:
                    prior[g["player"]] = (sampled, track)
                height = g["y2"] - g["y1"]
                rows.append(
                    {
                        "source_frame": frame.index,
                        "player": g["player"],
                        "height": height,
                        "size_band": "small_lt32"
                        if height < 32
                        else "medium_32_64"
                        if height < 64
                        else "large_ge64",
                        "matched_iou50": matched,
                        "tracklet_id": track,
                        "observed_switch": changed,
                    }
                )
    data = pd.DataFrame(rows)
    data.to_csv(output / "matches.csv", index=False)
    size = data.groupby("size_band").agg(
        annotated_boxes=("player", "size"), recall=("matched_iou50", "mean")
    )
    size.to_csv(output / "recall_by_size.csv")
    report = {
        **provenance(),
        "match": "128057",
        "seconds": seconds,
        "iou_threshold": 0.5,
        "annotated_player_boxes": len(data),
        "known_player_box_recall": float(data.matched_iou50.mean()),
        "consecutive_observed_identity_switches": int(switches),
        "consecutive_matched_pairs": observations,
        "raw_tracker_max_time_lost_frames": tracker.max_time_lost,
        "reference_assisted_identity": False,
        "complete_identity_accuracy_metric": False,
        "caveat": (
            "Unannotated staff/referees prevent an honest all-person precision claim; "
            "MOT label alignment was visually sampled."
        ),
    }
    write_json(output / "report.json", report)
    return report
