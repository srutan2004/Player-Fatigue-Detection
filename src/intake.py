"""Inspect selected video and save a reproducible source manifest and audit frames."""

import csv
from pathlib import Path

import cv2

from src.annotations import audit_annotations
from src.calibration_audit import audit_calibration
from src.common import provenance, sha256_file, write_json
from src.video_io import frame_time_fields, iter_video_frames, probe_video


def inspect_video(config: dict, hash_source: bool = True) -> dict:
    """Create a small visual/timestamp audit before expensive player extraction.

    Unknown camera geometry and unverified identities remain open feasibility
    questions; metadata inspection alone must not imply validated field positions.
    """
    output, video, pilot = config["output_dir"], config["video"], config["pilot"]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        **provenance(),
        "dataset": config["dataset"],
        "config": config,
        "status": "awaiting_video",
        "open_checks": [
            "Visually verify annotation/frame alignment and cross-half identities.",
            "Audit target crops for source-pixel detail; confidence is not accuracy.",
            "Validate the supplied camera model or a local mapping on held-out lines.",
        ],
    }
    if not video["path"].is_file():
        report["action"] = (
            f"Place a permitted continuous source video at {video['path']} or edit config."
        )
        write_json(output / "source_manifest.json", report)
        return report
    report["video"] = probe_video(video["path"])
    duration = report["video"]["duration_seconds_metadata"]
    if duration and pilot["start_seconds"] + pilot["duration_seconds"] > duration + 0.1:
        raise ValueError("Requested pilot extends beyond the source duration; shorten the config")
    report["video"]["sha256"] = sha256_file(video["path"]) if hash_source else None
    image_dir = output / "intake_frames"
    image_dir.mkdir(exist_ok=True)
    rows = []
    settings = (
        pilot["start_seconds"],
        pilot["duration_seconds"],
        min(pilot["sample_fps"], 10 / pilot["duration_seconds"]),
    )
    for frame in iter_video_frames(video["path"], *settings):
        filename = f"frame_{frame.index:08d}.jpg"
        if not cv2.imwrite(str(image_dir / filename), frame.image):
            raise OSError("Could not write intake audit image")
        rows.append(
            {
                **frame_time_fields(
                    frame, pilot["start_seconds"], video["half"], video["match_time_offset_seconds"]
                ),
                "image": filename,
            }
        )
    if not rows:
        raise ValueError("Pilot interval contains no decoded frames; check start/duration")
    with (output / "intake_frame_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report["sampled_frames"] = len(rows)
    report["observed_first_last_seconds"] = [rows[0]["video_seconds"], rows[-1]["video_seconds"]]
    report["calibration"] = {
        "path": video.get("calibration"),
        "exists": bool(video.get("calibration") and Path(video["calibration"]).exists()),
        "validation_status": "not_yet_validated",
    }
    if report["calibration"]["exists"]:
        report["calibration"]["asset_audit"] = audit_calibration(Path(video["calibration"]))
    if video.get("annotations"):
        report["annotations"] = audit_annotations(video["annotations"])
    report["status"] = "intake_complete_measurements_unverified"
    write_json(output / "source_manifest.json", report)
    return report
