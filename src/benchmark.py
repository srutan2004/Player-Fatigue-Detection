"""Bounded detector/pose feasibility pilot, with image audits and timing evidence."""

import csv
import json
import time

import cv2
import numpy as np
import psutil
import torch

from src.common import provenance, sha256_file, write_json
from src.video_io import frame_time_fields, iter_video_frames, probe_video
from src.vision import detect_people, load_model, padded_crop, pose_crop


def _sync(device):
    """Exclude asynchronous CUDA execution from falsely optimistic timings."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _csv(path, rows, columns):
    """Write headers even for empty outputs so no detections is an explicit result."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_pilot(config: dict) -> dict:
    """Measure pretrained feasibility, not persistent tracking or fatigue.

    A sparse image audit deliberately spans small/medium/large detections. Human
    review is required for detection misses and anatomical accuracy. These samples
    must not be used to estimate gait cadence or player-level longitudinal trends.
    """
    output, video, settings = config["output_dir"], config["video"], config["pilot"]
    metadata = probe_video(video["path"])
    duration = metadata["duration_seconds_metadata"]
    if duration and settings["start_seconds"] + settings["duration_seconds"] > duration + 0.1:
        raise ValueError("Requested pilot extends beyond the source duration; shorten the config")
    if settings["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; run the environment check first")
    output.mkdir(parents=True, exist_ok=True)
    crop_dir, frame_dir = output / "audit_crops", output / "detection_frames"
    crop_dir.mkdir(exist_ok=True)
    frame_dir.mkdir(exist_ok=True)
    write_json(
        output / "pilot_status.json", {**provenance(), "status": "running", "config": config}
    )
    detector = load_model(config["models_dir"], settings["detector"])
    pose_model = load_model(config["models_dir"], settings["pose_model"])
    # Warm up outside measured stage totals.
    warm = np.zeros((640, 640, 3), dtype=np.uint8)
    detector.predict(warm, imgsz=settings["imgsz"], device=settings["device"], verbose=False)
    pose_model.predict(warm, imgsz=640, device=settings["device"], verbose=False)
    _sync(settings["device"])
    if settings["device"].startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(settings["device"])
    timing, detections, poses, audits = [], [], [], []
    process = psutil.Process()
    max_ram = process.memory_info().rss
    audit_interval = settings["duration_seconds"] / settings["max_audit_crops"]
    next_audit = settings["start_seconds"]
    started = time.perf_counter()
    writer = None
    try:
        for frame in iter_video_frames(
            video["path"],
            settings["start_seconds"],
            settings["duration_seconds"],
            settings["sample_fps"],
        ):
            fields = frame_time_fields(
                frame, settings["start_seconds"], video["half"], video["match_time_offset_seconds"]
            )
            _sync(settings["device"])
            t0 = time.perf_counter()
            people = detect_people(detector, frame.image, settings)
            _sync(settings["device"])
            detection_time = time.perf_counter() - t0
            canvas = frame.image.copy()
            for person in people:
                box = person["bbox_xyxy"]
                detections.append({**fields, **person, "bbox_xyxy": json.dumps(box)})
                x1, y1, x2, y2 = map(round, box)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(
                    canvas,
                    f"d{person['detection_index']} {person['confidence']:.2f}",
                    (x1, max(14, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )
            pose_time = 0.0
            if frame.seconds + 1e-8 >= next_audit and len(audits) < settings["max_audit_crops"]:
                # Three size strata; selection is not a target identity claim.
                if people:
                    ordered = sorted(people, key=lambda p: p["bbox_xyxy"][3] - p["bbox_xyxy"][1])
                    stratum = len(audits) % 3
                    selected = ordered[round((len(ordered) - 1) * stratum / 2)]
                    box = selected["bbox_xyxy"]
                    crop, bounds = padded_crop(frame.image, box, settings["crop_padding"])
                    _sync(settings["device"])
                    t0 = time.perf_counter()
                    pose = pose_crop(pose_model, crop, bounds, box, settings)
                    _sync(settings["device"])
                    pose_time = time.perf_counter() - t0
                    audit_id = f"f{frame.index:08d}_d{selected['detection_index']}"
                    for joint in pose["joints"]:
                        poses.append({**fields, "audit_id": audit_id, **joint})
                        if joint["valid"]:
                            point = (round(joint["x"] - bounds[0]), round(joint["y"] - bounds[1]))
                            cv2.circle(crop, point, 2, (0, 255, 255), -1)
                            cv2.circle(
                                canvas, (round(joint["x"]), round(joint["y"])), 3, (0, 255, 255), -1
                            )
                    # Save original pixels for an unbiased manual audit as well as joint overlays.
                    original = frame.image[bounds[1] : bounds[3], bounds[0] : bounds[2]]
                    for name, image in (
                        (f"{audit_id}.jpg", original),
                        (f"{audit_id}_pose.jpg", crop),
                    ):
                        if not cv2.imwrite(str(crop_dir / name), image):
                            raise OSError("Could not write crop audit")
                    audit = {
                        **fields,
                        "audit_id": audit_id,
                        "detection_index": selected["detection_index"],
                        "player_id": "unknown",
                        "size_stratum": ["small", "medium", "large"][stratum],
                        "box_height_source_pixels": box[3] - box[1],
                        "box_width_source_pixels": box[2] - box[0],
                        "crop_bounds_xyxy": json.dumps(bounds),
                        "pose_status": pose["status"],
                        "gait_joints_confident": pose["gait_joints_confident"],
                        "manual_is_player": "",
                        "manual_identity": "",
                        "manual_hips_usable": "",
                        "manual_knees_usable": "",
                        "manual_ankles_usable": "",
                        "manual_notes": "",
                    }
                    audits.append(audit)
                if not cv2.imwrite(str(frame_dir / f"frame_{frame.index:08d}.jpg"), canvas):
                    raise OSError("Could not write detection audit frame")
                next_audit = frame.seconds + audit_interval
            # Preview is sampled, not source-rate motion; label that explicitly in-frame.
            scale = min(1.0, 1600 / canvas.shape[1])
            preview = cv2.resize(
                canvas, (round(canvas.shape[1] * scale), round(canvas.shape[0] * scale))
            )
            cv2.putText(
                preview,
                f"AUDIT PREVIEW | source {frame.seconds:.2f}s | IDs are frame-local",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output / "pilot_preview.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    settings["sample_fps"],
                    (preview.shape[1], preview.shape[0]),
                )
                if not writer.isOpened():
                    raise OSError("Could not create MP4 preview")
            writer.write(preview)
            max_ram = max(max_ram, process.memory_info().rss)
            timing.append(
                {
                    **fields,
                    "person_detections": len(people),
                    "detector_seconds": detection_time,
                    "pose_seconds": pose_time,
                }
            )
            if len(timing) % 20 == 0:
                print(
                    f"Processed {len(timing)} sampled frames; {len(audits)} crop audits", flush=True
                )
    finally:
        if writer is not None:
            writer.release()
    if not timing:
        raise ValueError("Pilot interval has no decoded frames")
    elapsed = time.perf_counter() - started
    fields = list(timing[0])[:6]
    _csv(output / "pilot_frame_metrics.csv", timing, list(timing[0]))
    _csv(
        output / "pilot_detections.csv",
        detections,
        fields + ["detection_index", "bbox_xyxy", "confidence"],
    )
    _csv(
        output / "pilot_keypoints.csv",
        poses,
        fields + ["audit_id", "name", "confidence", "valid", "x", "y"],
    )
    _csv(output / "crop_audit.csv", audits, list(audits[0]) if audits else ["audit_id"])
    mean_detector = sum(row["detector_seconds"] for row in timing) / len(timing)
    mean_pose = sum(row["pose_seconds"] for row in timing) / len(audits) if audits else None
    report = {
        **provenance(),
        "status": "pilot_executed_manual_review_required",
        "config": config,
        "video": metadata,
        "sampled_frames": len(timing),
        "crop_audits": len(audits),
        "detections": len(detections),
        "elapsed_excluding_model_load_warmup_seconds": elapsed,
        "mean_detector_seconds_per_sample": mean_detector,
        "mean_pose_seconds_per_audited_crop": mean_pose,
        "process_peak_sampled_rss_gib": max_ram / 1024**3,
        "peak_torch_vram_mib": torch.cuda.max_memory_allocated(settings["device"]) / 1024**2
        if settings["device"].startswith("cuda")
        else None,
        "gait_joint_confidence_passes": sum(row["gait_joints_confident"] for row in audits),
        "model_sha256": {
            name: sha256_file(config["models_dir"] / name)
            for name in (settings["detector"], settings["pose_model"])
        },
        "illustrative_model_compute_estimate": {
            "detector_90000_samples_minutes": mean_detector * 90000 / 60,
            "pose_108000_crops_minutes": mean_pose * 108000 / 60 if mean_pose else None,
            "exclusions": "decoding, I/O, tracking, calibration, review, batch differences",
        },
        "limitations": [
            "Audited people are anonymous frame-local detections, not persistent player tracks.",
            "Confidence/availability is not anatomical accuracy; complete crop_audit.csv manually.",
            "Detection recall/misses require manual/source-label comparison.",
            "Sparse feasibility sampling cannot support cadence or fatigue analysis.",
            "Calibration, annotation alignment and player selection remain unverified.",
        ],
    }
    write_json(output / "pilot_report.json", report)
    write_json(output / "pilot_status.json", {"status": "completed_review_required"})
    return report
