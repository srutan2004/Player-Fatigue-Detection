"""ByteTrack extraction with reviewed/reference-assisted identity and explicit gaps."""

import csv
import json
import os
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil
import supervision as sv
import torch
from scipy.optimize import linear_sum_assignment

from src.calibrate import GroundMapping
from src.common import provenance, sha256_file, write_json
from src.pose_extract import flatten_pose
from src.quality import CameraGuard
from src.reference import reference_people
from src.video_io import iter_video_frames, probe_video
from src.vision import box_iou, detect_people, load_model, padded_crop, pose_crop


def _shirt_color(image, box):
    """Use kit color as supporting team evidence; it cannot identify a teammate."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    a, b = max(0, round(x1 + 0.25 * w)), min(image.shape[1], round(x2 - 0.25 * w))
    c, d = max(0, round(y1 + 0.2 * h)), min(image.shape[0], round(y1 + 0.5 * h))
    if b <= a or d <= c:
        return "unknown"
    hsv = cv2.cvtColor(image[c:d, a:b], cv2.COLOR_BGR2HSV)
    blue = (
        (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 135) & (hsv[:, :, 1] >= 140) & (hsv[:, :, 2] >= 40)
    ).mean()
    white = ((hsv[:, :, 1] < 120) & (hsv[:, :, 2] > 80)).mean()
    # White kits contain blue stripes; shadows reduce brightness and add a cool cast.
    # Only a dominant unambiguous palette is supporting evidence. Mixed kits stay unknown.
    if blue >= 0.45 and white >= 0.45:
        return "unknown"
    return "blue" if blue >= 0.45 else "white" if white >= 0.45 else "unknown"


def _reference_links(current, references, mapping, image, colors, prior):
    """Associate reference identities globally, without using reference positions as measurements.

    GSR boxes can be synthetic in shape. Reference pitch position, supporting shirt
    color and weak overlap evidence constrain one-to-one identity links. The result
    is annotation-assisted identity review, not autonomous re-identification accuracy.
    """
    if not current or not references:
        return {}
    refs = [r for r in references if r["pitch_x"] is not None and r["pitch_y"] is not None]
    if not refs:
        return {}
    ids = list(current)
    boxes = np.asarray([current[track] for track in ids])
    world = mapping.transform(np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
    reference_world = np.array([[r["pitch_x"] + 52.5, r["pitch_y"] + 34] for r in refs])
    distance = np.linalg.norm(reference_world[:, None] - world[None, :], axis=2)
    cost = np.where(np.isfinite(distance), distance, 1000.0)
    kit = [_shirt_color(image, box) for box in boxes] if colors else ["unknown"] * len(boxes)
    for i, ref in enumerate(refs):
        ref_box = [ref[k] for k in ("x1", "y1", "x2", "y2")]
        expected = colors.get(ref["team"])
        for j, box in enumerate(boxes):
            cost[i, j] += 0.5 * (1 - box_iou(box, ref_box))
            if expected and kit[j] != "unknown" and kit[j] != expected:
                cost[i, j] += 10
            if prior.get(str(ref["player"])) == ids[j] and distance[i, j] <= 3:
                cost[i, j] -= 0.5
    # Dummy assignments let an absent target stay missing rather than steal a teammate.
    padded = np.c_[cost, np.full((len(refs), len(refs)), 3.5)]
    rows, columns = linear_sum_assignment(padded)
    links = {}
    for i, j in zip(rows, columns, strict=True):
        if j >= len(ids) or distance[i, j] > 3 or cost[i, j] >= 3.5:
            continue
        ranked = np.sort(cost[i])
        previous = prior.get(str(refs[i]["player"]))
        if len(ranked) > 1 and ranked[1] - ranked[0] < 0.2 and previous != ids[j]:
            continue
        links[str(refs[i]["player"])] = (ids[j], float(distance[i, j]))
    return links


def _visible_tracklet_continuity(
    previous, current, references, target, links, mapping, image, colors, timestamp, maximum_seconds
):
    """Keep a recently seeded, still-visible tracklet through coarse reference noise.

    This never fills missing detections or guesses a new tracklet. Strict motion,
    overlap, kit and reference-distance limits bound the annotation uncertainty.
    """
    if not previous or timestamp - previous.get("last_reference_time", -np.inf) > maximum_seconds:
        return None
    track = previous["track"]
    if track not in current or track in {link[0] for link in links.values()}:
        return None
    box = current[track]
    if any(
        box_iou(box, other) > 0.35 for other_track, other in current.items() if other_track != track
    ):
        return None
    refs = [r for r in references if str(r["player"]) == str(target["reference_player_id"])]
    if not refs or refs[0]["pitch_x"] is None or refs[0]["pitch_y"] is None:
        return None
    world = mapping.transform([[(box[0] + box[2]) / 2, box[3]]])[0]
    old = previous["box"]
    old_world = mapping.transform([[(old[0] + old[2]) / 2, old[3]]])[0]
    reference_world = np.array([refs[0]["pitch_x"] + 52.5, refs[0]["pitch_y"] + 34])
    dt = timestamp - previous["time"]
    if (
        dt <= 0
        or not np.isfinite(world).all()
        or not np.isfinite(old_world).all()
        or np.linalg.norm(world - reference_world) > 4.5
        or np.linalg.norm(world - old_world) > 12 * dt + 0.3
    ):
        return None
    expected = colors.get(refs[0]["team"])
    kit = _shirt_color(image, box) if expected else "unknown"
    if expected and kit not in ("unknown", expected):
        return None
    return track


class GapLog:
    """Store missing observations as intervals rather than interpolated movement."""

    def __init__(self):
        self.open = {}
        self.rows = []

    def observe(self, player, frame, timestamp, reason=None):
        """Close a gap at the first recovered frame, retaining its exclusion reason."""
        pending = self.open.get(player)
        if pending and pending["reason"] != reason:
            self.rows.append(
                {**pending, "end_frame_exclusive": frame, "end_time_exclusive": timestamp}
            )
            del self.open[player]
        if reason and player not in self.open:
            self.open[player] = {
                "player_id": player,
                "start_frame": frame,
                "start_time": timestamp,
                "reason": reason,
            }

    def close(self, frame, timestamp):
        """Bound any final loss at the segment end so it remains visible in the report."""
        for player in list(self.open):
            self.observe(player, frame, timestamp)


def extract_segment(config: dict) -> dict:
    """Extract pretrained poses and calibrated anchors for selected people at native FPS.

    ByteTrack runs at the configured detection rate. Between updates, short-lived
    crops use tagged box propagation; ambiguous identities are excluded. Optional
    annotation-assisted links replace manual review and are explicitly reported,
    never counted as autonomous tracking accuracy.
    """
    settings, video = config["pilot"], config["video"]
    pipeline = config.get("pipeline")
    if not isinstance(pipeline, dict) or not isinstance(pipeline.get("targets"), list):
        raise ValueError("Extraction needs pipeline.targets; select explicit players in the YAML")
    targets = pipeline["targets"]
    if not 0 <= pipeline.get("identity_continuity_seconds", 0) <= 1:
        raise ValueError("Recently verified tracklet continuity is bounded to one second")
    if not targets or len(targets) > 3:
        raise ValueError("Choose one to three explicit target players")
    for target in targets:
        previous_end = -1
        for link in sorted(
            target.get("reviewed_tracklets", []), key=lambda link: link["start_seconds"]
        ):
            if (
                link.get("reviewed") is not True
                or link["start_seconds"] < previous_end
                or link["end_seconds"] <= link["start_seconds"]
            ):
                raise ValueError(
                    "Manual tracklet links must be reviewed, positive and non-overlapping"
                )
            previous_end = link["end_seconds"]
    output = config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    frame_path = output / "keypoints_positions.csv"
    if frame_path.exists():
        raise ValueError("Extraction already exists; choose a new output directory")
    metadata = probe_video(video["path"])
    fps = metadata["average_fps"]
    if not fps or fps < 20:
        raise ValueError("Native-rate gait extraction requires a source at >=20 FPS")
    if (
        settings["start_seconds"] + settings["duration_seconds"]
        > metadata["duration_seconds_metadata"] + 0.1
    ):
        raise ValueError("Requested segment extends past the source video")
    mapping = GroundMapping.load(Path(pipeline["ground_mapping"]))
    if not mapping.validated:
        raise ValueError("Field mapping failed validation; metric positions are disabled")
    track_fps = float(pipeline.get("tracking_fps", fps / 2))
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,  # Supervision scales this 30-FPS buffer by frame_rate.
        frame_rate=track_fps,
    )
    cached_path = pipeline.get("cached_tracks")
    cached = None
    if cached_path:
        cached_path = Path(cached_path)
        cached_report = json.loads((cached_path.parent / "extraction_report.json").read_text())
        original = cached_report["config"]
        if (
            Path(original["video"]["path"]).resolve() != video["path"]
            or cached_report["detection_fps"] != track_fps
            or any(
                original["pilot"][key] != settings[key]
                for key in (
                    "start_seconds",
                    "duration_seconds",
                    "imgsz",
                    "detector",
                    "detection_confidence",
                )
            )
        ):
            raise ValueError("Cached raw tracker source/settings differ from this extraction")
        table = pd.read_csv(cached_path)
        cached = {int(frame): part for frame, part in table.groupby("frame", sort=False)}
    detector = load_model(config["models_dir"], settings["detector"]) if cached is None else None
    extract_pose = bool(pipeline.get("extract_pose", True))
    pose_model = load_model(config["models_dir"], settings["pose_model"]) if extract_pose else None
    device = settings["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.startswith("cuda") and (cached is None or extract_pose):
        torch.cuda.set_device(torch.device(device))
        torch.cuda.reset_peak_memory_stats(device)
    reference = (
        sqlite3.connect(pipeline["reference_index"]) if pipeline.get("reference_index") else None
    )
    assisted = bool(pipeline.get("reference_assisted_identity", False))
    if assisted and reference is None:
        raise ValueError("Reference-assisted identity requires the reference index")
    gap_log = GapLog()
    states, identity_links, identity_segments = {}, [], {t["player_id"]: 0 for t in targets}
    last_linked_tracks = {}
    tracks, quality = [], []
    next_detection = settings["start_seconds"]
    camera_guard = CameraGuard()
    camera_checks = []
    camera_valid = True
    started, frame_count, pose_count = time.perf_counter(), 0, 0
    detector_seconds = pose_seconds = 0.0
    preview_writer = None
    last_frame = None
    write_json(
        output / "extraction_status.json",
        {
            **provenance(),
            "status": "running",
            "pid": os.getpid(),
            "process_create_time": psutil.Process().create_time(),
        },
    )
    header = [
        "clip_name",
        "player_id",
        "source_frame_index",
        "timestamp",
        "timestamp_source",
        "detection_confidence",
        "box_propagation_age",
        "occlusion_risk",
        "match_seconds",
        "half",
        "identity_segment",
        "tracklet_id",
        "identity_source",
        "observation_status",
        "pose_status",
        "box_height",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "gait_pose_valid",
        "ground_pixel_x",
        "ground_pixel_y",
        "anchor_type",
        "position_valid",
        "world_x",
        "world_y",
        *flatten_pose(None),
    ]
    handle = frame_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    try:
        for frame in iter_video_frames(
            video["path"], settings["start_seconds"], settings["duration_seconds"], fps
        ):
            last_frame = frame
            frame_count += 1
            is_update = frame.seconds + 1e-8 >= next_detection
            if is_update:
                t0 = time.perf_counter()
                if cached is None:
                    people = detect_people(detector, frame.image, settings)
                    detector_seconds += time.perf_counter() - t0
                    boxes = np.asarray([p["bbox_xyxy"] for p in people], dtype=float).reshape(-1, 4)
                    scores = np.asarray([p["confidence"] for p in people], dtype=float)
                    found = tracker.update_with_detections(
                        sv.Detections(
                            xyxy=boxes, confidence=scores, class_id=np.zeros(len(boxes), dtype=int)
                        )
                    )
                    current = (
                        {
                            int(track): box
                            for track, box in zip(found.tracker_id, found.xyxy, strict=True)
                        }
                        if found.tracker_id is not None
                        else {}
                    )
                    current_scores = (
                        {
                            int(track): float(score)
                            for track, score in zip(found.tracker_id, found.confidence, strict=True)
                        }
                        if found.tracker_id is not None
                        else {}
                    )
                else:
                    part = cached.get(frame.index)
                    current = (
                        {}
                        if part is None
                        else {
                            int(row.tracklet_id): np.asarray(json.loads(row.bbox_xyxy))
                            for row in part.itertuples()
                        }
                    )
                    current_scores = (
                        {}
                        if part is None
                        else {
                            int(row.tracklet_id): float(row.detection_confidence)
                            for row in part.itertuples()
                        }
                    )
                camera_check = camera_guard.check(frame.image, list(current.values()))
                camera_valid = camera_check["mapping_valid"]
                camera_checks.append({"timestamp": frame.seconds, **camera_check})
                references = reference_people(reference, frame.index) if reference else []
                prior = {
                    str(t.get("reference_player_id")): states[t["player_id"]]["track"]
                    for t in targets
                    if t["player_id"] in states
                }
                reference_links = (
                    _reference_links(
                        current,
                        references,
                        mapping,
                        frame.image,
                        pipeline.get("team_colors", {}),
                        prior,
                    )
                    if assisted
                    else {}
                )
                for track, box in current.items():
                    tracks.append(
                        {
                            "frame": frame.index,
                            "timestamp": frame.seconds,
                            "tracklet_id": track,
                            "bbox_xyxy": json.dumps(box.tolist()),
                            "detection_confidence": current_scores.get(track),
                        }
                    )
                for target in targets:
                    player = target["player_id"]
                    previous = states.get(player)
                    chosen = previous["track"] if previous else target.get("initial_tracklet_id")
                    reason = "manual_initial_selection"
                    if "reviewed_tracklets" in target:
                        reviewed = [
                            link
                            for link in target["reviewed_tracklets"]
                            if link["start_seconds"] <= frame.seconds < link["end_seconds"]
                        ]
                        chosen = int(reviewed[0]["tracklet_id"]) if reviewed else None
                        reason = "manual_reviewed_tracklet"
                    if assisted:
                        link = reference_links.get(str(target["reference_player_id"]))
                        chosen = link[0] if link else None
                        reason = "reference_assisted_pitch_and_color"
                        if chosen is None and pipeline.get("identity_continuity_seconds", 0) > 0:
                            chosen = _visible_tracklet_continuity(
                                previous,
                                current,
                                references,
                                target,
                                reference_links,
                                mapping,
                                frame.image,
                                pipeline.get("team_colors", {}),
                                frame.seconds,
                                pipeline["identity_continuity_seconds"],
                            )
                            reason = "reference_seeded_visible_tracklet_continuity"
                    if chosen in current:
                        box = current[chosen].copy()
                        changed = (
                            player in last_linked_tracks and last_linked_tracks[player] != chosen
                        )
                        if changed:
                            identity_segments[player] += 1
                        if previous is None or changed:
                            identity_links.append(
                                {
                                    "player_id": player,
                                    "source_frame": frame.index,
                                    "tracklet_id": chosen,
                                    "source": reason,
                                    "identity_segment": identity_segments[player],
                                }
                            )
                        last_linked_tracks[player] = chosen
                        velocity = np.zeros(4)
                        if previous is not None and not changed:
                            dt = frame.seconds - previous["time"]
                            if dt > 0:
                                velocity = (box - previous["box"]) / dt
                        states[player] = {
                            "box": box,
                            "time": frame.seconds,
                            "velocity": velocity,
                            "track": chosen,
                            "source": reason,
                            "confidence": current_scores.get(chosen),
                            "last_reference_time": (
                                previous["last_reference_time"]
                                if reason == "reference_seeded_visible_tracklet_continuity"
                                and previous
                                else frame.seconds
                            ),
                        }
                    else:
                        states.pop(player, None)
                next_detection = (
                    settings["start_seconds"]
                    + (int((frame.seconds - settings["start_seconds"]) * track_fps + 1e-7) + 1)
                    / track_fps
                )
            canvas = frame.image.copy() if frame_count % 5 == 1 else None
            for target in targets:
                player = target["player_id"]
                state = states.get(player)
                base = {
                    "clip_name": pipeline["clip_name"],
                    "player_id": player,
                    "source_frame_index": frame.index,
                    "timestamp": frame.seconds,
                    "timestamp_source": frame.timestamp_source,
                    "detection_confidence": None,
                    "box_propagation_age": None,
                    "occlusion_risk": False,
                    "match_seconds": video["match_time_offset_seconds"] + frame.seconds,
                    "half": video["half"],
                    "identity_segment": identity_segments[player],
                    "tracklet_id": None,
                    "identity_source": None,
                    "observation_status": "track_loss",
                    "pose_status": "unavailable",
                    "box_height": None,
                    "bbox_x1": None,
                    "bbox_y1": None,
                    "bbox_x2": None,
                    "bbox_y2": None,
                    "gait_pose_valid": False,
                    "ground_pixel_x": None,
                    "ground_pixel_y": None,
                    "anchor_type": "unavailable",
                    "position_valid": False,
                    "world_x": None,
                    "world_y": None,
                    **flatten_pose(None),
                }
                if state is None or frame.seconds - state["time"] > 0.12:
                    gap_log.observe(player, frame.index, frame.seconds, "track_loss")
                    writer.writerow(base)
                    continue
                box = state["box"] + state["velocity"] * (frame.seconds - state["time"])
                occlusion_risk = any(
                    box_iou(box, other) > 0.35
                    for track, other in current.items()
                    if track != state["track"]
                )
                try:
                    crop, bounds = padded_crop(frame.image, box.tolist(), settings["crop_padding"])
                except ValueError:
                    gap_log.observe(player, frame.index, frame.seconds, "invalid_predicted_crop")
                    base["observation_status"] = "invalid_predicted_crop"
                    writer.writerow(base)
                    continue
                t0 = time.perf_counter()
                pose = (
                    pose_crop(pose_model, crop, bounds, box.tolist(), settings)
                    if extract_pose
                    else {
                        "status": "deferred_to_batched_pose_pass",
                        "joints": [],
                        "gait_joints_confident": False,
                    }
                )
                pose_seconds += time.perf_counter() - t0 if extract_pose else 0.0
                pose_count += int(extract_pose)
                joints = flatten_pose(pose)
                # Retain one primary anchor even when intermittent ankle poses are available.
                anchor = ((box[0] + box[2]) / 2, box[3])
                anchor_type = "bbox_bottom_fallback"
                world = mapping.transform([anchor])[0]
                position_valid = bool(np.isfinite(world).all()) and camera_valid
                base.update(joints)
                base.update(
                    tracklet_id=state["track"],
                    identity_source=state["source"],
                    observation_status="detected" if is_update else "short_box_propagation",
                    pose_status=pose["status"],
                    box_height=float(box[3] - box[1]),
                    bbox_x1=float(box[0]),
                    bbox_y1=float(box[1]),
                    bbox_x2=float(box[2]),
                    bbox_y2=float(box[3]),
                    gait_pose_valid=pose["gait_joints_confident"],
                    detection_confidence=state["confidence"],
                    box_propagation_age=frame.seconds - state["time"],
                    occlusion_risk=occlusion_risk,
                    ground_pixel_x=float(anchor[0]),
                    ground_pixel_y=float(anchor[1]),
                    anchor_type=anchor_type,
                    position_valid=position_valid,
                    world_x=float(world[0]) if position_valid else None,
                    world_y=float(world[1]) if position_valid else None,
                )
                reason = (
                    None
                    if position_valid and (pose["gait_joints_confident"] or not extract_pose)
                    else ("invalid_ground_mapping" if not position_valid else "low_pose_confidence")
                )
                gap_log.observe(player, frame.index, frame.seconds, reason)
                writer.writerow(base)
                if canvas is not None:
                    first, last = (
                        tuple(np.round(box[:2]).astype(int)),
                        tuple(np.round(box[2:]).astype(int)),
                    )
                    cv2.rectangle(canvas, first, last, (0, 220, 0), 2)
                    cv2.putText(
                        canvas, str(player), first, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                    for joint in pose["joints"]:
                        if joint["valid"]:
                            cv2.circle(
                                canvas, (round(joint["x"]), round(joint["y"])), 3, (0, 255, 255), -1
                            )
                if frame_count % 25 == 1:
                    quality.append(
                        {
                            "player_id": player,
                            "timestamp": frame.seconds,
                            "position_valid": position_valid,
                            "gait_joints_confident": pose["gait_joints_confident"],
                            "anchor_type": anchor_type,
                        }
                    )
            if canvas is not None:
                scale = min(1.0, 1600 / canvas.shape[1])
                preview = cv2.resize(
                    canvas, (round(canvas.shape[1] * scale), round(canvas.shape[0] * scale))
                )
                cv2.putText(
                    preview,
                    f"{pipeline['clip_name']} | {frame.seconds:.2f}s | {mapping.method}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )
                if preview_writer is None:
                    preview_writer = cv2.VideoWriter(
                        str(output / "annotated.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps / 5,
                        (preview.shape[1], preview.shape[0]),
                    )
                    if not preview_writer.isOpened():
                        raise OSError("Could not create annotated preview")
                preview_writer.write(preview)
            if frame_count % 1000 == 0:
                handle.flush()
                print(
                    f"{pipeline['clip_name']}: {frame_count} frames, {pose_count} target poses",
                    flush=True,
                )
    finally:
        handle.close()
        if preview_writer is not None:
            preview_writer.release()
        if reference is not None:
            reference.close()
    if last_frame is None:
        raise ValueError("No frames in the configured extraction interval")
    gap_log.close(last_frame.index + 1, last_frame.seconds + 1 / fps)
    for name, rows in (
        ("tracks", tracks),
        ("identity_links", identity_links),
        ("gaps", gap_log.rows),
        ("quality", quality),
        ("camera_checks", camera_checks),
    ):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = (
                list(dict.fromkeys(key for row in rows for key in row)) if rows else ["status"]
            )
            table = csv.DictWriter(stream, fieldnames=columns)
            table.writeheader()
            table.writerows(rows)
    report = {
        **provenance(),
        "status": "completed",
        "config": config,
        "source_frames": frame_count,
        "target_pose_calls": pose_count,
        "detector_seconds": detector_seconds,
        "pose_seconds": pose_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "reference_assisted_identity": assisted,
        "tracking_accuracy_measured": False,
        "cached_raw_tracker_source": str(cached_path) if cached_path else None,
        "raw_tracker_max_time_lost_frames": (
            cached_report.get(
                "raw_tracker_max_time_lost_frames", int(round(track_fps) / 30 * round(track_fps))
            )
            if cached_path
            else tracker.max_time_lost
        ),
        "fresh_tracker_default_lost_horizon_seconds": 1,
        "identity_continuity_seconds": pipeline.get("identity_continuity_seconds", 0),
        "native_pose_fps": fps,
        "detection_fps": track_fps,
        "ground_mapping": mapping.payload,
        "target_csv": str(frame_path),
        "weights_sha256": {
            name: sha256_file(config["models_dir"] / name)
            for name in (settings["detector"], settings["pose_model"])
        },
    }
    write_json(output / "extraction_report.json", report)
    write_json(output / "extraction_status.json", {"status": "completed"})
    return report
