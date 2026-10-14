"""Native-rate target pose extraction in bounded batches from cached tracking ROIs."""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.common import provenance, sha256_file, write_json
from src.metrics import movement_metrics, straight_running_mask
from src.pose_extract import flatten_pose, save_pose_gaps
from src.video_io import iter_video_frames, probe_video
from src.vision import load_model, padded_crop, pose_result


def running_candidate_mask(frames: pd.DataFrame, minimum_seconds: float = 4) -> pd.Series:
    """Select continuous motion bouts without using unobserved/interpolated anatomy.

    Selection is based on independent calibrated movement; pose availability is
    assessed afterward. This avoids spending GPU time on stationary/occluded frames.
    """
    selected = pd.Series(False, index=frames.index)
    for _, data in frames.groupby(["clip_name", "player_id"], sort=False):
        valid = straight_running_mask(data)
        boundary = (
            ~valid
            | ~valid.shift(fill_value=False)
            | (data.timestamp.diff() > 0.075)
            | data.identity_segment.ne(data.identity_segment.shift())
        )
        for _, part in data[valid].groupby(boundary.cumsum()[valid]):
            if (
                len(part) > 1
                and part.timestamp.iloc[-1] - part.timestamp.iloc[0] >= minimum_seconds
            ):
                selected.loc[part.index] = True
    return selected


def pose_from_tracks(config: dict, tracks_path: Path, output: Path, batch_size: int = 8) -> dict:
    """Extract native-frame poses from selected cached ROIs, retaining all missing rows.

    Batched crops preserve temporal sampling while bounding GPU memory. Primary
    field positions retain the consistent tracking anchor; pose anatomy is used for
    projected gait proxies. No detector rerun or joint interpolation is performed.
    """
    if not 1 <= batch_size <= 8:
        raise ValueError("Pose batches are bounded to 1–8 crops on the 8 GB GPU")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "keypoints_positions.csv"
    if destination.exists():
        raise ValueError("Pose output already exists; preserve it or use a new directory")
    frames = pd.read_csv(tracks_path, dtype={"player_id": str, "clip_name": str})
    frames = movement_metrics(frames)
    observed = frames.observation_status.isin(["detected", "short_box_propagation"])
    mode = config["pipeline"].get("pose_mode", "running_bouts")
    candidate = running_candidate_mask(frames) if mode == "running_bouts" else observed
    candidate &= observed & frames.position_valid
    requested = candidate.copy()
    if "occlusion_risk" in frames:
        candidate &= ~frames.occlusion_risk.fillna(False).astype(bool)
    frames["pose_status"] = "not_requested_outside_running_bout"
    frames.loc[requested & ~candidate, "pose_status"] = "excluded_occlusion_risk"
    frames["gait_pose_valid"] = False
    for column in flatten_pose(None):
        frames[column] = np.nan
    metadata = probe_video(config["video"]["path"])
    index_to_rows = {
        int(frame): list(part.index)
        for frame, part in frames[candidate].groupby("source_frame_index")
    }
    settings = config["pilot"]
    model = load_model(config["models_dir"], settings["pose_model"]) if index_to_rows else None
    device = settings["device"]
    if device.startswith("cuda") and model is not None:
        torch.cuda.set_device(torch.device(device))
        torch.cuda.reset_peak_memory_stats(device)
    pending, audit_images, sample_rows = [], {}, []
    preview = None
    poses, batches = 0, 0
    started = time.perf_counter()
    audit_dir = output / "pose_audit_frames"
    audit_dir.mkdir(exist_ok=True)
    write_json(
        output / "pose_status.json",
        {**provenance(), "status": "running", "candidate_rows": int(candidate.sum())},
    )

    def complete_batch():
        nonlocal poses, batches, preview
        if not pending:
            return
        results = model.predict(
            [item["crop"] for item in pending],
            imgsz=640,
            batch=len(pending),
            conf=0.15,
            device=device,
            verbose=False,
            stream=False,
        )
        for item, result in zip(pending, results, strict=True):
            row_index = item["row_index"]
            pose = pose_result(result, item["bounds"], item["bbox"], settings)
            values = flatten_pose(pose)
            for column, value in values.items():
                frames.at[row_index, column] = value
            frames.at[row_index, "pose_status"] = pose["status"]
            frames.at[row_index, "gait_pose_valid"] = pose["gait_joints_confident"]
            poses += 1
            source_frame = int(frames.at[row_index, "source_frame_index"])
            audit = audit_images.get(source_frame)
            if audit is not None:
                canvas = audit["image"]
                box = item["bbox"]
                cv2.rectangle(
                    canvas,
                    tuple(np.round(box[:2]).astype(int)),
                    tuple(np.round(box[2:]).astype(int)),
                    (0, 220, 0),
                    2,
                )
                for joint in pose["joints"]:
                    if joint["valid"]:
                        cv2.circle(
                            canvas, (round(joint["x"]), round(joint["y"])), 3, (0, 255, 255), -1
                        )
                audit["remaining"] -= 1
                if audit["remaining"] == 0:
                    seconds = float(frames.at[row_index, "timestamp"])
                    scale = min(1.0, 1600 / canvas.shape[1])
                    image = cv2.resize(
                        canvas, (round(canvas.shape[1] * scale), round(canvas.shape[0] * scale))
                    )
                    cv2.putText(
                        image,
                        f"POSE AUDIT MONTAGE | source {seconds:.2f}s",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    if preview is None:
                        preview = cv2.VideoWriter(
                            str(output / "pose_audit.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            metadata["average_fps"],
                            (image.shape[1], image.shape[0]),
                        )
                        if not preview.isOpened():
                            raise OSError("Could not write pose audit video")
                    preview.write(image)
                    if source_frame % 25 == 0:
                        cv2.imwrite(str(audit_dir / f"frame_{source_frame:08d}.jpg"), canvas)
                    del audit_images[source_frame]
            if poses % 1000 == 0:
                print(f"{config['pipeline']['clip_name']}: {poses} native target poses", flush=True)
            if pose["gait_joints_confident"] and len(sample_rows) < 30:
                sample_rows.append(
                    {
                        "row_index": row_index,
                        "source_frame": source_frame,
                        "timestamp": frames.at[row_index, "timestamp"],
                    }
                )
        batches += 1
        pending.clear()

    audited_frame_count = 0
    try:
        for frame in iter_video_frames(
            config["video"]["path"],
            settings["start_seconds"],
            settings["duration_seconds"],
            metadata["average_fps"],
            required_indices=set(index_to_rows),
        ):
            row_indices = index_to_rows[frame.index]
            if audited_frame_count < 1000:
                audit_images[frame.index] = {
                    "image": frame.image.copy(),
                    "remaining": len(row_indices),
                }
                audited_frame_count += 1
            for index in row_indices:
                row = frames.loc[index]
                box = [float(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
                try:
                    crop, bounds = padded_crop(frame.image, box, settings["crop_padding"])
                except ValueError:
                    frames.at[index, "pose_status"] = "invalid_crop"
                    if frame.index in audit_images:
                        audit_images[frame.index]["remaining"] -= 1
                    continue
                pending.append({"row_index": index, "crop": crop, "bounds": bounds, "bbox": box})
                if len(pending) == batch_size:
                    complete_batch()
        complete_batch()
    finally:
        if preview is not None:
            preview.release()
    frames.to_csv(destination, index=False)
    frames.to_parquet(output / "keypoints_positions.parquet", index=False)
    save_pose_gaps(frames, output / "pose_gaps.csv")
    report = {
        **provenance(),
        "status": "completed",
        "tracks_source": str(tracks_path),
        "video": metadata,
        "pose_mode": mode,
        "native_fps": metadata["average_fps"],
        "candidate_rows": int(candidate.sum()),
        "pose_calls": poses,
        "batches": batches,
        "confident_gait_pose_rows": int(frames.gait_pose_valid.sum()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_torch_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2
        if model is not None and device.startswith("cuda")
        else None,
        "weight_sha256": sha256_file(config["models_dir"] / settings["pose_model"]),
        "joint_interpolation_performed": False,
        "position_anchor": "consistent_cached_tracking_anchor",
        "pose_status_counts": {
            str(key): int(value) for key, value in frames.pose_status.value_counts().items()
        },
        "accepted_pose_audit_samples": sample_rows,
    }
    write_json(output / "pose_report.json", report)
    write_json(output / "pose_status.json", {"status": "completed"})
    return report
