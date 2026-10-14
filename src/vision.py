"""Pretrained per-image baselines; persistent player tracking belongs to Phase 2."""

import math
from pathlib import Path

import cv2
import numpy as np

from src.common import configure_ultralytics


def load_model(directory: Path, name: str):
    """Load explicit pretrained weights, keeping downloads inside the project."""
    configure_ultralytics()
    from ultralytics import YOLO

    directory.mkdir(parents=True, exist_ok=True)
    return YOLO(str(directory / name))


def tile_bounds(width: int, height: int, size: int, overlap: float):
    """Cover source pixels with overlapping tiles to preserve distant-player detail."""
    if size < 32 or not 0 <= overlap < 1:
        raise ValueError("tile_size must be >=32 and overlap in [0,1)")
    step = max(1, round(size * (1 - overlap)))

    def starts(length):
        last = max(0, length - size)
        return sorted(set([*range(0, last + 1, step), last]))

    for y in starts(height):
        for x in starts(width):
            yield x, y, min(x + size, width), min(y + size, height)


def detect_people(model, image: np.ndarray, settings: dict) -> list[dict]:
    """Detect people, optionally in native-resolution tiles, merging overlap duplicates.

    Person detections include referees/staff and do not establish player identity.
    The low threshold preserves candidates needed by later ByteTrack association.
    """
    height, width = image.shape[:2]
    size = settings.get("tile_size")
    bounds = (
        list(tile_bounds(width, height, int(size), settings.get("tile_overlap", 0.15)))
        if size
        else [(0, 0, width, height)]
    )
    boxes, scores = [], []
    for x1, y1, x2, y2 in bounds:
        result = model.predict(
            image[y1:y2, x1:x2],
            classes=[0],
            conf=settings["detection_confidence"],
            imgsz=settings["imgsz"],
            device=settings["device"],
            verbose=False,
        )[0]
        for box, score in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=True
        ):
            boxes.append(
                [float(box[0] + x1), float(box[1] + y1), float(box[2] + x1), float(box[3] + y1)]
            )
            scores.append(float(score))
    if not boxes:
        return []
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
    keep = np.asarray(cv2.dnn.NMSBoxes(xywh, scores, 0.0, 0.5)).reshape(-1)
    return [
        {"detection_index": i, "bbox_xyxy": boxes[int(k)], "confidence": scores[int(k)]}
        for i, k in enumerate(keep)
    ]


def padded_crop(image: np.ndarray, bbox: list[float], padding: float) -> tuple:
    """Preserve native pixels and the exact clipped origin for restoring joint coordinates."""
    if padding < 0 or len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
        raise ValueError("Invalid padding or bounding box")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box must have positive area")
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    height, width = image.shape[:2]
    bounds = (
        max(0, math.floor(x1 - dx)),
        max(0, math.floor(y1 - dy)),
        min(width, math.ceil(x2 + dx)),
        min(height, math.ceil(y2 + dy)),
    )
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Bounding box does not intersect the source image")
    return image[bounds[1] : bounds[3], bounds[0] : bounds[2]].copy(), bounds


def box_iou(a, b) -> float:
    """Associate a crop pose with its selected person instead of a neighboring player."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def pose_crop(model, crop: np.ndarray, bounds: tuple, target_box: list, settings: dict) -> dict:
    """Return source-image joints and a confidence gate, not a pose-accuracy claim.

    YOLO restores letterbox scaling internally to input-crop pixels; only the clipped
    crop origin is added here. Low-confidence joints are kept explicitly missing.
    """
    result = model.predict(crop, conf=0.15, imgsz=640, device=settings["device"], verbose=False)[0]
    return pose_result(result, bounds, target_box, settings)


def pose_result(result, bounds: tuple, target_box: list, settings: dict) -> dict:
    """Restore one batch result to source coordinates with the same visibility gates."""
    if result.keypoints is None or len(result.keypoints) == 0:
        return {"status": "no_pose", "joints": [], "gait_joints_confident": False}
    local_box = [
        target_box[0] - bounds[0],
        target_box[1] - bounds[1],
        target_box[2] - bounds[0],
        target_box[3] - bounds[1],
    ]
    overlaps = [box_iou(box, local_box) for box in result.boxes.xyxy.cpu().numpy()]
    selected = int(np.argmax(overlaps))
    if overlaps[selected] < 0.2:
        return {"status": "ambiguous_pose", "joints": [], "gait_joints_confident": False}
    xy = result.keypoints.xy[selected].cpu().numpy()
    confidence = result.keypoints.conf[selected].cpu().numpy()
    names = [
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
    joints = []
    for index, (point, conf) in enumerate(zip(xy, confidence, strict=True)):
        valid = bool(conf >= settings["joint_confidence"])
        joints.append(
            {
                "name": names[index],
                "confidence": float(conf),
                "valid": valid,
                "x": float(point[0] + bounds[0]) if valid else None,
                "y": float(point[1] + bounds[1]) if valid else None,
            }
        )
    return {
        "status": "pose_returned",
        "association_iou": overlaps[selected],
        "joints": joints,
        "gait_joints_confident": all(joints[i]["valid"] for i in range(11, 17)),
    }
