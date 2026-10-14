"""Render a continuous native-rate evidence clip without running another neural model."""

import json

import av
import cv2
import numpy as np
import pandas as pd

from src.common import ROOT, load_config, sha256_file, write_json


def render_demo():
    """Show cached joints, independent valid speed and minimap with visible unavailable states.

    The clip is selected around the previously reviewed running bout for inspection,
    not to demonstrate a decline. Seeking is used only for presentation; source PTS
    must match the stored measurements within one millisecond.
    """
    config = load_config(ROOT / "configs/study_117093_second.yaml")
    data = pd.read_parquet(ROOT / "results/metrics_csv/117093_second/frame_metrics.parquet")
    data = data[data.player_id == "A_blue_36"].set_index("source_frame_index")
    output = ROOT / "results/phase4"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "annotated_demo.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError("Could not open preview writer")
    start, end = 1413.0, 1423.0
    edges = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    count = 0
    try:
        with av.open(str(config["video"]["path"])) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            origin = float(stream.start_time * stream.time_base) if stream.start_time else 0.0
            container.seek(
                int((start + origin) / float(stream.time_base)), stream=stream, backward=True
            )
            for frame in container.decode(stream):
                seconds = float(frame.pts * stream.time_base) - origin
                if seconds < start - 1e-6:
                    continue
                if seconds >= end - 1e-6:
                    break
                index = round(seconds * 25)
                row = data.loc[index]
                if abs(row.video_timestamp - seconds) > 0.001:
                    raise ValueError("Preview/source timestamp mismatch")
                image = frame.to_ndarray(format="bgr24")
                box = row[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
                for a, b in edges:
                    points = row[[a + "_x", a + "_y", b + "_x", b + "_y"]].to_numpy(float)
                    if np.isfinite(points).all():
                        cv2.line(
                            image,
                            tuple(np.round(points[:2]).astype(int)),
                            tuple(np.round(points[2:]).astype(int)),
                            (20, 40, 255),
                            3,
                        )
                if np.isfinite(box).all():
                    cv2.rectangle(
                        image,
                        tuple(np.round(box[:2]).astype(int)),
                        tuple(np.round(box[2:]).astype(int)),
                        (0, 230, 80),
                        3,
                    )
                canvas = np.full((720, 1280, 3), 245, dtype=np.uint8)
                cv2.putText(
                    canvas,
                    f"A_blue_36 | source {seconds:.2f}s | "
                    f"configured match {row.timestamp / 60:.2f}min",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (30, 50, 60),
                    2,
                )
                speed = f"{row.speed:.2f} m/s" if row.movement_valid else "unavailable"
                cv2.putText(
                    canvas,
                    f"Independent ground speed: {speed} | pose: {row.pose_status}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (30, 50, 60),
                    2,
                )
                canvas[80:418] = cv2.resize(image, (1280, 338))
                if np.isfinite(box).all():
                    center = (box[0] + box[2]) / 2
                    x1, x2 = max(0, round(center - 180)), min(image.shape[1], round(center + 180))
                    y1, y2 = max(0, round(box[1] - 40)), min(image.shape[0], round(box[3] + 40))
                    crop = image[y1:y2, x1:x2]
                    if crop.size:
                        canvas[450:690, 20:500] = cv2.resize(crop, (480, 240))
                cv2.rectangle(canvas, (535, 455), (955, 687), (40, 120, 50), 2)
                cv2.line(canvas, (745, 455), (745, 687), (40, 120, 50), 1)
                if row.movement_valid:
                    px, py = (
                        round(535 + row.world_x / 105 * 420),
                        round(455 + row.world_y / 68 * 232),
                    )
                    cv2.circle(canvas, (px, py), 6, (220, 70, 25), -1)
                cv2.putText(
                    canvas,
                    "Source crop + measured ground minimap",
                    (535, 438),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (30, 50, 60),
                    1,
                )
                for j, text in enumerate(
                    [
                        "2D projected joints",
                        "Confidence gaps stay missing",
                        "No physiological diagnosis",
                        "25 fps continuous source",
                        "10-second reviewed-bout excerpt",
                    ]
                ):
                    cv2.putText(
                        canvas,
                        text,
                        (975, 470 + 35 * j),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (30, 50, 60),
                        1,
                    )
                writer.write(canvas)
                if count == 160:
                    cv2.imwrite(str(output / "annotated_demo_preview.png"), canvas)
                count += 1
    finally:
        writer.release()
    if count != 250:
        raise ValueError(f"Expected 250 continuous frames, got {count}")
    report = {
        "status": "completed",
        "path": str(path),
        "frames": count,
        "fps": 25,
        "source_interval_seconds": [start, end],
        "sha256": sha256_file(path),
        "selection": "Previously reviewed running bout; not selected on decline",
        "source_pts_alignment": "all frames matched stored video_timestamp within 1ms",
    }
    write_json(output / "annotated_demo.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(render_demo(), indent=2))
