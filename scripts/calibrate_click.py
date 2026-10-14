"""Collect distributed ground reference clicks on a permitted local video."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.calibrate import fit_ground_mapping
from src.common import load_config
from src.video_io import iter_video_frames


def collect_points(config: dict, references: list, destination: Path) -> None:
    """Pair clicked pitch features with known meter coordinates, then validate the warp.

    Distributed ground-plane correspondences reduce apparent motion from poor
    geometry. Elevated body landmarks must not be used as calibration references.
    """
    if len(references) < 8 or len({tuple(point) for point in references}) != len(references):
        raise ValueError(
            "Use at least eight distinct known pitch coordinates for held-out validation"
        )
    if not np.isfinite(references).all():
        raise ValueError("World reference points must be finite numeric pairs")
    settings = config["pilot"]
    frame = next(iter_video_frames(config["video"]["path"], settings["start_seconds"], 0.2, 25))
    scale = min(1.0, 1920 / frame.image.shape[1])
    preview = cv2.resize(
        frame.image, (round(frame.image.shape[1] * scale), round(frame.image.shape[0] * scale))
    )
    points = []
    name = "Pitch calibration: left-click; U undo; Esc cancel"
    cv2.namedWindow(name)

    def click(event, x, y, flags, param):
        """Record source-image pixels without confusing display scale with pitch meters."""
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(references):
            points.append([x / scale, y / scale])

    cv2.setMouseCallback(name, click)
    try:
        while len(points) < len(references):
            canvas = preview.copy()
            for i, pixel in enumerate(points):
                cv2.circle(
                    canvas,
                    tuple(np.round(np.array(pixel) * scale).astype(int)),
                    4,
                    (0, 255, 255),
                    -1,
                )
            cv2.putText(
                canvas,
                f"Click known field point {references[len(points)]} meters",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow(name, canvas)
            key = cv2.waitKey(20) & 255
            if key == 27:
                raise RuntimeError("Calibration canceled; no mapping was published")
            if key in (ord("u"), ord("U")) and points:
                points.pop()
    finally:
        cv2.destroyWindow(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {f"({x},{y})": pixel for (x, y), pixel in zip(references, points, strict=True)},
            indent=2,
        )
    )
    fit_ground_mapping(
        destination,
        destination.with_name(destination.stem + "_mapping.json"),
        image_size=(frame.image.shape[1], frame.image.shape[0]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--world-points",
        type=Path,
        required=True,
        help="JSON list of known (x,y) pitch coordinates",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect_points(load_config(args.config), json.loads(args.world_points.read_text()), args.output)
