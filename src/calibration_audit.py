"""Inspect supplied camera assets without assuming their coordinate direction."""

import json
import re
import zipfile
from pathlib import Path

import numpy as np


def audit_calibration(keypoints: Path) -> dict:
    """Establish a calibration path, while leaving physical projection unverified.

    A finite matrix is not proof of correct calibration. Panoramic distortion maps,
    source images and pitch homographies can refer to different pixel spaces.
    """
    points = json.loads(keypoints.read_text("utf-8"))
    if not isinstance(points, dict):
        raise ValueError("Expected a dictionary of pitch-coordinate labels to pixel points")
    numeric = []
    for label, pixel in points.items():
        if re.fullmatch(r"\(\s*-?[\d.]+\s*,\s*-?[\d.]+\s*\)", label):
            if len(pixel) != 2 or not np.isfinite(np.asarray(pixel, dtype=float)).all():
                raise ValueError(f"Invalid field keypoint: {label}")
            numeric.append(pixel)
    report = {
        "path": str(keypoints),
        "field_correspondence_count": len(numeric),
        "arrays": {},
        "validation_status": "assets_inspected_projection_not_validated",
    }
    if numeric:
        report["keypoint_pixel_bounds"] = {
            "min": np.min(numeric, axis=0).tolist(),
            "max": np.max(numeric, axis=0).tolist(),
        }
    prefix = keypoints.name.removesuffix("_keypoints.json")
    for suffix in ("homography.npy", "mapx.npy", "mapy.npy", "camera_intrinsics.npz"):
        path = keypoints.parent / f"{prefix}_{suffix}"
        if not path.exists():
            report["arrays"][suffix] = {"exists": False}
            continue
        if path.suffix == ".npz":
            entries = {}
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    with archive.open(member) as handle:
                        version = np.lib.format.read_magic(handle)
                        reader = (
                            np.lib.format.read_array_header_1_0
                            if version == (1, 0)
                            else np.lib.format.read_array_header_2_0
                        )
                        shape, _, dtype = reader(handle)
                        entries[member] = {
                            "shape": list(shape),
                            "dtype": str(dtype),
                            "object_payload_not_loaded": dtype.hasobject,
                        }
            report["arrays"][suffix] = entries
        else:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            report["arrays"][suffix] = {"shape": list(array.shape), "dtype": str(array.dtype)}
            if suffix == "homography.npy":
                report["arrays"][suffix]["finite_and_invertible"] = bool(
                    array.shape == (3, 3)
                    and np.isfinite(array).all()
                    and np.linalg.matrix_rank(array) == 3
                )
    report["required_checks"] = [
        "Confirm whether keypoint pixels belong to raw, padded or undistorted video.",
        "Verify mapx/mapy direction and source dimensions before remapping.",
        "Check pitch origin/axis direction, homography direction and held-out line overlays.",
        "Only ground-contact anchors can be mapped to pitch positions; hips remain elevated.",
    ]
    return report
