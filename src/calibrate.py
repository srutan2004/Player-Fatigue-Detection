"""Validated ground-plane mappings for fixed-view/panoramic pilot footage."""

import json
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from src.common import write_json


def _terms(points, degree, scale):
    """Use normalized low-order pixel terms to avoid an ill-conditioned field warp."""
    uv = (np.asarray(points) - scale) / scale
    return np.array(
        [uv[:, 0] ** i * uv[:, 1] ** j for i in range(degree + 1) for j in range(degree + 1 - i)]
    ).T


def _fit_rational(pixels, world, degree, scale):
    """Fit a low-order rational planar camera warp, not an anatomical/learned model."""
    a = _terms(pixels, degree, scale)
    b = (world - [52.5, 34]) / [52.5, 34]
    n = a.shape[1]
    matrix = np.r_[
        np.c_[a, np.zeros_like(a), -b[:, 0, None] * a[:, 1:]],
        np.c_[np.zeros_like(a), a, -b[:, 1, None] * a[:, 1:]],
    ]
    return np.linalg.lstsq(
        np.r_[matrix, np.eye(3 * n - 1) * 1e-4],
        np.r_[b[:, 0], b[:, 1], np.zeros(3 * n - 1)],
        rcond=None,
    )[0]


def _rational_transform(points, coefficients, degree, scale):
    """Reject near-singular projections before they can create false pitch motion."""
    terms = _terms(points, degree, scale)
    n = terms.shape[1]
    denominator = 1 + terms[:, 1:] @ coefficients[2 * n :]
    with np.errstate(divide="ignore", invalid="ignore"):
        mapped = np.c_[terms @ coefficients[:n], terms @ coefficients[n : 2 * n]] / denominator[
            :, None
        ] * [52.5, 34] + [52.5, 34]
    mapped[np.abs(denominator) < 0.05] = np.nan
    return mapped


def pixel_to_world(point, homography_matrix) -> np.ndarray:
    """Map a ground-contact pixel to meters; elevated hips violate this geometry."""
    pixels = np.asarray(point, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pixels, np.asarray(homography_matrix)).reshape(-1, 2)


class GroundMapping:
    """Use a global homography or a locally validated panoramic ground mapping.

    Local affine or rational warps account for panoramic distortion inside the
    annotated convex hull. Unavailable/implausible projections stay missing.
    """

    def __init__(self, payload: dict):
        self.payload = payload
        self.method = payload["selected_method"]
        self.validated = payload["validation_passed"]
        self._h = np.asarray(payload["homography"], dtype=float)
        self._local = LinearNDInterpolator(payload["pixel_points"], payload["world_points"])
        self._hull = Delaunay(payload["pixel_points"])

    def transform(self, points) -> np.ndarray:
        """Return meters for valid on-plane anchors, retaining missing local estimates."""
        pixels = np.asarray(points, dtype=float).reshape(-1, 2)
        if not self.validated:
            return np.full_like(pixels, np.nan)
        if self.method == "homography":
            result = pixel_to_world(pixels, self._h)
        elif self.method.startswith("rational_"):
            degree = int(self.method.split("_")[1])
            result = _rational_transform(
                pixels,
                np.asarray(self.payload["rational_coefficients"]),
                degree,
                np.asarray(self.payload["pixel_scale"]),
            )
            result[self._hull.find_simplex(pixels) < 0] = np.nan
        else:
            result = np.asarray(self._local(pixels), dtype=float)
        result[
            (~np.isfinite(pixels).all(axis=1))
            | (result[:, 0] < -2)
            | (result[:, 0] > 107)
            | (result[:, 1] < -2)
            | (result[:, 1] > 70)
        ] = np.nan
        return result

    @classmethod
    def load(cls, path: Path):
        """Load only JSON numeric calibration data; no pickle execution is needed."""
        return cls(json.loads(path.read_text("utf-8")))


def fit_ground_mapping(
    keypoint_path: Path, output: Path, seed: int = 42, image_size: tuple = (4096, 1080)
) -> dict:
    """Compare mappings on withheld correspondences before computing metric movement.

    Pitch labels are corner-origin meters. Global homography is preferred when
    accurate; a local mapping is an explicit panoramic alternative. These tests
    validate field geometry, not foot-anchor localization or physiological fatigue.
    """
    points = json.loads(keypoint_path.read_text("utf-8"))
    world, pixels = [], []
    for label, pixel in points.items():
        if re.fullmatch(r"\(\s*-?[\d.]+\s*,\s*-?[\d.]+\s*\)", label):
            world.append([float(x) for x in label.strip("()").split(",")])
            pixels.append(pixel)
    world, pixels = np.asarray(world), np.asarray(pixels)
    if len(world) < 8 or not np.isfinite(pixels).all():
        raise ValueError("At least eight finite field correspondences are required for validation")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(world))
    count = max(4, round(len(world) * 0.2))
    test, train = indices[:count], indices[count:]
    h, inliers = cv2.findHomography(pixels[train], world[train], cv2.RANSAC, 1.0)
    if h is None:
        raise ValueError("Homography estimation failed")
    candidates = {
        "homography": pixel_to_world(pixels[test], h),
        "piecewise_affine": np.asarray(
            LinearNDInterpolator(pixels[train], world[train])(pixels[test])
        ),
    }
    scale = np.asarray(image_size, dtype=float) / 2
    for degree in (2, 3):
        coefficients = _fit_rational(pixels[train], world[train], degree, scale)
        candidates[f"rational_{degree}"] = _rational_transform(
            pixels[test], coefficients, degree, scale
        )
    validation = {}
    for name, estimated in candidates.items():
        finite = np.isfinite(estimated).all(axis=1)
        errors = np.linalg.norm(estimated[finite] - world[test][finite], axis=1)
        median = float(np.median(errors)) if len(errors) else None
        p95 = float(np.percentile(errors, 95)) if len(errors) else None
        validation[name] = {
            "tested_points": len(test),
            "mapped_points": int(finite.sum()),
            "median_error_m": median,
            "p95_error_m": p95,
            "passed": bool(
                finite.sum() >= 4 and finite.mean() >= 0.6 and median <= 1.5 and p95 <= 3.0
            ),
        }
    selected = next(
        (
            name
            for name in ("homography", "rational_2", "rational_3", "piecewise_affine")
            if validation[name]["passed"]
        ),
        "homography",
    )
    final_h, _ = cv2.findHomography(pixels, world, cv2.RANSAC, 1.0)
    payload = {
        "selected_method": selected,
        "validation_passed": validation[selected]["passed"],
        "validation": validation,
        "validation_seed": seed,
        "validation_indices": test.tolist(),
        "pixel_points": pixels.tolist(),
        "world_points": world.tolist(),
        "homography": final_h.tolist(),
        "pixel_scale": scale.tolist(),
        "rational_coefficients": _fit_rational(
            pixels,
            world,
            int(selected.split("_")[1]) if selected.startswith("rational_") else 2,
            scale,
        ).tolist(),
        "coordinate_convention": "corner_origin_x_0_105_y_0_68_meters",
        "assumptions": [
            "Fixed camera; validate keypoint overlay against source footage.",
            "Mappings operate on the pixel space of the supplied field keypoints.",
            "Ground anchors only; this is not 3D body reconstruction.",
        ],
    }
    write_json(output, payload)
    return payload
