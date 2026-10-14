"""Background registration guards against camera motion masquerading as player speed."""

import cv2
import numpy as np


class CameraGuard:
    """Compare pitch/background structure with the calibration view, excluding people."""

    def __init__(self, motion_threshold_pixels: float = 2.0):
        self.reference = None
        self.points = None
        self.shape = None
        self.threshold = motion_threshold_pixels
        self.consecutive_motion = 0
        self.valid = True

    def check(self, image, person_boxes=()) -> dict:
        """Flag persistent pan/zoom/cuts; unavailable registration is reported as unverified.

        This guard invalidates the fixed camera map; it does not claim to recalibrate
        moving broadcast shots. Drift is measured in the 640-pixel registration view.
        """
        scale = 640 / image.shape[1]
        gray = cv2.cvtColor(
            cv2.resize(image, (640, round(image.shape[0] * scale))), cv2.COLOR_BGR2GRAY
        )
        if self.reference is None:
            mask = np.full_like(gray, 255)
            mask[: round(gray.shape[0] * 0.15)] = 0
            mask[round(gray.shape[0] * 0.8) :] = 0
            for box in person_boxes:
                x1, y1, x2, y2 = np.round(np.asarray(box) * scale).astype(int)
                cv2.rectangle(mask, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), 0, -1)
            self.reference = gray
            self.shape = gray.shape
            self.points = cv2.goodFeaturesToTrack(gray, 250, 0.01, 5, mask=mask)
            if self.points is None or len(self.points) < 10:
                return {"status": "registration_unavailable", "mapping_valid": False}
            return {"status": "reference_initialized", "mapping_valid": True}
        if self.points is None or len(self.points) < 10:
            return {"status": "registration_unavailable", "mapping_valid": False}
        points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.reference, gray, self.points, None, winSize=(21, 21), maxLevel=3
        )
        if points is None:
            return {"status": "registration_unavailable", "mapping_valid": False}
        selected = status.reshape(-1).astype(bool)
        selected &= np.isfinite(points).all(axis=(1, 2))
        for box in person_boxes:
            x1, y1, x2, y2 = np.asarray(box) * scale
            target = points[:, 0]
            selected &= ~(
                (target[:, 0] >= x1 - 3)
                & (target[:, 0] <= x2 + 3)
                & (target[:, 1] >= y1 - 3)
                & (target[:, 1] <= y2 + 3)
            )
        if selected.sum() < 10:
            return {"status": "registration_unavailable", "mapping_valid": False}
        affine, inliers = cv2.estimateAffinePartial2D(
            self.points[selected], points[selected], method=cv2.RANSAC, ransacReprojThreshold=1.0
        )
        if affine is None or inliers.mean() < 0.6:
            return {"status": "registration_unavailable", "mapping_valid": False}
        h, w = gray.shape
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])
        displacement = float(np.max(np.linalg.norm(corners @ affine.T - corners[:, :2], axis=1)))
        self.consecutive_motion = (
            self.consecutive_motion + 1 if displacement > self.threshold else 0
        )
        if self.consecutive_motion >= 3:
            self.valid = False
        return {
            "status": "camera_motion" if self.consecutive_motion >= 3 else "registered",
            "displacement_registration_pixels": displacement,
            "inlier_fraction": float(inliers.mean()),
            "mapping_valid": self.valid,
        }
