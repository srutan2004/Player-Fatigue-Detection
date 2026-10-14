"""Independent synthetic controls for camera drift, valid gait and missing mechanics."""

import cv2
import numpy as np
import pandas as pd
import pytest

from src.metrics import gait_metrics, movement_metrics, straight_running_mask, window_metrics
from src.quality import CameraGuard


def test_camera_translation_invalidates_fixed_mapping_after_persistence():
    image = np.random.default_rng(42).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    guard = CameraGuard()
    assert guard.check(image)["mapping_valid"]
    assert guard.check(image)["mapping_valid"]
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 6], [0, 1, 0]]), (640, 360))
    for _ in range(3):
        report = guard.check(shifted)
    assert report["status"] == "camera_motion"
    assert not report["mapping_valid"]


def test_unobservable_camera_registration_cannot_certify_a_ground_mapping():
    guard = CameraGuard()
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(2):
        report = guard.check(blank)
        assert report["status"] == "registration_unavailable"
        assert not report["mapping_valid"]


def test_native_rate_mechanics_and_window_measurement_have_known_units():
    t = np.arange(0, 40, 0.04)
    phase = 2 * np.pi * 1.5 * t
    data = pd.DataFrame(
        {
            "clip_name": "control",
            "player_id": "p",
            "timestamp": t,
            "world_x": 3 * t,
            "world_y": 0.0,
            "position_valid": True,
            "box_height": 100.0,
            "ground_pixel_y": 170.0,
        }
    )
    for leg, sign in (("left", 1), ("right", -1)):
        data[f"{leg}_hip_x"] = 100 + 10 * t
        data[f"{leg}_hip_y"] = 100 + 2 * np.sin(phase)
        data[f"{leg}_knee_x"] = 100 + 10 * t + sign * 8 * np.sin(phase)
        data[f"{leg}_knee_y"] = 135.0
        data[f"{leg}_ankle_x"] = 100 + 10 * t + sign * 15 * np.sin(phase)
        data[f"{leg}_ankle_y"] = 170.0
    frames = movement_metrics(data)
    bouts, cycles = gait_metrics(frames)
    assert len(bouts) == 1
    assert len(cycles) > 50
    assert cycles.stride_frequency.median() == pytest.approx(1.5, abs=0.1)
    assert cycles.stride_length.median() == pytest.approx(2.0, abs=0.15)
    windows = window_metrics(frames, bouts, cycles, duration=30, step=30)
    assert windows.iloc[0].gait_eligible
    assert windows.iloc[0].avg_speed == pytest.approx(3.0, abs=0.02)
    assert windows.iloc[0].vertical_oscillation_unit == "normalized_image_proxy"


def test_unavailable_pose_produces_empty_typed_tables_and_missing_gait():
    t = np.arange(0, 130, 0.04)
    data = pd.DataFrame(
        {
            "clip_name": "control",
            "player_id": "p",
            "timestamp": t,
            "world_x": 3 * t,
            "world_y": 0.0,
            "position_valid": True,
        }
    )
    for joint in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"):
        data[f"{joint}_x"] = np.nan
        data[f"{joint}_y"] = np.nan
    frames = movement_metrics(data)
    bouts, cycles = gait_metrics(frames)
    assert bouts.empty and "bout_id" in bouts.columns
    assert cycles.empty and "stride_length" in cycles.columns
    windows = window_metrics(frames, bouts, cycles)
    assert not windows.gait_eligible.any()
    assert windows.stride_length.isna().all()
    assert windows.avg_speed.notna().all()


def test_gradual_turn_is_rejected_despite_small_per_frame_heading_changes():
    t = np.arange(0, 10, 0.04)
    data = pd.DataFrame(
        {"timestamp": t, "speed": 3.0, "movement_valid": True, "heading": np.deg2rad(25 * t)}
    )
    mask = straight_running_mask(data)
    assert not mask[(t > 2) & (t < 8)].any()
    data["heading"] = 3.13 + 0.01 * np.sin(t)  # Angle wrapping must not resemble a turn.
    data.loc[data.heading > np.pi, "heading"] -= 2 * np.pi
    assert straight_running_mask(data).all()
