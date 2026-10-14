"""Known-kinematics controls guard against plausible but false movement results."""

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.calibrate import GroundMapping, fit_ground_mapping
from src.metrics import cadence_events, knee_flexion, movement_metrics
from src.reference import build_reference_index, reference_people


def test_constant_velocity_and_gap_do_not_create_a_speed_spike():
    t = np.r_[np.arange(0, 2, 0.04), np.arange(4, 6, 0.04)]
    x = 3 * t
    x[50:] += 100  # A distant reappearance must not become sprint distance.
    frames = pd.DataFrame(
        {
            "timestamp": t,
            "world_x": x,
            "world_y": 0.0,
            "clip_name": "A",
            "player_id": "p",
            "position_valid": True,
        }
    )
    result = movement_metrics(frames)
    assert result.speed.dropna().median() == pytest.approx(3.0, abs=0.01)
    assert np.isnan(result.loc[50, "distance_increment"])
    assert result.valid_dt.sum() < 4.0
    assert result.distance_increment.sum() < 12.0


def test_same_leg_cadence_is_not_doubled_into_step_frequency():
    t = np.arange(0, 10, 0.04)
    signal = 0.1 * np.sin(2 * np.pi * 1.5 * t)
    peaks, assessment = cadence_events(signal, t)
    assert assessment["status"] == "accepted"
    assert 1.4 < assessment["strides_per_second"] < 1.6
    assert 12 <= len(peaks) <= 16
    _, rejected = cadence_events(signal[::5], t[::5])
    assert rejected["status"] == "insufficient_or_irregular_sample_rate"


def test_knee_angle_and_missing_anatomy():
    angle = knee_flexion(
        np.array([[0.0, 1.0], [0.0, 1.0], [np.nan, 1.0]]),
        np.zeros((3, 2)),
        np.array([[0.0, -1.0], [1.0, 0.0], [1.0, 0.0]]),
    )
    assert angle[:2] == pytest.approx([0.0, 90.0])
    assert np.isnan(angle[2])


def test_field_geometry_is_validated_on_withheld_correspondences(tmp_path):
    path = tmp_path / "field.json"
    points = {
        f"({x},{y})": [10 * x + 100, 6 * y + 40]
        for x in (0, 20, 40, 60, 80, 105)
        for y in (0, 20, 40, 68)
    }
    path.write_text(json.dumps(points))
    output = tmp_path / "mapping.json"
    report = fit_ground_mapping(path, output)
    assert report["validation_passed"]
    assert report["selected_method"] == "homography"
    np.testing.assert_allclose(
        GroundMapping.load(output).transform([[400, 220]]), [[30, 30]], atol=1e-4
    )
    invalid = GroundMapping.load(output).transform([[4000, 220], [np.nan, 220]])
    assert np.isnan(invalid).all()


def test_shipped_image_id_and_box_dict_map_to_decoded_frame_zero(tmp_path):
    source = tmp_path / "reference.json"
    source.write_text(
        json.dumps(
            {
                "images": [{"image_id": "3903000001", "file_name": "000001.jpg"}],
                "annotations": [
                    {
                        "image_id": "3903000001",
                        "track_id": "7",
                        "attributes": {"player_id": "p8", "role": "player"},
                        "bbox_image": {"x": 10, "y": 20, "w": 30, "h": 80},
                    }
                ],
            }
        )
    )
    output = tmp_path / "reference.sqlite"
    build_reference_index(source, output)
    with sqlite3.connect(output) as connection:
        people = reference_people(connection, 0)
    assert len(people) == 1
    assert people[0]["player"] == "p8"
    assert people[0]["x2"] == 40
