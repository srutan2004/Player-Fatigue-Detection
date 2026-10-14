"""Verify display thinning cannot hide a native position loss or identity transition."""

import numpy as np
import pandas as pd

from src.dashboard import observed_trajectories, records


def test_map_sampling_preserves_native_gaps_and_identity_switches(tmp_path):
    folder = tmp_path / "example_first"
    folder.mkdir()
    frames = pd.DataFrame(
        {
            "player_id": ["p"] * 100,
            "timestamp": np.arange(100) / 25,
            "world_x": np.arange(100) / 25,
            "world_y": [34.0] * 100,
            "movement_valid": [True] * 100,
            "identity_segment": [1] * 100,
            "speed": [1.0] * 100,
        }
    )
    frames.loc[12, "movement_valid"] = False  # Loss between 1-second display samples.
    frames.loc[40:, "identity_segment"] = 2  # Switch also falls between display samples.
    frames.to_parquet(folder / "frame_metrics.parquet", index=False)
    points = observed_trajectories(tmp_path, ["example_first"])[0]["points"]
    assert [p["x"] for p in points] == [0.0, None, 1.0, None, 2.0, 3.0]


def test_exported_missing_measurement_is_null_not_zero():
    rows = records(pd.DataFrame({"speed": [np.nan, 2.0]}))
    assert rows == [{"speed": None}, {"speed": 2.0}]
