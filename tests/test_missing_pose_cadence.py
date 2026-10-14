"""Short confidence holes may support period estimation, but never fabricate joints."""

import numpy as np

from src.metrics import cadence_events


def test_sparse_confidence_holes_use_only_observed_events():
    t = np.arange(0, 12, 0.04)
    signal = 0.15 * np.sin(2 * np.pi * 1.5 * t)
    signal[11::30] = np.nan
    original = signal.copy()
    events, report = cadence_events(signal, t)
    assert report["status"] == "accepted"
    assert report["support_method"] == "lomb_scargle_observed_samples"
    assert 1.4 < report["strides_per_second"] < 1.6
    assert np.isfinite(signal[events]).all()
    np.testing.assert_array_equal(np.isnan(signal), np.isnan(original))


def test_long_pose_loss_is_not_interpolated_into_stride_cycles():
    t = np.arange(0, 12, 0.04)
    signal = 0.15 * np.sin(2 * np.pi * 1.5 * t)
    signal[20:28] = np.nan
    events, report = cadence_events(signal, t)
    assert report["status"] == "pose_gap_too_long"
    assert len(events) == 0
