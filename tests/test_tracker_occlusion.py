"""Check the effective tracker horizon at the actual low detector frame rate."""

import numpy as np
import supervision as sv


def test_tracker_preserves_identity_after_half_second_occlusion_at_low_detector_rate():
    tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, frame_rate=12.5)
    initial = None
    for frame in range(13):
        if 5 <= frame < 12:
            detection = sv.Detections.empty()
        else:
            detection = sv.Detections(
                xyxy=np.array([[100 + frame * 2, 50, 160 + frame * 2, 200]], dtype=float),
                confidence=np.array([0.9]),
                class_id=np.array([0]),
            )
        found = tracker.update_with_detections(detection)
        if frame == 4:
            initial = int(found.tracker_id[0])
        if frame == 12:
            assert int(found.tracker_id[0]) == initial
