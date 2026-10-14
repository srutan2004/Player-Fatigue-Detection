"""Ambiguous teammates and missing references must not acquire invented identities."""

import numpy as np

from src.detect_track import _reference_links, _shirt_color, _visible_tracklet_continuity


class Mapping:
    def transform(self, points):
        return np.asarray(points, dtype=float)


def test_shadowed_white_kit_is_not_rejected_as_blue():
    for bgr in ([110, 110, 110], [150, 110, 95]):
        image = np.full((100, 60, 3), bgr, dtype=np.uint8)
        assert _shirt_color(image, [0, 0, 60, 100]) == "white"
    image = np.full((100, 60, 3), [200, 60, 20], dtype=np.uint8)
    assert _shirt_color(image, [0, 0, 60, 100]) == "blue"
    image[:, :30] = [180, 180, 180]
    assert _shirt_color(image, [0, 0, 60, 100]) == "unknown"


def reference(player, x):
    return {
        "player": player,
        "pitch_x": x - 52.5,
        "pitch_y": 10 - 34,
        "team": "left",
        "x1": x - 1,
        "y1": 0,
        "x2": x + 1,
        "y2": 10,
    }


def test_global_association_does_not_claim_an_absent_player():
    boxes = {1: np.array([9, 0, 11, 10]), 2: np.array([19, 0, 21, 10])}
    links = _reference_links(
        boxes, [reference("near", 10), reference("absent", 50)], Mapping(), None, {}, {}
    )
    assert links["near"][0] == 1
    assert "absent" not in links


def test_ambiguous_overlapping_teammates_require_a_previous_trusted_link():
    boxes = {1: np.array([9, 0, 11, 10]), 2: np.array([9, 0, 11, 10])}
    assert _reference_links(boxes, [reference("p", 10)], Mapping(), None, {}, {}) == {}
    links = _reference_links(boxes, [reference("p", 10)], Mapping(), None, {}, {"p": 2})
    assert links["p"][0] == 2


def test_visible_continuity_cannot_fill_a_detection_loss_or_guess_a_new_track():
    previous = {
        "track": 1,
        "box": np.array([9.0, 0, 11, 10]),
        "time": 0.4,
        "last_reference_time": 0.0,
    }
    target = {"reference_player_id": "p"}
    refs = [reference("p", 10.2)]
    current = {1: np.array([9.16, 0, 11.16, 10])}
    arguments = (refs, target, {}, Mapping(), None, {}, 0.48, 0.8)
    assert _visible_tracklet_continuity(previous, current, *arguments) == 1
    assert _visible_tracklet_continuity(previous, {}, *arguments) is None
    assert _visible_tracklet_continuity(previous, {2: current[1]}, *arguments) is None
    assert (
        _visible_tracklet_continuity(
            previous, current, refs, target, {}, Mapping(), None, {}, 0.9, 0.8
        )
        is None
    )
    assert (
        _visible_tracklet_continuity(
            previous, current, refs, target, {"other": (1, 0)}, Mapping(), None, {}, 0.48, 0.8
        )
        is None
    )
