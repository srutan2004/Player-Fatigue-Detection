"""Regression tests for timestamp alignment and crop geometry, using encoded video."""

from fractions import Fraction

import av
import numpy as np
import pytest

from src.video_io import frame_time_fields, iter_video_frames, probe_video
from src.vision import padded_crop, tile_bounds


@pytest.fixture
def source_video(tmp_path):
    path = tmp_path / "source.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(25):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), index * 5, dtype=np.uint8), format="bgr24"
            )
            frame.pts, frame.time_base = index, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_clip_sampling_preserves_source_indices_and_half_offset(source_video):
    frames = list(iter_video_frames(source_video, 0.2, 0.4, 12.5))
    assert [frame.index for frame in frames] == [5, 7, 9, 11, 13]
    assert [frame.seconds for frame in frames] == pytest.approx([0.2, 0.28, 0.36, 0.44, 0.52])
    times = frame_time_fields(frames[1], 0.2, half=2, offset=2700)
    assert times["match_seconds"] == pytest.approx(2700.28)
    assert times["clip_seconds"] == pytest.approx(0.08)
    assert times["half"] == 2
    assert probe_video(source_video)["width"] == 64


def test_sampling_above_source_rate_never_duplicates_frames(source_video):
    frames = list(iter_video_frames(source_video, 0, 1, 100))
    assert len(frames) == 25
    assert len({frame.index for frame in frames}) == 25


def test_crop_clips_padding_without_shifting_source_origin():
    image = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    crop, bounds = padded_crop(image, [1.0, 2.0, 11.0, 12.0], 0.5)
    assert bounds == (0, 0, 16, 17)
    np.testing.assert_array_equal(crop, image[:17, :16])
    crop, bounds = padded_crop(image, [10.0, 8.0, 20.0, 18.0], 0.1)
    assert bounds == (9, 7, 21, 19)
    np.testing.assert_array_equal(crop[3, 4], image[10, 13])
    with pytest.raises(ValueError):
        padded_crop(image, [10, 10, 5, 15], 0.1)


def test_tiles_cover_native_edges_without_unbounded_padding():
    mask = np.zeros((95, 131), dtype=np.uint8)
    for x1, y1, x2, y2 in tile_bounds(131, 95, 64, 0.15):
        assert 0 <= x1 < x2 <= 131
        assert 0 <= y1 < y2 <= 95
        mask[y1:y2, x1:x2] = 1
    assert mask.min() == 1


def test_nonzero_stream_pts_origin_is_not_a_one_second_annotation_shift(tmp_path):
    path = tmp_path / "offset.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(25):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((48, 64, 3), dtype=np.uint8), format="bgr24"
            )
            frame.pts, frame.time_base = index + 25, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    assert probe_video(path)["start_pts"] > 0
    frames = list(iter_video_frames(path, 0.2, 0.2, 25))
    assert [frame.index for frame in frames] == [5, 6, 7, 8, 9]
    assert frames[0].seconds == pytest.approx(0.2)
