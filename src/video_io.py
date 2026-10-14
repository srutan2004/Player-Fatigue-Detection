"""Stream video with decoded source-frame indices and presentation timestamps."""

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass
class VideoFrame:
    """One sampled image; timestamps remain distinct from match-clock labels."""

    index: int
    seconds: float
    timestamp_source: str
    image: np.ndarray


def probe_video(path: Path) -> dict:
    """Inspect codec metadata; duration/frame counts are estimates until decoded.

    Anatomical resolution must be assessed in source pixels, before inference
    resizing. An apparent 4K source alone does not guarantee usable ankle detail.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("Source contains no video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else container.duration / av.time_base
            if container.duration
            else None
        )
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "codec": stream.codec_context.name,
            "average_fps": float(rate) if rate else None,
            "average_rate_fraction": str(rate) if rate else None,
            "time_base": str(stream.time_base),
            "start_pts": stream.start_time,
            "duration_seconds_metadata": duration,
            "frame_count_metadata": stream.frames or None,
        }


def iter_video_frames(
    path: Path,
    start: float,
    duration: float,
    sample_fps: float,
    required_indices: set[int] | None = None,
):
    """Yield source frames sequentially, preserving exact decoded frame indices.

    Presentation time is relative to the stream start. Missing PTS use an explicit
    metadata-rate fallback. No pixels or time intervals are interpolated. Decoding
    from the beginning trades seeking speed for trustworthy annotation alignment.
    """
    if start < 0 or duration <= 0 or sample_fps <= 0:
        raise ValueError("Require start >= 0, duration > 0 and sample_fps > 0")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else None
        origin = (
            float(stream.start_time * stream.time_base) if stream.start_time is not None else None
        )
        next_time = start
        previous = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None:
                absolute = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = absolute
                timestamp, source = absolute - origin, "presentation_timestamp"
            elif fps:
                timestamp, source = index / fps, "metadata_fps_fallback"
            else:
                raise ValueError("Frame has no PTS and video has no usable frame rate")
            if previous is not None and timestamp <= previous:
                raise ValueError(f"Non-increasing presentation timestamp at source frame {index}")
            previous = timestamp
            if timestamp >= start + duration - 1e-8:
                break
            if timestamp + 1e-8 < next_time:
                continue
            # Advance the sampling grid, without duplicating frames at low source FPS.
            next_time = start + (int((timestamp - start) * sample_fps + 1e-7) + 1) / sample_fps
            if required_indices is None or index in required_indices:
                yield VideoFrame(index, timestamp, source, frame.to_ndarray(format="bgr24"))


def frame_time_fields(frame: VideoFrame, start: float, half: int, offset: float) -> dict:
    """Keep half/clip/match times separate; the match offset is supplied, not guessed."""
    return {
        "source_frame_index": frame.index,
        "video_seconds": frame.seconds,
        "clip_seconds": frame.seconds - start,
        "half": half,
        "match_seconds": offset + frame.seconds,
        "timestamp_source": frame.timestamp_source,
    }
