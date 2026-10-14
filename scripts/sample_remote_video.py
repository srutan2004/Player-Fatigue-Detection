"""Get a bounded decoded preview while the full source download continues."""

import time
from pathlib import Path

import av
import cv2
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url

from src.common import write_json

if __name__ == "__main__":
    output = Path("results/phase1/remote_preview")
    output.mkdir(parents=True, exist_ok=True)
    av.logging.set_level(av.logging.PANIC)
    started = time.perf_counter()
    try:
        revision = "325bd4c57ee8e135ab652f3921d274526ada496b"
        filename = "videos/117093/117093_panorama_1st_half.mp4"
        metadata = get_hf_file_metadata(
            hf_hub_url(
                "atomscott/soccertrack-v2", filename, revision=revision, repo_type="dataset"
            ),
            token=get_token(),
        )
        with av.open(metadata.location, options={"rw_timeout": "120000000"}) as container:
            stream = container.streams.video[0]
            origin = float((stream.start_time or 0) * stream.time_base)
            report = {
                "revision": revision,
                "filename": filename,
                "width": stream.width,
                "height": stream.height,
                "fps": float(stream.average_rate),
                "frames": [],
            }
            for target in (300, 315, 330, 345):
                container.seek(int((target + origin) / stream.time_base), stream=stream)
                for frame in container.decode(stream):
                    absolute = float(frame.pts * frame.time_base)
                    seconds = absolute - origin
                    if seconds >= target:
                        path = output / f"source_{seconds:.2f}.jpg"
                        cv2.imwrite(str(path), frame.to_ndarray(format="bgr24"))
                        report["frames"].append(
                            {
                                "source_seconds": seconds,
                                "absolute_pts_seconds": absolute,
                                "path": str(path),
                            }
                        )
                        print(f"Preview frame decoded at source {seconds:.2f}s", flush=True)
                        break
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "report.json", report)
    except Exception as exc:
        # Signed CDN URLs must not appear in exception tracebacks/logs.
        write_json(output / "report.json", {"status": "failed", "error_type": type(exc).__name__})
        print(f"Remote preview failed: {type(exc).__name__}")
        raise SystemExit(1)
