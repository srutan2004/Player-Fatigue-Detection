"""Resume the locked 100-minute study without overwriting completed vision stages."""

import json
import time
from pathlib import Path

import pandas as pd
import psutil

from src.common import ROOT, load_config, write_json
from src.detect_track import extract_segment
from src.pose_pass import pose_from_tracks
from src.process_metrics import process_metrics


def run_study(manifest_path: Path = ROOT / "data/manifests/study.json") -> dict:
    """Run tracks, native pose bouts and measurements sequentially on the laptop GPU.

    Identity-assisted trajectories remain independent detected measurements.
    Restrict the final cohort explicitly; supplementary pilot players are retained
    in their original audit folder and cannot silently enter paired comparisons.
    """
    manifest = json.loads(manifest_path.read_text())
    reports = []
    started = time.perf_counter()
    for clip in manifest["clips"]:
        config = load_config(ROOT / clip["config"])
        track_dir = config["output_dir"]
        report_path = track_dir / "extraction_report.json"
        # Wait for a stage explicitly launched in this same workflow to finish.
        while (track_dir / "extraction_status.json").exists() and not report_path.exists():
            active = json.loads((track_dir / "extraction_status.json").read_text())
            status = active["status"]
            if status != "running":
                raise RuntimeError(f"{clip['clip_name']} extraction is {status}; inspect its log")
            pid = active.get("pid")
            if not pid or not psutil.pid_exists(pid):
                raise RuntimeError(
                    "Partial extraction has no active owner; inspect it and choose a new folder"
                )
            try:
                process = psutil.Process(pid)
                if process.create_time() != active.get("process_create_time"):
                    raise RuntimeError("The saved extraction owner no longer exists")
            except psutil.NoSuchProcess as exc:
                raise RuntimeError("Extraction ended without a completion report") from exc
            print(f"Waiting for active extraction: {clip['clip_name']}", flush=True)
            time.sleep(10)
        if not report_path.exists():
            extract_segment(config)
        else:
            existing = json.loads(report_path.read_text())
            actual = existing["config"]
            for field in ("start_seconds", "duration_seconds", "detector", "imgsz"):
                if actual["pilot"][field] != config["pilot"][field]:
                    raise ValueError(f"Cached {clip['clip_name']} has different {field}")
            if Path(actual["video"]["path"]).resolve() != config["video"]["path"]:
                raise ValueError("Cached source differs")
        pose_dir = ROOT / config["pipeline"].get(
            "pose_output_dir", f"results/phase3/study/{clip['clip_name']}"
        )
        pose_report = pose_dir / "pose_report.json"
        selected = pose_dir / "selected_tracks.csv"
        if not pose_report.exists():
            pose_dir.mkdir(parents=True, exist_ok=True)
            tracks = pd.read_csv(track_dir / "keypoints_positions.csv")
            players = [target["player_id"] for target in clip["targets"]]
            tracks[tracks.player_id.isin(players)].to_csv(selected, index=False)
            pose_from_tracks(config, selected, pose_dir)
        metric_dir = ROOT / "results/metrics_csv" / clip["clip_name"]
        metrics = process_metrics(pose_dir / "keypoints_positions.csv", metric_dir)
        reports.append({"clip": clip["clip_name"], "metrics": metrics})
        write_json(
            ROOT / "results/phase3/study_status.json",
            {"status": "running", "completed_clips": len(reports), "clips": reports},
        )
        print(f"Completed {clip['clip_name']} measurements", flush=True)
    summary = {
        "status": "completed",
        "completed_clips": len(reports),
        "clips": reports,
        "elapsed_driver_seconds": time.perf_counter() - started,
    }
    write_json(ROOT / "results/phase3/study_status.json", summary)
    return summary


if __name__ == "__main__":
    run_study()
