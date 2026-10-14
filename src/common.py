"""Small reproducibility helpers shared by intake and benchmarking."""

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    """Atomically save a strict JSON report so an interrupted run is distinguishable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", "utf-8")
    temp.replace(path)


def sha256_file(path: Path) -> str:
    """Hash compressed source/weight bytes without loading a match into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance() -> dict:
    """Record the code revision and time; uncommitted changes remain explicit."""

    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
    }


def load_config(path: str | Path) -> dict:
    """Validate pilot settings before expensive video/model work begins.

    Sampling here is a feasibility audit, not the native-rate gait extraction
    planned for Phase 3. Paths resolve against the declared project root.
    """
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a YAML mapping")
    base = (config_path.parent / value.get("project_root", "..")).resolve()
    for section in ("video", "pilot", "dataset"):
        if not isinstance(value.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    pilot = value["pilot"]
    for key in ("duration_seconds", "sample_fps", "imgsz", "max_audit_crops"):
        if float(pilot[key]) <= 0:
            raise ValueError(f"pilot.{key} must be positive")
    if int(pilot["max_audit_crops"]) > 100:
        raise ValueError("Phase 1 audit is bounded to at most 100 crops")
    if float(pilot.get("start_seconds", 0)) < 0:
        raise ValueError("pilot.start_seconds must not be negative")
    if int(value["video"]["half"]) not in (1, 2):
        raise ValueError("video.half must be 1 or 2")
    for key in ("detection_confidence", "joint_confidence"):
        if not 0 <= float(pilot[key]) <= 1:
            raise ValueError(f"pilot.{key} must be in [0, 1]")
    value["project_root"] = base
    value["config_path"] = config_path
    for key in ("path", "annotations", "calibration"):
        if value["video"].get(key):
            value["video"][key] = (base / value["video"][key]).resolve()
    value["output_dir"] = (base / value.get("output_dir", "results/phase1/pilot")).resolve()
    value["models_dir"] = (base / value.get("models_dir", "models")).resolve()
    if "pipeline" in value:
        for key in ("reference_index", "ground_mapping", "cached_tracks", "pose_output_dir"):
            if value["pipeline"].get(key):
                value["pipeline"][key] = (base / value["pipeline"][key]).resolve()
    return value


def configure_ultralytics() -> None:
    """Keep model settings local and prevent automatic dependency installation."""
    directory = ROOT / ".cache" / "ultralytics"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "Ultralytics").mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(directory))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
