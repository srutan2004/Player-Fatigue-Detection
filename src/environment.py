"""Read-only system checks and real CUDA/pretrained-network forward-pass tests."""

import importlib.metadata
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from src.common import ROOT, configure_ultralytics, provenance, sha256_file, write_json


def check_environment(output: Path, smoke_models: bool = False, device: str = "cuda:0") -> dict:
    """Test actual tensor execution; a driver listing alone does not prove CUDA works.

    Black-image model passes are installation tests only: their successful execution
    does not validate player detection, pose quality, or football measurement.
    """
    configure_ultralytics()
    report = {
        **provenance(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_logical": psutil.cpu_count(),
        "cpu_physical": psutil.cpu_count(logical=False),
        "ram_total_gib": psutil.virtual_memory().total / 1024**3,
        "disk_free_gib": shutil.disk_usage(ROOT).free / 1024**3,
        "packages": {},
        "cuda": {},
        "model_smoke": {},
        "errors": [],
    }
    for name in (
        "torch",
        "torchvision",
        "ultralytics",
        "supervision",
        "av",
        "numpy",
        "opencv-python",
        "pandas",
        "pyarrow",
        "huggingface-hub",
    ):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
            report["errors"].append(f"Missing dependency: {name}")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        report["nvidia_smi"] = {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = {"error": str(exc)}
    try:
        import torch
        from torchvision.ops import nms

        report["cuda"] = {"available": torch.cuda.is_available(), "build": torch.version.cuda}
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available; install the CUDA PyTorch build")
        tensor = torch.ones((256, 256), device=device)
        product = tensor @ tensor
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            properties = torch.cuda.get_device_properties(device)
            report["cuda"].update(
                {
                    "name": properties.name,
                    "vram_mib": properties.total_memory / 1024**2,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
        if not torch.allclose(product, torch.full_like(product, 256)):
            raise RuntimeError("Tensor operation produced an unexpected result")
        keep = nms(
            torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=device),
            torch.tensor([0.9], device=device),
            0.5,
        )
        report["tensor_test"] = {
            "device": device,
            "matmul_passed": True,
            "torchvision_nms_passed": keep.numel() == 1,
        }
        if smoke_models:
            from ultralytics import YOLO

            model_dir = ROOT / "models"
            model_dir.mkdir(exist_ok=True)
            for name in ("yolov8n.pt", "yolov8s-pose.pt"):
                path = model_dir / name
                started = time.perf_counter()
                model = YOLO(str(path))
                model.predict(
                    np.zeros((640, 640, 3), dtype=np.uint8), device=device, imgsz=640, verbose=False
                )
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                report["model_smoke"][name] = {
                    "passed": True,
                    "load_and_forward_seconds": time.perf_counter() - started,
                    "weight_sha256": sha256_file(path),
                    "input": "black_image_installation_test_only",
                }
                del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = "passed" if not report["errors"] else "failed"
    write_json(output, report)
    return report
