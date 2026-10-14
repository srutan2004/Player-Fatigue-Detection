"""A bounded, independent broadcast-domain audit; short clips cannot reveal fatigue."""

import concurrent.futures
import io
import json
import struct
import time
import zipfile
import zlib
from pathlib import Path

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url

from src.common import ROOT, provenance, sha256_file, write_json

REPO = "SoccerNet/SN-GSR-2024"
REVISION = "3cc710eb6d53a23350a3d2863311c1c0a2e645d7"


class RemoteZip(io.RawIOBase):
    """Read validated byte ranges of a pinned public archive without downloading 8.85 GB."""

    def __init__(self):
        self.meta = get_hf_file_metadata(
            hf_hub_url(REPO, "test.zip", repo_type="dataset", revision=REVISION)
        )
        self.position = 0
        self.total = 0

    def readable(self):
        """Expose the ZIP central directory using standard seekable IO semantics."""
        return True

    def seekable(self):
        """ZIP members are independently compressed and safe to select by stored offset."""
        return True

    def tell(self):
        """Return compressed archive offset, unrelated to video time."""
        return self.position

    def seek(self, offset, whence=0):
        """Move inside the compressed file; reject invalid offsets."""
        self.position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.meta.size + offset
        )
        if not 0 <= self.position <= self.meta.size:
            raise ValueError("Offset outside archive")
        return self.position

    def fetch(self, start, length):
        """Validate status, exact Content-Range and byte length; credentials never enter reports."""
        end = start + length - 1
        response = requests.get(
            self.meta.location, headers={"Range": f"bytes={start}-{end}"}, timeout=90
        )
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.meta.size}"
            or len(response.content) != length
        ):
            raise RuntimeError(f"Remote archive range mismatch ({response.status_code})")
        return response.content

    def read(self, n=-1):
        """Bound central-directory network reads independently of selected members."""
        n = min(self.meta.size - self.position, n if n >= 0 else self.meta.size)
        if not n:
            return b""
        if self.total + n > 20 * 1024**2:
            raise RuntimeError("Central directory exceeds 20 MiB budget")
        content = self.fetch(self.position, n)
        self.position += n
        self.total += n
        return content


def download_samples(destination: Path = ROOT / "data/processed/broadcast_audit") -> dict:
    """Select first five sorted test clips before viewing results; sample about 2 fps.

    This is a detection/pose/camera stress audit, not native-rate gait evaluation or
    new training. Exact member CRC and local SHA256 are checked; full ZIP not copied.
    """
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if all(
            (destination / r["path"]).exists()
            and sha256_file(destination / r["path"]) == r["sha256"]
            for r in manifest["files"]
        ):
            return manifest
    remote = RemoteZip()
    with zipfile.ZipFile(remote) as archive:
        infos = {i.filename: i for i in archive.infolist()}
    clips = sorted(n.split("/")[0] for n in infos if n.endswith("/Labels-GameState.json"))[:5]
    names = []
    for clip in clips:
        names.append(f"{clip}/Labels-GameState.json")
        native = sorted(n for n in infos if n.startswith(f"{clip}/img1/") and n.endswith(".jpg"))
        names.extend(native[round(i * 12.5)] for i in range(int((len(native) - 1) / 12.5) + 1))
    network_bound = sum(infos[n].compress_size + 1024 for n in names) + remote.total
    if network_bound > 200 * 1024**2:
        raise RuntimeError("Selected broadcast samples exceed 200 MiB budget")

    def get_member(name):
        """Decode only a selected local member; verify uncompressed size and CRC32."""
        info = infos[name]
        path = destination / name
        if not path.exists():
            # One range includes header, filename, extra field and compressed payload.
            raw = remote.fetch(info.header_offset, info.compress_size + 1024)
            if raw[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP local header")
            filename_length, extra_length = struct.unpack_from("<HH", raw, 26)
            offset = 30 + filename_length + extra_length
            if offset > 1024:
                raise ValueError("ZIP header larger than bounded prefix")
            compressed = raw[offset : offset + info.compress_size]
            content = (
                zlib.decompress(compressed, -15)
                if info.compress_type == zipfile.ZIP_DEFLATED
                else compressed
            )
            if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                raise ValueError("ZIP member checksum failed")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return {
            "path": name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "archive_crc32": info.CRC,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        files = list(pool.map(get_member, names))
    report = {
        **provenance(),
        "repo": REPO,
        "revision": REVISION,
        "archive": "test.zip",
        "archive_etag": remote.meta.etag,
        "archive_bytes": remote.meta.size,
        "network_upper_bound_bytes": network_bound,
        "clips": clips,
        "files": files,
        "sampling": "Every ~12.5 native frame indices, approximately 2 fps; not gait samples",
        "status": "downloaded",
    }
    write_json(manifest_path, report)
    return report


def audit_broadcast(
    destination: Path = ROOT / "data/processed/broadcast_audit",
    output: Path = ROOT / "results/phase4/broadcast_audit",
) -> dict:
    """Measure detection recall, pose availability and static-map failure on five broadcasts.

    Labeled player/goalkeeper boxes define recall only. Other people and ignore regions
    prevent an all-person precision claim. Pose confidence is availability, not joint
    accuracy. ~2 fps samples are insufficient for cadence or a tracker benchmark.
    """
    import cv2
    import numpy as np
    import pandas as pd
    from scipy.optimize import linear_sum_assignment

    from src.quality import CameraGuard
    from src.vision import box_iou, detect_people, load_model, padded_crop, pose_crop

    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    manifest = download_samples(destination)
    detector = load_model(ROOT / "models", "yolov8n.pt")
    pose_model = load_model(ROOT / "models", "yolov8m-pose.pt")
    settings = {
        "detection_confidence": 0.15,
        "imgsz": 1920,
        "device": "cuda:0",
        "joint_confidence": 0.5,
    }
    frame_rows, clip_rows, pose_rows = [], [], []
    for clip in manifest["clips"]:
        labels = json.loads((destination / clip / "Labels-GameState.json").read_text("utf-8"))
        images = {r["file_name"]: r for r in labels["images"]}
        annotations = {}
        for annotation in labels["annotations"]:
            if (
                annotation.get("supercategory") == "object"
                and annotation.get("attributes", {}).get("role") in ("player", "goalkeeper")
                and annotation.get("bbox_image")
            ):
                annotations.setdefault(annotation["image_id"], []).append(annotation)
        guard = CameraGuard()
        sheets = []
        paths = sorted((destination / clip / "img1").glob("*.jpg"))
        for frame_number, path in enumerate(paths):
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Downloaded JPEG does not decode")
            detections = detect_people(detector, image, settings)
            boxes = [d["bbox_xyxy"] for d in detections]
            camera = guard.check(image, boxes)
            meta = images[path.name]
            gt = annotations.get(meta["image_id"], []) if meta.get("has_labeled_person") else []
            gt_boxes = [
                [
                    a["bbox_image"]["x"],
                    a["bbox_image"]["y"],
                    a["bbox_image"]["x"] + a["bbox_image"]["w"],
                    a["bbox_image"]["y"] + a["bbox_image"]["h"],
                ]
                for a in gt
            ]
            overlaps = np.array([[box_iou(a, b) for b in boxes] for a in gt_boxes])
            matches = []
            if len(gt_boxes) and len(boxes):
                gi, di = linear_sum_assignment(1 - overlaps)
                matches = [
                    (int(g), int(d)) for g, d in zip(gi, di, strict=True) if overlaps[g, d] >= 0.5
                ]
            matched_gt = {g for g, _ in matches}
            frame_rows.append(
                {
                    "clip": clip,
                    "source_frame": int(path.stem) - 1,
                    "source_seconds_assuming_25fps": (int(path.stem) - 1) / 25,
                    "labeled_players": len(gt),
                    "matched_players": len(matches),
                    "detections": len(boxes),
                    "camera_status": camera["status"],
                    "static_mapping_valid": camera["mapping_valid"],
                    "registration_displacement_px": camera.get(
                        "displacement_registration_pixels", np.nan
                    ),
                    "small_players_lt40px": sum(b[3] - b[1] < 40 for b in gt_boxes),
                    "small_matched": sum(gt_boxes[g][3] - gt_boxes[g][1] < 40 for g in matched_gt),
                }
            )
            if frame_number % 6 == 0:
                chosen = sorted(
                    matches, key=lambda p: gt_boxes[p[0]][3] - gt_boxes[p[0]][1], reverse=True
                )[:2]
                overlay = image.copy()
                for g, d in chosen:
                    box = boxes[d]
                    crop, bounds = padded_crop(image, box, 0.1)
                    pose = pose_crop(pose_model, crop, bounds, box, settings)
                    pose_rows.append(
                        {
                            "clip": clip,
                            "source_frame": int(path.stem) - 1,
                            "track_id": gt[g]["track_id"],
                            "body_height_source_px": box[3] - box[1],
                            "six_joint_confident": pose["gait_joints_confident"],
                            "pose_status": pose["status"],
                        }
                    )
                    cv2.rectangle(
                        overlay,
                        tuple(np.round(box[:2]).astype(int)),
                        tuple(np.round(box[2:]).astype(int)),
                        (40, 230, 80),
                        2,
                    )
                    for joint in pose["joints"]:
                        if joint["valid"]:
                            cv2.circle(
                                overlay,
                                (round(joint["x"]), round(joint["y"])),
                                3,
                                (20, 70, 255),
                                -1,
                            )
                cv2.putText(
                    overlay,
                    f"{clip} frame {int(path.stem) - 1} | static map: {camera['mapping_valid']}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                sheets.append(cv2.resize(overlay, (640, 360)))
        rows = [r for r in frame_rows if r["clip"] == clip]
        total = sum(r["labeled_players"] for r in rows)
        poses = [r for r in pose_rows if r["clip"] == clip]
        clip_rows.append(
            {
                "clip": clip,
                "sampled_frames": len(rows),
                "annotated_player_boxes": total,
                "player_recall_iou50": sum(r["matched_players"] for r in rows) / total
                if total
                else None,
                "static_map_valid_fraction": np.mean([r["static_mapping_valid"] for r in rows]),
                "pose_crops": len(poses),
                "six_joint_confident_fraction": np.mean([r["six_joint_confident"] for r in poses])
                if poses
                else None,
            }
        )
        if sheets:
            while len(sheets) % 2:
                sheets.append(np.zeros_like(sheets[0]))
            cv2.imwrite(
                str(output / f"{clip}_audit.jpg"),
                np.vstack([np.hstack(sheets[i : i + 2]) for i in range(0, len(sheets), 2)]),
            )
        print(f"Completed broadcast audit {clip}", flush=True)
    pd.DataFrame(frame_rows).to_csv(output / "frames.csv", index=False)
    pd.DataFrame(clip_rows).to_csv(output / "clips.csv", index=False)
    pd.DataFrame(pose_rows).to_csv(output / "poses.csv", index=False)
    report = {
        **provenance(),
        "status": "completed",
        "dataset": REPO,
        "revision": REVISION,
        "source_manifest": str(destination / "manifest.json"),
        "clips": clip_rows,
        "sampled_frames": len(frame_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "purpose": "separate short broadcast stress audit, no longitudinal fatigue inference",
        "limitations": (
            "Sampled detection recall only, no all-person precision/IDF1/HOTA/joint ground truth. "
            "25 fps time assumption matches released 750-frame/30-second clips; "
            "no cadence at sampled rate."
        ),
        "static_mapping": (
            "CameraGuard tests hypothetical fixed first-frame mapping; "
            "no broadcast world calibration is fitted or used."
        ),
        "pose_selection": (
            "Two tallest matched players every sixth sampled frame, "
            "fixed before confidence measurement"
        ),
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(audit_broadcast(), indent=2))
