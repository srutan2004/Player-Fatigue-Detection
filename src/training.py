"""One small, auditable detector adaptation experiment with hard data/run limits."""

import csv
import json
import math
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.common import ROOT, provenance, sha256_file, write_json
from src.reference import reference_people
from src.splits import validate_source_blocks
from src.video_io import iter_video_frames
from src.vision import box_iou, load_model, tile_bounds


def propose_missing_people(dataset_dir: Path) -> dict:
    """Find missing referee/staff labels; proposals require visual acceptance.

    MOT player annotations omit some real people. Training with those omissions
    would turn visible referees into false negatives for the single person class.
    These model proposals are not automatically promoted to ground truth.
    """
    manifest = json.loads((dataset_dir / "subset_manifest.json").read_text("utf-8"))
    paths = [
        dataset_dir / "images" / row["split"] / f"{row['image']}.jpg" for row in manifest["images"]
    ]
    model = load_model(ROOT / "models", "yolov8n.pt")
    proposals = []
    directory = dataset_dir / "label_proposals"
    directory.mkdir(exist_ok=True)

    def bounded_predictions():
        # Image-path lists are treated as one batch by Ultralytics; slice explicitly.
        for offset in range(0, len(paths), 4):
            yield from model.predict(
                [str(path) for path in paths[offset : offset + 4]],
                stream=False,
                batch=4,
                imgsz=640,
                conf=0.35,
                classes=[0],
                device=0,
                verbose=False,
            )

    results = bounded_predictions()
    for row, path, result in zip(manifest["images"], paths, results, strict=True):
        height, width = result.orig_shape
        labels = dataset_dir / "labels" / row["split"] / f"{row['image']}.txt"
        existing = []
        for line in labels.read_text().splitlines():
            _, x, y, w, h = map(float, line.split())
            existing.append(
                [
                    (x - w / 2) * width,
                    (y - h / 2) * height,
                    (x + w / 2) * width,
                    (y + h / 2) * height,
                ]
            )
        for box, confidence in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=True
        ):
            if any(box_iou(box, old) > 0.15 for old in existing) or box[3] - box[1] < 8:
                continue
            proposal_id = len(proposals)
            image = cv2.imread(str(path))
            x1, y1, x2, y2 = box
            dx, dy = (x2 - x1) * 0.5, (y2 - y1) * 0.3
            crop = image[
                max(0, int(y1 - dy)) : min(height, int(y2 + dy)),
                max(0, int(x1 - dx)) : min(width, int(x2 + dx)),
            ]
            thumb = cv2.resize(crop, (190, 230))
            cv2.putText(
                thumb, str(proposal_id), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
            cv2.imwrite(str(directory / f"proposal_{proposal_id:04d}.jpg"), thumb)
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "image": row["image"],
                    "split": row["split"],
                    "bbox_xyxy": box.tolist(),
                    "confidence": float(confidence),
                    "image_size": [width, height],
                    "review_status": "pending",
                }
            )
    for offset in range(0, len(proposals), 20):
        canvas = np.full((5 * 250, 4 * 200, 3), 230, dtype=np.uint8)
        for position, proposal in enumerate(proposals[offset : offset + 20]):
            image = cv2.imread(str(directory / f"proposal_{proposal['proposal_id']:04d}.jpg"))
            y, x = position // 4 * 250, position % 4 * 200
            canvas[y : y + 230, x : x + 190] = image
        cv2.imwrite(str(directory / f"review_{offset // 20:02d}.jpg"), canvas)
    report = {**provenance(), "proposals": proposals, "status": "awaiting_visual_review"}
    write_json(directory / "proposals.json", report)
    return report


def apply_label_review(dataset_dir: Path, rejected_ids: list[int], review_notes: str) -> dict:
    """Apply visually accepted proposals once and preserve reviewer decisions/provenance."""
    path = dataset_dir / "label_proposals/proposals.json"
    report = json.loads(path.read_text("utf-8"))
    if report["status"] != "awaiting_visual_review":
        raise ValueError("Label review has already been applied")
    accepted = 0
    excluded_images = {p["image"] for p in report["proposals"] if p["proposal_id"] in rejected_ids}
    for proposal in report["proposals"]:
        if proposal["proposal_id"] in rejected_ids:
            proposal["review_status"] = "rejected"
            continue
        if proposal["image"] in excluded_images:
            proposal["review_status"] = "image_quarantined"
            continue
        proposal["review_status"] = "visually_accepted"
        x1, y1, x2, y2 = proposal["bbox_xyxy"]
        width, height = proposal["image_size"]
        label = [
            0,
            (x1 + x2) / 2 / width,
            (y1 + y2) / 2 / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]
        target = dataset_dir / "labels" / proposal["split"] / f"{proposal['image']}.txt"
        with target.open("a", encoding="utf-8") as handle:
            if target.stat().st_size:
                handle.write("\n")
            handle.write(" ".join(f"{value:.6f}" for value in label))
        accepted += 1
    report.update(
        status="review_applied",
        accepted=accepted,
        rejected=len(rejected_ids),
        review_notes=review_notes,
    )
    manifest_path = dataset_dir / "subset_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in manifest["images"]:
        if row["image"] in excluded_images:
            row["review_status"] = "quarantined_uncertain_proposal"
            for folder, suffix in (("images", ".jpg"), ("labels", ".txt")):
                original = dataset_dir / folder / row["split"] / f"{row['image']}{suffix}"
                quarantined = dataset_dir / "quarantine" / folder / row["split"] / original.name
                if not original.resolve().is_relative_to(dataset_dir.resolve()) or not (
                    quarantined.resolve().is_relative_to(dataset_dir.resolve())
                ):
                    raise ValueError("Quarantine move must remain within the generated subset")
                quarantined.parent.mkdir(parents=True, exist_ok=True)
                original.replace(quarantined)
        else:
            counts[row["split"]] += 1
            row["review_status"] = "source_annotation_and_proposal_review_complete"
    manifest.update(
        review_status="reviewed",
        actual_split_counts=counts,
        review_notes=review_notes,
        quarantined_images=len(excluded_images),
    )
    report["actual_split_counts"] = counts
    write_json(manifest_path, manifest)
    write_json(path, report)
    return report


def prepare_training_subset(spec_path: Path, output: Path, seed: int = 42) -> dict:
    """Seed at most 600 native tiles from disjoint annotated source blocks.

    Labels come from the reference image boxes and require visual audit. Splitting
    happens at match/temporal-block level before tiles are produced; every visible
    reference person is retained, including referees. No fatigue labels are created.
    """
    spec = json.loads(spec_path.read_text("utf-8"))
    blocks = spec["blocks"]
    validate_source_blocks(blocks)
    budgets = {"train": 400, "validation": 100, "test": 100}
    rng = np.random.default_rng(seed)
    rows = []
    for split, budget in budgets.items():
        selected = [block for block in blocks if block["split"] == split]
        if len(selected) != 1:
            raise ValueError("The bounded experiment requires one source block per split")
        block = selected[0]
        video, reference = ROOT / block["video"], ROOT / block["reference_index"]
        if not video.exists() or not reference.exists():
            raise FileNotFoundError(f"Missing video/reference for {split}")
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        if list(image_dir.glob("*.jpg")):
            raise ValueError(
                "Training subset already exists; preserve it or choose a new directory"
            )
        duration = block["end_seconds"] - block["start_seconds"]
        frame_count = math.ceil(budget / 4)
        written = 0
        with sqlite3.connect(reference) as connection:
            for frame in iter_video_frames(
                video, block["start_seconds"], duration, frame_count / duration
            ):
                people = reference_people(connection, frame.index)
                h, w = frame.image.shape[:2]
                candidates = list(tile_bounds(w, h, 640, 0.15))
                rng.shuffle(candidates)
                for x1, y1, x2, y2 in candidates:
                    labels = []
                    for person in people:
                        a, b = max(person["x1"], x1), max(person["y1"], y1)
                        c, d = min(person["x2"], x2), min(person["y2"], y2)
                        if c <= a or d <= b:
                            continue
                        # Retain clipped visible people rather than making them false negatives.
                        if c - a < 2 or d - b < 4:
                            continue
                        labels.append(
                            [
                                0,
                                ((a + c) / 2 - x1) / (x2 - x1),
                                ((b + d) / 2 - y1) / (y2 - y1),
                                (c - a) / (x2 - x1),
                                (d - b) / (y2 - y1),
                            ]
                        )
                    # Keep a small deterministic background sample, never all empty tiles.
                    if not labels and rng.random() > 0.1:
                        continue
                    name = f"{block['match_id']}_h{block['half']}_f{frame.index:06d}_x{x1}_y{y1}"
                    crop = frame.image[y1:y2, x1:x2]
                    if not cv2.imwrite(str(image_dir / f"{name}.jpg"), crop):
                        raise OSError("Failed to write training tile")
                    (label_dir / f"{name}.txt").write_text(
                        "\n".join(" ".join(f"{value:.6f}" for value in row) for row in labels),
                        encoding="utf-8",
                    )
                    rows.append(
                        {
                            "image": name,
                            "split": split,
                            "source_frame": frame.index,
                            "match_id": block["match_id"],
                            "half": block["half"],
                            "timestamp": frame.seconds,
                            "tile_bounds": [x1, y1, x2, y2],
                            "people": len(labels),
                            "label_source": "reference_boxes",
                            "review_status": "pending",
                        }
                    )
                    written += 1
                    if written % 4 == 0 or written >= budget:
                        break
                if written >= budget:
                    break
        if written != budget:
            raise ValueError(
                f"Only {written}/{budget} {split} tiles available; inspect source labels"
            )
    size = sum(path.stat().st_size for path in (output / "images").rglob("*.jpg"))
    if size > 1024**3:
        raise ValueError("Training images exceed the agreed 1 GiB hard budget")
    dataset = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/validation",
        "test": "images/test",
        "names": {0: "person"},
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset), encoding="utf-8")
    manifest = {
        **provenance(),
        "source_spec": spec,
        "seed": seed,
        "budgets": budgets,
        "image_bytes": size,
        "review_status": "pending",
        "images": rows,
    }
    write_json(output / "subset_manifest.json", manifest)
    make_training_contact_sheets(output, rows)
    return manifest


def make_training_contact_sheets(output: Path, rows: list[dict]) -> None:
    """Render every seeded label for visual review before the bounded training run."""
    sheets = output / "audit_sheets"
    sheets.mkdir(exist_ok=True)
    for offset in range(0, len(rows), 30):
        canvas = np.full((6 * 340, 5 * 320, 3), 235, dtype=np.uint8)
        for position, row in enumerate(rows[offset : offset + 30]):
            image = cv2.imread(str(output / "images" / row["split"] / f"{row['image']}.jpg"))
            for line in (
                (output / "labels" / row["split"] / f"{row['image']}.txt").read_text().splitlines()
            ):
                _, x, y, w, h = map(float, line.split())
                height, width = image.shape[:2]
                first = (round((x - w / 2) * width), round((y - h / 2) * height))
                last = (round((x + w / 2) * width), round((y + h / 2) * height))
                cv2.rectangle(image, first, last, (0, 255, 0), 2)
            image = cv2.resize(image, (320, 320))
            y0, x0 = (position // 5) * 340, (position % 5) * 320
            canvas[y0 : y0 + 320, x0 : x0 + 320] = image
            cv2.putText(
                canvas,
                f"{offset + position}: {row['split']} f{row['source_frame']}",
                (x0 + 3, y0 + 335),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )
        cv2.imwrite(str(sheets / f"sheet_{offset // 30:02d}.jpg"), canvas)


def train_bounded(dataset_dir: Path, report_dir: Path, epochs: int = 15) -> dict:
    """Run exactly one reviewed YOLOv8n adaptation, capped at 20 epochs/90 minutes.

    Validation selects the detector; an untouched match is tested after selection.
    Reference-box quality limits what detection scores can establish.
    """
    if not 1 <= epochs <= 20:
        raise ValueError("The agreed detector run is limited to 20 epochs")
    manifest = json.loads((dataset_dir / "subset_manifest.json").read_text("utf-8"))
    if manifest["review_status"] != "reviewed":
        raise ValueError("Inspect/correct the tile audit sheets and record review before training")
    report_dir.mkdir(parents=True, exist_ok=True)
    registry = report_dir / "bounded_run.json"
    if registry.exists():
        raise ValueError("The single bounded training run was already started; inspect its report")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The bounded run requires verified CUDA; do not silently train on CPU")
    report = {
        **provenance(),
        "status": "started",
        "epochs_requested": epochs,
        "maximum_epochs": 20,
        "maximum_minutes": 90,
        "subset_manifest_sha256": sha256_file(dataset_dir / "subset_manifest.json"),
    }
    write_json(registry, report)
    started = time.perf_counter()
    model = load_model(ROOT / "models", "yolov8n.pt")
    val_options = {
        "data": str(dataset_dir / "dataset.yaml"),
        "imgsz": 640,
        "batch": 4,
        "device": 0,
        "workers": 0,
        "classes": [0],
        "plots": False,
    }

    def summarize(metrics):
        return {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

    report["pretrained_validation"] = summarize(model.val(**val_options))
    write_json(registry, report)

    def stop_at_budget(trainer):
        if time.perf_counter() - started > 90 * 60:
            trainer.stop = True

    model.add_callback("on_train_epoch_end", stop_at_budget)
    model.train(
        data=str(dataset_dir / "dataset.yaml"),
        epochs=epochs,
        imgsz=640,
        batch=4,
        device=0,
        amp=True,
        freeze=10,
        patience=5,
        workers=0,
        cache=False,
        seed=42,
        optimizer="AdamW",
        project=str(report_dir),
        name="detector",
        exist_ok=False,
    )
    best = report_dir / "detector" / "weights" / "best.pt"
    fine = load_model(best.parent, best.name)
    report["adapted_validation"] = summarize(fine.val(**val_options))
    improved = (
        report["adapted_validation"]["map50_95"] > report["pretrained_validation"]["map50_95"]
    )
    chosen = best if improved else ROOT / "models/yolov8n.pt"
    selected_model = fine if improved else load_model(ROOT / "models", "yolov8n.pt")
    report["selection"] = {
        "weight": str(chosen),
        "sha256": sha256_file(chosen),
        "rule": "highest_validation_map50_95",
    }
    report["chosen_test"] = summarize(selected_model.val(**val_options, split="test"))
    report["baseline_test"] = summarize(
        load_model(ROOT / "models", "yolov8n.pt").val(**val_options, split="test")
    )
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "completed"
    write_json(registry, report)
    with (report_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["evaluation", "map50", "map50_95", "precision", "recall"]
        )
        writer.writeheader()
        for name in ("pretrained_validation", "adapted_validation", "chosen_test", "baseline_test"):
            writer.writerow({"evaluation": name, **report[name]})
    return report
