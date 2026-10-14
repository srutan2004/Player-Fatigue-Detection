"""Four-phase command-line workflow with actionable, non-silent failure states."""

import argparse
import json
import sys
from pathlib import Path

from src.common import ROOT, load_config, write_json


def main(argv=None) -> int:
    """Run an explicit stage; training occurs only through train-detector."""
    parser = argparse.ArgumentParser(description="Football movement pipeline — Phases 1–4")
    sub = parser.add_subparsers(dest="command", required=True)
    environment = sub.add_parser(
        "check-environment", help="Verify CUDA and optionally model passes"
    )
    environment.add_argument(
        "--output", type=Path, default=ROOT / "results/phase1/environment.json"
    )
    environment.add_argument("--smoke-models", action="store_true")
    environment.add_argument("--device", default="cuda:0")
    dataset = sub.add_parser("dataset-probe", help="Resolve release and test gated-file access")
    dataset.add_argument("--output", type=Path, default=ROOT / "results/phase1/dataset_access.json")
    dataset.add_argument("--revision", default="v1.2")
    dataset.add_argument("--match", default="117093")
    download = sub.add_parser("dataset-download", help="Download explicitly selected files only")
    download.add_argument("--probe", type=Path, default=ROOT / "results/phase1/dataset_access.json")
    download.add_argument("--file", action="append", required=True, dest="filenames")
    download.add_argument("--destination", type=Path, default=ROOT / "data/hub")
    download.add_argument("--max-gib", type=float, default=5)
    download.add_argument("--transport", choices=["ranges", "hub"], default="ranges")
    for command in ("inspect", "pilot", "extract", "select-player"):
        action = sub.add_parser(command)
        action.add_argument(
            "--config",
            type=Path,
            required=command in ("extract", "select-player"),
            default=ROOT / "configs/pilot.yaml",
        )
        if command == "inspect":
            action.add_argument("--skip-source-hash", action="store_true")
    annotation = sub.add_parser("audit-annotations")
    annotation.add_argument("path", type=Path)
    annotation.add_argument("--output", type=Path, required=True)
    index = sub.add_parser("reference-index")
    index.add_argument("source", type=Path)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--index-base", type=int, choices=[0, 1], default=1)
    mot = sub.add_parser("mot-index")
    mot.add_argument("source", type=Path)
    mot.add_argument("--output", type=Path, required=True)
    mot.add_argument("--index-base", type=int, choices=[0, 1], default=0)
    field = sub.add_parser("fit-field")
    field.add_argument("points", type=Path)
    field.add_argument("--output", type=Path, required=True)
    field.add_argument("--width", type=int, default=4096)
    field.add_argument("--height", type=int, default=1080)
    prepare = sub.add_parser("prepare-detector")
    prepare.add_argument("spec", type=Path)
    prepare.add_argument("--output", type=Path, default=ROOT / "data/detector_subset")
    train = sub.add_parser("train-detector")
    train.add_argument("--dataset", type=Path, default=ROOT / "data/detector_subset")
    train.add_argument("--output", type=Path, default=ROOT / "results/phase2/training")
    train.add_argument("--epochs", type=int, default=15)
    metrics = sub.add_parser("metrics")
    metrics.add_argument("source", type=Path)
    metrics.add_argument("--output", type=Path, required=True)
    poses = sub.add_parser("pose-pass")
    poses.add_argument("source", type=Path)
    poses.add_argument("--config", type=Path, required=True)
    poses.add_argument("--output", type=Path, required=True)
    poses.add_argument("--batch", type=int, default=8)
    analysis = sub.add_parser(
        "analyze", help="Baseline, trends, sensitivity, proxy validation and dashboard"
    )
    analysis.add_argument("--study", type=Path, default=ROOT / "data/manifests/study.json")
    analysis.add_argument("--metrics", type=Path, default=ROOT / "results/metrics_csv")
    analysis.add_argument("--output", type=Path, default=ROOT / "results/phase4")
    args = parser.parse_args(argv)
    try:
        if args.command == "check-environment":
            from src.environment import check_environment

            report = check_environment(args.output, args.smoke_models, args.device)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["status"] == "passed" else 1
        if args.command == "dataset-probe":
            from src.dataset import probe_dataset

            report = probe_dataset(args.output, args.revision, args.match)
            print(json.dumps(report, indent=2, default=str))
            return 0 if report["status"] == "accessible" else 2
        if args.command == "dataset-download":
            from src.dataset import download_files

            report = download_files(
                args.probe, args.filenames, args.destination, args.max_gib, args.transport
            )
        elif args.command == "audit-annotations":
            from src.annotations import audit_annotations

            report = audit_annotations(args.path)
            write_json(args.output, report)
        elif args.command == "inspect":
            from src.intake import inspect_video

            report = inspect_video(load_config(args.config), not args.skip_source_hash)
        elif args.command == "pilot":
            from src.benchmark import run_pilot

            config = load_config(args.config)
            try:
                report = run_pilot(config)
            except Exception as exc:
                write_json(
                    config["output_dir"] / "pilot_status.json",
                    {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                )
                raise
        elif args.command == "reference-index":
            from src.reference import build_reference_index

            report = build_reference_index(args.source, args.output, args.index_base)
        elif args.command == "mot-index":
            from src.reference import build_mot_index

            report = build_mot_index(args.source, args.output, args.index_base)
        elif args.command == "fit-field":
            from src.calibrate import fit_ground_mapping

            report = fit_ground_mapping(
                args.points, args.output, image_size=(args.width, args.height)
            )
        elif args.command == "prepare-detector":
            from src.training import prepare_training_subset

            report = prepare_training_subset(args.spec, args.output)
        elif args.command == "train-detector":
            from src.training import train_bounded

            report = train_bounded(args.dataset, args.output, args.epochs)
        elif args.command == "extract":
            from src.detect_track import extract_segment

            config = load_config(args.config)
            try:
                report = extract_segment(config)
            except Exception as exc:
                write_json(
                    config["output_dir"] / "extraction_status.json",
                    {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                )
                raise
        elif args.command == "select-player":
            from src.select_player import selection_frame

            config = load_config(args.config)
            report = selection_frame(config, config["output_dir"] / "selection")
        elif args.command == "analyze":
            from src.fatigue_analysis import run_analysis

            report = run_analysis(args.study, args.metrics, args.output)
        elif args.command == "metrics":
            from src.process_metrics import process_metrics

            report = process_metrics(args.source, args.output)
        elif args.command == "pose-pass":
            from src.pose_pass import pose_from_tracks

            report = pose_from_tracks(
                load_config(args.config), args.source, args.output, args.batch
            )
        printable = {
            key: value
            for key, value in report.items()
            if key
            not in (
                "images",
                "config",
                "ground_mapping",
                "annotations",
                "pixel_points",
                "world_points",
            )
        }
        if "images" in report:
            printable["image_records_saved"] = len(report["images"])
        print(json.dumps(printable, indent=2, default=str))
        return 2 if report.get("status") == "awaiting_video" else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
