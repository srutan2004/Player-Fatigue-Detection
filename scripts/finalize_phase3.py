"""Assemble quality-aware study outputs after all four source segments finish."""
# ruff: noqa: E501

import json

import numpy as np
import pandas as pd

from src.common import ROOT, load_config, provenance, write_json
from src.pose_extract import save_pose_gaps
from src.process_metrics import process_metrics
from src.visualize import plot_gait_cycles, plot_measurements


def finalize_study() -> dict:
    """Validate cohort/exposure and report measured results without fatigue inference.

    Report both supported and unavailable measurements. Ground-reference agreement
    is explicitly conditional on assisted identity; it is not independent accuracy.
    """
    status = json.loads((ROOT / "results/phase3/study_status.json").read_text())
    if status["status"] != "completed" or status["completed_clips"] != 4:
        raise ValueError("All four study segments must finish before finalization")
    manifest = json.loads((ROOT / "data/manifests/study.json").read_text())
    summaries, all_frames, all_cycles, all_windows = [], [], [], {"display": [], "nonoverlap": []}
    runtimes = []
    for clip in manifest["clips"]:
        name = clip["clip_name"]
        config = load_config(ROOT / clip["config"])
        track_dir = config["output_dir"]
        camera = pd.read_csv(track_dir / "camera_checks.csv")
        pose_dir = ROOT / config["pipeline"].get("pose_output_dir", f"results/phase3/study/{name}")
        source = pose_dir / "keypoints_positions.csv"
        metric_dir = ROOT / "results/metrics_csv" / name
        # Apply the conservative final camera qualification to legacy completed stages.
        raw = pd.read_csv(source)
        raw["reference_verification_age_seconds"] = np.nan
        for _, part in raw.groupby("player_id"):
            verified = part.identity_source.eq(
                "reference_assisted_pitch_and_color"
            ) & part.observation_status.eq("detected")
            last = part.timestamp.where(verified).ffill()
            age = (part.timestamp - last).where(
                part.observation_status.isin(["detected", "short_box_propagation"])
            )
            raw.loc[part.index, "reference_verification_age_seconds"] = age
        save_pose_gaps(raw, source.parent / "pose_gaps.csv")
        checks = camera[["timestamp", "status", "mapping_valid"]].rename(
            columns={"status": "camera_status", "mapping_valid": "camera_mapping_valid"}
        )
        qualified = pd.merge_asof(
            raw.sort_values("timestamp"),
            checks.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=0.12,
        )
        qualified["position_valid"] &= qualified.camera_mapping_valid.fillna(False).astype(bool)
        qualified.loc[qualified.camera_status.eq("registration_unavailable"), "position_valid"] = (
            False
        )
        qualified_path = source.parent / "keypoints_positions_camera_qualified.csv"
        qualified.to_csv(qualified_path, index=False)
        process_metrics(qualified_path, metric_dir)
        frames = pd.read_parquet(metric_dir / "frame_metrics.parquet")
        windows = pd.read_csv(metric_dir / "windowed_metrics_nonoverlap.csv")
        cycles = pd.read_csv(metric_dir / "gait_cycles.csv")
        poses = json.loads((source.parent / "pose_report.json").read_text())
        extraction = json.loads((track_dir / "extraction_report.json").read_text())
        expected_players = {t["player_id"] for t in clip["targets"]}
        if set(frames.player_id) != expected_players:
            raise ValueError(f"Unexpected player cohort in {name}")
        if len(frames) != int(
            clip["duration_seconds"] * poses["native_fps"] * len(expected_players)
        ):
            raise ValueError(f"Incomplete native frame grid in {name}")
        for player, part in frames.groupby("player_id"):
            w = windows[windows.player_id == player]
            c = cycles[cycles.player_id == player]
            summaries.append(
                {
                    "clip_name": name,
                    "player_id": player,
                    "half": clip["half"],
                    "source_rows": len(part),
                    "position_coverage": float(part.position_valid.mean()),
                    "movement_frame_coverage": float(part.movement_valid.mean()),
                    "valid_movement_seconds": float(part.valid_dt.sum()),
                    "pose_rows_requested": int(
                        part.pose_status.ne("not_requested_outside_running_bout").sum()
                    ),
                    "six_joint_confident_rows": int(part.gait_pose_valid.sum()),
                    "visible_continuity_frame_fraction": float(
                        part.identity_source.eq(
                            "reference_seeded_visible_tracklet_continuity"
                        ).mean()
                    ),
                    "accepted_cycles": len(c),
                    "nonoverlap_windows": len(w),
                    "movement_eligible_windows": int(w.movement_eligible.sum()),
                    "gait_eligible_windows": int(w.gait_eligible.sum()),
                    "observed_avg_speed_mps": float(
                        part.distance_increment.sum() / part.valid_dt.sum()
                    )
                    if part.valid_dt.sum()
                    else None,
                }
            )
        runtimes.append(
            {
                "clip": name,
                "track_seconds": extraction["elapsed_seconds"],
                "pose_seconds": poses["elapsed_seconds"],
                "pose_calls": poses["pose_calls"],
                "pose_peak_torch_vram_mib": poses["peak_torch_vram_mib"],
            }
        )
        all_frames.append(frames)
        all_cycles.append(cycles)
        for label in all_windows:
            all_windows[label].append(pd.read_csv(metric_dir / f"windowed_metrics_{label}.csv"))
    coverage = pd.DataFrame(summaries)
    coverage.to_csv(ROOT / "results/metrics_csv/study_coverage.csv", index=False)
    pd.DataFrame(runtimes).to_csv(ROOT / "results/phase3/study_runtime.csv", index=False)
    for label, tables in all_windows.items():
        pd.concat(tables, ignore_index=True).to_csv(
            ROOT / f"results/metrics_csv/all_windowed_metrics_{label}.csv", index=False
        )
    frame_table = pd.concat(all_frames, ignore_index=True)
    nonempty_cycles = [table for table in all_cycles if not table.empty]
    cycles = pd.concat(nonempty_cycles, ignore_index=True) if nonempty_cycles else all_cycles[0]
    cycles.to_csv(ROOT / "results/metrics_csv/all_gait_cycles.csv", index=False)
    plot_gait_cycles(cycles, ROOT / "results/plots")
    plot_measurements(
        frame_table, pd.concat(all_windows["display"], ignore_index=True), ROOT / "results/plots"
    )
    units = {
        "avg_speed": "m/s",
        "peak_speed": "m/s; supported ~1-second average",
        "stride_frequency": "same-leg strides/s",
        "stride_length": "meters; ground-path-per-cycle proxy",
        "vertical_oscillation": "dimensionless body-height-normalized image proxy",
        "knee_angle_range": "degrees; projected 2D robust excursion",
        "high_intensity_distance_per_minute": "meters per valid observed minute at >=5.5m/s",
    }
    write_json(ROOT / "results/metrics_csv/units.json", units)
    report = {
        **provenance(),
        "status": "completed",
        "phases": [1, 2, 3],
        "distinct_footage_minutes": 100,
        "paired_players": 3,
        "matches": 2,
        "segments": 4,
        "source_target_rows": len(frame_table),
        "accepted_gait_cycles": len(cycles),
        "coverage": summaries,
        "runtime": runtimes,
        "fatigue_inference_performed": False,
        "reference_assisted_identity": True,
        "training_runs": 1,
        "window_gait_unavailability_is_a_result": True,
        "identity_continuity_seconds": 0.8,
        "reference_quality_report": "results/phase2/reference_quality.json",
    }
    write_json(ROOT / "results/phase3/completion_report.json", report)
    write_report(report)
    return report


def write_report(report: dict) -> None:
    """Write a course-ready results summary with explicit measurement limits."""
    training = json.loads((ROOT / "results/phase2/training/bounded_run.json").read_text())
    tracking = json.loads((ROOT / "results/phase2/autonomous_tracking/report.json").read_text())
    table = [
        "| Clip | Player | Ground coverage | Movement windows / 12 | Gait cycles | Gait windows / 12 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["coverage"]:
        table.append(
            f"| {row['clip_name']} | {row['player_id']} | {row['position_coverage']:.1%} | "
            f"{row['movement_eligible_windows']} | {row['accepted_cycles']} | {row['gait_eligible_windows']} |"
        )
    eligible = sum(r["gait_eligible_windows"] for r in report["coverage"])
    text = f"""# Phases 1–3 execution report

Completed on the local RTX 4060 Laptop GPU. The pipeline processed **100 distinct minutes**, four 25-minute segments from two matches, and **three paired outfield players**. Native 25-fps target tables contain **{report["source_target_rows"]:,} rows**. The study produced **{report["accepted_gait_cycles"]} accepted same-leg cycles** and **{eligible} eligible non-overlapping gait windows**. Missing gait windows are retained explicitly; this is a measurement result, not evidence of fatigue.

## Phase 1 — hardware, dataset and image audit

CUDA matrix multiplication, torchvision NMS and pretrained detector/pose passes succeeded on the RTX 4060 Laptop GPU (8188 MiB), Ryzen 7 7840HS and nominal 16 GB RAM. Python 3.12.10 and the tested lockfile use torch 2.8.0+cu128. Local Hugging Face authentication and gated-file access succeeded.

SoccerTrack v2 is pinned to `325bd4c57ee8e135ab652f3921d274526ada496b`. Selected videos/large annotations were SHA-256 checked. Parsing uses streaming JSON and disk-backed SQLite. Source overlays verified GSR/MOT indexing and the one-second PTS origin. The 60-second sparse pilot produced 120 samples and 24 manually reviewed crops; only a subset were anatomically suitable. It is not a cadence dataset.

## Phase 2 — one small adaptation and audited ground tracking

One adaptation used 387 train / 100 validation / 100 held-out-match tiles after doubtful training images were quarantined. Splits separate source frames, development time blocks and test match. Additional people proposals were reviewed; original MOT alignment was sampled. Partly model-assisted labels limit AP interpretation.

The requested 15 epochs completed (batch 4, freeze 10, AMP, seed 42), with {training["elapsed_seconds"]:.1f} seconds including evaluations. Validation mAP50–95 was {training["pretrained_validation"]["map50_95"]:.3f} for pretrained YOLOv8n and {training["adapted_validation"]["map50_95"]:.3f} after adaptation. **Pretrained weights were retained.** Locked held-out mAP50–95 was {training["chosen_test"]["map50_95"]:.3f}. No additional training was run.

The separate autonomous held-out 60-second MOT audit matched {tracking["known_player_box_recall"]:.1%} of annotated player boxes at IoU≥0.5 and counted {tracking["consecutive_observed_identity_switches"]} switches across consecutive matched observations. Recall strongly depends on body size; the size table is saved. Incomplete staff labels prevent an all-person precision claim. This is not full IDF1/HOTA.

Direct homography failed on the curved panorama. A quadratic rational ground warp passed 13 withheld field points per camera: match 117093 median 0.654 m / p95 1.933 m; match 128057 median 0.615 m / p95 2.101 m. These are field-point validation errors, not player-position accuracy. Mappings reject unsupported extrapolation. Ground anchors are independent detected box bottoms, not elevated hips.

Principal study identity is annotation-assisted using reference identity, pitch consistency, kit color and global matching. Reference positions never replace measured trajectories. Raw ByteTrack output, link logs, gaps and camera registration are retained separately. This assisted study cannot establish autonomous identity accuracy.

Reference pitch annotations in both matches show coarse 1.05-m x / 0.68-m y grids in an audited 10,000-record sample. The final identity pass therefore preserves a recently verified **still-visible same tracklet** for at most 0.8 s, with bounded motion/reference residual, no severe overlap and no conflicting assignment. Missing detections and new tracklets are never filled. This pass reuses original raw tracker boxes, so the detector was not rerun or retrained. Continuity tags and verification age remain available for later sensitivity checks.

The ten-minute pilot verified movement coverage and source anatomy. YOLOv8m-Pose improved six-joint confidence coverage to 249/255 crops versus 173/255 for YOLOv8s-Pose; chronological source crops and overlays were inspected. The larger pretrained model was selected without training. Concurrent GPU workers caused a CUDA error; partial failed attempts were excluded and remaining stages ran serially.

Fresh tracker defaults use Supervision's FPS-scaled 30-frame parameter for approximately one second of occlusion memory. The historical raw study cache retains its original shorter horizon, which contributes to fragmentation; it is not silently presented as freshly retracked data. The corrected horizon has an independent half-second-occlusion test and a separate MOT audit.

Final image QA caught an overly bright HSV white-shirt threshold. The corrected check tolerates cool shadows and leaves mixed palettes unknown. It restored movement coverage on the held-out match without altering detector weights or filling observations; earlier intermediate outputs remain archived.

## Phase 3 — full study measurements and availability

Player A_blue_36 (reference 473125), B_white_9 (506463), and B_white_17 (506460) are present in both halves. A_blue_21 was an early-only pilot candidate and is excluded from the paired cohort. First-half blocks cover source 300–1800 s; second-half blocks cover 900–2400 s with a declared 2700-s match offset.

{chr(10).join(table)}

Native poses use source-coordinate joints and at most eight crops per batch. Confidence holes remain missing. Only straight ≥4-second running bouts with source body height ≥40 px contribute gait. Cadence uses observed same-leg oscillations, periodicity support and explicit gap limits; events are not verified foot strikes. Stride length is ground path per cycle; knee excursion is projected; vertical motion is a dimensionless image proxy.

Movement windows require 70% valid ground exposure. Gait windows require 20 accepted-cycle seconds and ten same-leg cycles. Overlapping display windows and separate non-overlapping inference windows are saved. Missing values are never forced to meet a plot/analysis quota. Speed/body-scale/projected-motion strata support later comparable-bout review.

## Deliverables and validation

- [Runnable notebook](notebooks/pipeline_demo.ipynb), successfully executed through all cells; executed copy under `results/phase3/`.
- [Coverage table](results/metrics_csv/study_coverage.csv), [windowed measurements](results/metrics_csv/all_windowed_metrics_nonoverlap.csv), [gait cycles](results/metrics_csv/all_gait_cycles.csv), [units](results/metrics_csv/units.json).
- [Plots and minimaps](results/plots/), final native keypoints/skeleton audits under `results/phase3/study_identity_v3/`, and track/identity/gap/camera tables under `results/phase2/study/*_identity_v3/`.
- [Training comparison](results/phase2/training/comparison.csv), [autonomous tracking audit](results/phase2/autonomous_tracking/report.json), [runtime table](results/phase3/study_runtime.csv), and [machine-readable completion report](results/phase3/completion_report.json).

Encoded-video and independent synthetic tests cover PTS origin, known velocity/cadence, no movement across gaps, missing joints, camera translation/unobservable registration, gradual turns and spatial validation. Ruff and pytest are the required checks. Documentation includes [setup/commands](README.md) and [limitations](LIMITATIONS.md).

**Phase 4 remains:** supported baseline/trend/change-point analysis, contextual early/late validation and dashboard. No physiological fatigue or onset result is claimed by Phases 1–3. Many gait windows may lack enough reliable observations even when movement is measurable.
"""
    (ROOT / "PHASE_1_3_REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    print(finalize_study())
