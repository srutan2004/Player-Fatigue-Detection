"""Validate delivery, summarize measured evidence and hash the completed four-phase project."""
# ruff: noqa: E501

import json
import subprocess

import pandas as pd

from src.common import ROOT, provenance, sha256_file, write_json


def finalize_project():
    """Require real completed stages and verified artifacts before marking implementation complete.

    Implementation completion and biological evidence are separate. Sparse pose windows,
    unavailable baselines and missing comparison periods remain explicit negative findings.
    """
    phase4 = ROOT / "results/phase4"
    reports = {}
    for key, path in {
        "measurements": ROOT / "results/phase3/completion_report.json",
        "analysis": phase4 / "analysis_report.json",
        "broadcast": phase4 / "broadcast_audit/report.json",
        "notebook": phase4 / "notebook_execution.json",
        "video": phase4 / "annotated_demo.json",
        "dashboard": phase4 / "dashboard_qa.json",
    }.items():
        reports[key] = json.loads(path.read_text("utf-8"))
        if reports[key]["status"] not in ("completed", "passed"):
            raise ValueError(f"Unfinished required stage: {key}")
    controls = json.loads((phase4 / "synthetic_controls.json").read_text("utf-8"))
    if not all(c["passed"] for c in controls["controls"]):
        raise ValueError("Synthetic controls did not pass")
    if not reports["notebook"]["all_code_cells_executed"]:
        raise ValueError("Notebook cells incomplete")
    if reports["dashboard"].get("dashboard_sha256") != sha256_file(phase4 / "dashboard.html"):
        raise ValueError("Dashboard QA does not match the delivered HTML")
    checks = []
    for command in (
        ["uv", "run", "--no-sync", "pytest", "-q"],
        ["uv", "run", "--no-sync", "ruff", "check", "src", "scripts", "tests", "main.py"],
        ["uv", "sync", "--locked", "--group", "dev", "--group", "notebook", "--dry-run"],
    ):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        checks.append(
            {
                "command": command,
                "exit_code": result.returncode,
                "output": result.stdout + result.stderr,
            }
        )
        if result.returncode:
            write_json(phase4 / "verification.json", {"status": "failed", "checks": checks})
            raise RuntimeError(f"Required check failed: {' '.join(command)}")
    write_json(phase4 / "verification.json", {"status": "passed", "checks": checks})
    plot_files = list((phase4 / "plots").glob("*.png"))
    if len(plot_files) < 9:
        raise ValueError("Expected eight metric panels plus paired comparison")
    pairs = pd.read_csv(phase4 / "early_late_comparison.csv")
    metrics = reports["analysis"]["metrics"]
    supported = [r for r in metrics if r["status"] == "supported_candidate"]
    study_path = ROOT / "data/manifests/study.json"
    study = json.loads(study_path.read_text("utf-8"))
    study["execution_status"] = "completed_phases_1_2_3_4"
    study.setdefault("selection_status", study["status"])
    study["status"] = "completed"
    write_json(study_path, study)
    rows = ["| Metric | Early | Late | Change |", "|---|---:|---:|---:|"]
    for r in pairs[(pairs.player_id == "A_blue_36") & pairs.paired_eligible].itertuples():
        rows.append(
            f"| {r.metric} | {r.early_mean:.3f} | {r.late_mean:.3f} | {r.change_pct:+.1f}% |"
        )
    broadcast = pd.read_csv(phase4 / "broadcast_audit/clips.csv")
    status_counts = pd.Series([r["status"] for r in metrics]).value_counts().to_dict()
    report = f"""# Final project report: football movement and fatigue-related decline

All four implementation phases have been executed on the RTX 4060 laptop. This completes
the software and its scoped case study; it does **not** establish physiological fatigue.
Three outfield players from two matches were selected before inspecting longitudinal
metrics. Four 25-minute segments supply 100 distinct footage minutes, 225,000 native
target-frame records, 72 nonoverlapping analysis windows and 63 accepted same-leg cycles.

## Delivered workflow

1. Verified CUDA environment, permitted pinned dataset access, source timing/schema and a pilot.
2. Independent YOLOv8n detection, ByteTrack tracklets, annotation-assisted player association,
   camera/calibration guards and one bounded detector adaptation experiment.
3. Native 25-fps pretrained YOLOv8m-Pose on suitable running bouts; confidence-gated joints,
   measured ground movement, projected gait cycles, exclusions and intermediate tables.
4. Own-player baseline normalization, descriptive HAC trends, conservative PELT change-point
   search, smoothing/window/penalty sensitivity, player-level early/late proxy validation,
   separate broadcast stress audit, plots, annotated video, minimap and interactive dashboard.

The software returns supported candidate, no supported change, quality-confounded,
unavailable baseline or insufficient data. It does not force a boundary for every metric.

## Main observed comparison

Only **A_blue_36** qualifies for paired early/late movement comparison. Both periods need
at least four eligible windows and 70% full-clip movement exposure. Period means weight
eligible windows by valid observed seconds. These are selected segment means, not averages
of complete match halves. B_white_17 and B_white_9 late periods fail the coverage/count gate.

{chr(10).join(rows)}

Speed units are m/s; high-intensity distance is meters per observed minute at speeds >=5.5 m/s.
The changes are compatible with reduced observed workload, but tactics, pacing, opportunity,
identity/anchor noise and selective missingness are alternative explanations. There is
only one qualified paired player, so the configured population paired test is not run.
Windows are never counted as independent players. No substitution timestamps are verified.

## Baseline and onset evidence

Baseline = the first 600 **elapsed seconds of the selected early clip**, not the first ten
minutes from kickoff. Require at least three eligible nonoverlapping windows and 420 seconds
of qualified movement exposure. A_blue_36 has 389.88 eligible baseline seconds;
B_white_17 has only two eligible baseline windows (203.00 seconds). Their movement baselines
are unavailable. B_white_9 has 465.96 eligible seconds and an available movement baseline.
Near-zero baselines have no valid percentage change and are explicitly excluded.

Trend fitting needs six contiguous eligible windows; change-point search needs eight,
with at least three observations on either side. Missing windows and clip boundaries break
sequences. Descriptive OLS uses lag-1 HAC uncertainty and Holm-adjusted p-values per player.
These tiny-series statistics are exploratory; adjusted p-values do not establish fatigue.

Candidate PELT changes need a drop of at least 10 baseline percentage points and twice local
noise, and agreement across at least two of three penalties. Sensitivity uses smoothing
0.6/0.8/1.2 seconds and nonoverlapping windows of 120/180/240 seconds. Corroboration needs at
least two smoothing values, two window widths, and half of assessable settings. Gait poses
are held fixed; no additional pose inference or training is performed by sensitivity analysis.

**Supported onset candidates: {len(supported)}.** B_white_9's primary average-speed candidate
fails robustness and remains quality-confounded. Unsupported candidates are stored for
inspection, but are not drawn as accepted onset lines. The unprocessed 30–60-minute interval
cannot reveal onset. The configured second-half offset omits first-half stoppage time.

Across 24 player/metric analyses: `{status_counts}`.

## Gait availability

There are **zero eligible nonoverlapping gait windows**. The accepted cycles remain useful
for an inspected, descriptive methodology demonstration, but no window reaches the required
20 accepted running seconds plus ten same-leg cycles. Consequently cadence, stride-length,
vertical-motion and knee-excursion decline inference are unavailable. Thresholds were not
lowered to manufacture results. Knee angles are 2D projections; vertical motion is normalized
image oscillation, not physical meters; ankle peaks are not sensor-validated foot strikes.

## Independent checks and hardware experiment

The one detector experiment used 387 training / 100 validation / 100 held-out-match tiles,
15 epochs, frozen early layers and AMP, completing in approximately 258 seconds including
evaluations. Pretrained validation mAP50–95 0.710 exceeded adapted 0.680; pretrained weights
were retained. Selected held-out mAP50–95 was 0.723. This demonstrates bounded training is
feasible on the 8-GB RTX 4060; larger-model training is not required. Labels are partly
model-assisted and this is not a new benchmark. See [PHASE_1_3_REPORT.md](PHASE_1_3_REPORT.md).

The no-identity-hint tracking audit reports known-player recall 0.794 at IoU 0.5 and 21
consecutive observed ID switches over 60 seconds. It is not full IDF1/HOTA. The long study
uses supplied identity hints, so its correspondence is conditional, not autonomous accuracy.
Raw study tracker caches retain their historical short loss horizon; fresh extraction uses
the corrected one-second ByteTrack buffer. No silent cache retracking was performed.

The curved panoramic view invalidates a single raw-pixel homography. A quadratic rational
ground warp passed held-out field-point checks: median/p95 0.654/1.933 m for match A and
0.615/2.101 m for match B. These are field-point errors, not ground-truth player-position
errors. Outside-hull mappings and unverified camera registration remain unavailable.

## Broadcast robustness

Five independent public [SoccerNet-GSR test clips](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024)
were selected in sorted order before inspection. Pinned commit:
`3cc710eb6d53a23350a3d2863311c1c0a2e645d7`. Only 300 sampled JPEGs and five label files were
fetched using validated archive byte ranges; no full 8.85-GB ZIP was downloaded. These
approximately 2-fps observations cannot support gait or long-term fatigue. The original
dataset is described by the [official SoccerNet game-state project](https://github.com/soccernet/sn-gamestate).

Per-clip known-player recall at IoU 0.5 ranges from {broadcast.player_recall_iou50.min():.1%}
to {broadcast.player_recall_iou50.max():.1%}. The six-joint confidence gate passes in 98/100
selected crops, chosen as the two tallest matched players every sixth sampled frame.
This favors visible large players and is availability, not anatomical accuracy. Fixed
first-frame calibration remains valid in only 1.7–8.3% of sampled frames by the camera guard;
moving broadcast cameras require dynamic calibration beyond this scoped implementation.
No short clips were stitched into the longitudinal study or used for training.

## Verification and delivery

The full test suite and lint check pass; the locked dependency dry run requires no changes.
Known-step, stationary-noise and constant software controls all pass. They establish algorithm
behavior on synthetic data, not clinical performance. The end-to-end notebook executed all
{reports["notebook"]["code_cells"]} code cells. The local standalone HTML dashboard rendered
without JavaScript errors and all three player-selector options were checked.
The React dashboard uses liquid-glass surfaces, dark/light themes and responsive layouts;
it provides six evidence sections, window inspection, gait filters, pitch playback,
searchable tables and data export. The latest browser QA passed
{reports["dashboard"].get("passed_checks", 3)} checks, including offline operation.
The continuous annotated 10-second video contains 250 source-aligned frames at 25 fps.

| Deliverable | Location |
|---|---|
| Source and CLI | `src/`, `scripts/`, `python -m src.cli --help` |
| Notebook and executed copy | `notebooks/pipeline_demo.ipynb`, `results/phase4/pipeline_demo.executed.ipynb` |
| Intermediate and windowed tables | `results/metrics_csv/`, `results/phase2/study/`, `results/phase3/study_identity_v3/` |
| Normalized metrics, baselines, trends, sensitivity, pairs | `results/phase4/*.csv` |
| Final static panels | `results/phase4/plots/`, copied under `results/plots/phase4/` |
| Interactive dashboard | `results/phase4/dashboard.html` |
| Continuous box/skeleton/speed/minimap preview | `results/phase4/annotated_demo.mp4` |
| Five broadcast audit sheets/tables | `results/phase4/broadcast_audit/` |
| Setup, methodology and limitations | `README.md`, `LIMITATIONS.md`, both implementation plans |
| Completion/checks and file hashes | `results/completion_report.json`, `results/phase4/verification.json`, `results/reproducibility.json` |

Scientific acceptance is quality-dependent: software execution and delivery are complete;
physiological fatigue detection and gait-decline validation are **not demonstrated** by this
case study. Collecting clearer continuous running observations and independent ground truth
would be required to support those claims. Dataset access conditions continue to apply;
raw footage and tokens are excluded from Git and are not redistributed by these reports.
"""
    (ROOT / "FINAL_REPORT.md").write_text(report, "utf-8")
    import shutil

    shutil.copytree(phase4 / "plots", ROOT / "results/plots/phase4", dirs_exist_ok=True)
    completion = {
        **provenance(),
        "status": "completed",
        "completed_phases": [1, 2, 3, 4],
        "distinct_footage_minutes": 100,
        "players": 3,
        "matches": 2,
        "long_segments": 4,
        "broadcast_audit_clips": 5,
        "supported_onset_candidates": len(supported),
        "eligible_gait_windows": 0,
        "qualified_paired_movement_players": 1,
        "implementation_complete": True,
        "physiological_fatigue_validated": False,
        "analysis_status_counts": status_counts,
        "checks": "results/phase4/verification.json",
        "report": "FINAL_REPORT.md",
        "dashboard": "results/phase4/dashboard.html",
    }
    write_json(ROOT / "results/completion_report.json", completion)
    source_paths = [
        *ROOT.glob("src/*.py"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("tests/*.py"),
        *ROOT.glob("configs/*"),
        *ROOT.glob("*.md"),
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        ROOT / "main.py",
        ROOT / "notebooks/pipeline_demo.ipynb",
        ROOT / "dashboard/app.jsx",
        ROOT / "dashboard/styles.css",
        ROOT / "dashboard/build.mjs",
        ROOT / "dashboard/package.json",
        ROOT / "dashboard/package-lock.json",
        ROOT / "dashboard/assets/app.js",
        ROOT / "dashboard/README.md",
        study_path,
    ]
    artifact_paths = [
        *phase4.rglob("*"),
        *(ROOT / "results/metrics_csv").rglob("*"),
        ROOT / "results/completion_report.json",
    ]

    def records(paths):
        """Hash delivered source/analysis bytes; upstream multi-GB sources retain their own manifests."""
        return [
            {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(set(paths))
            if p.is_file()
        ]

    write_json(
        ROOT / "results/reproducibility.json",
        {
            **provenance(),
            "source_files": records(source_paths),
            "artifacts": records(artifact_paths),
            "source_video_hashes": "data/hub/manifests/*.json",
            "dataset_revision": study["revision"],
            "broadcast_revision": reports["broadcast"]["revision"],
            "model_hashes": "Phase 2 extraction and Phase 3 pose reports",
            "note": "File hashes capture the delivered working-tree snapshot; original execution preceded the first Git commit.",
        },
    )
    return completion


if __name__ == "__main__":
    print(json.dumps(finalize_project(), indent=2))
