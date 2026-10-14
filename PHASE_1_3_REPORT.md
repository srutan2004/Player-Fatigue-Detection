# Phases 1–3 execution report

Completed on the local RTX 4060 Laptop GPU. The pipeline processed **100 distinct minutes**, four 25-minute segments from two matches, and **three paired outfield players**. Native 25-fps target tables contain **225,000 rows**. The study produced **63 accepted same-leg cycles** and **0 eligible non-overlapping gait windows**. Missing gait windows are retained explicitly; this is a measurement result, not evidence of fatigue.

## Phase 1 — hardware, dataset and image audit

CUDA matrix multiplication, torchvision NMS and pretrained detector/pose passes succeeded on the RTX 4060 Laptop GPU (8188 MiB), Ryzen 7 7840HS and nominal 16 GB RAM. Python 3.12.10 and the tested lockfile use torch 2.8.0+cu128. Local Hugging Face authentication and gated-file access succeeded.

SoccerTrack v2 is pinned to `325bd4c57ee8e135ab652f3921d274526ada496b`. Selected videos/large annotations were SHA-256 checked. Parsing uses streaming JSON and disk-backed SQLite. Source overlays verified GSR/MOT indexing and the one-second PTS origin. The 60-second sparse pilot produced 120 samples and 24 manually reviewed crops; only a subset were anatomically suitable. It is not a cadence dataset.

## Phase 2 — one small adaptation and audited ground tracking

One adaptation used 387 train / 100 validation / 100 held-out-match tiles after doubtful training images were quarantined. Splits separate source frames, development time blocks and test match. Additional people proposals were reviewed; original MOT alignment was sampled. Partly model-assisted labels limit AP interpretation.

The requested 15 epochs completed (batch 4, freeze 10, AMP, seed 42), with 258.0 seconds including evaluations. Validation mAP50–95 was 0.710 for pretrained YOLOv8n and 0.680 after adaptation. **Pretrained weights were retained.** Locked held-out mAP50–95 was 0.723. No additional training was run.

The separate autonomous held-out 60-second MOT audit matched 79.4% of annotated player boxes at IoU≥0.5 and counted 21 switches across consecutive matched observations. Recall strongly depends on body size; the size table is saved. Incomplete staff labels prevent an all-person precision claim. This is not full IDF1/HOTA.

Direct homography failed on the curved panorama. A quadratic rational ground warp passed 13 withheld field points per camera: match 117093 median 0.654 m / p95 1.933 m; match 128057 median 0.615 m / p95 2.101 m. These are field-point validation errors, not player-position accuracy. Mappings reject unsupported extrapolation. Ground anchors are independent detected box bottoms, not elevated hips.

Principal study identity is annotation-assisted using reference identity, pitch consistency, kit color and global matching. Reference positions never replace measured trajectories. Raw ByteTrack output, link logs, gaps and camera registration are retained separately. This assisted study cannot establish autonomous identity accuracy.

Reference pitch annotations in both matches show coarse 1.05-m x / 0.68-m y grids in an audited 10,000-record sample. The final identity pass therefore preserves a recently verified **still-visible same tracklet** for at most 0.8 s, with bounded motion/reference residual, no severe overlap and no conflicting assignment. Missing detections and new tracklets are never filled. This pass reuses original raw tracker boxes, so the detector was not rerun or retrained. Continuity tags and verification age remain available for later sensitivity checks.

The ten-minute pilot verified movement coverage and source anatomy. YOLOv8m-Pose improved six-joint confidence coverage to 249/255 crops versus 173/255 for YOLOv8s-Pose; chronological source crops and overlays were inspected. The larger pretrained model was selected without training. Concurrent GPU workers caused a CUDA error; partial failed attempts were excluded and remaining stages ran serially.

Fresh tracker defaults use Supervision's FPS-scaled 30-frame parameter for approximately one second of occlusion memory. The historical raw study cache retains its original shorter horizon, which contributes to fragmentation; it is not silently presented as freshly retracked data. The corrected horizon has an independent half-second-occlusion test and a separate MOT audit.

Final image QA caught an overly bright HSV white-shirt threshold. The corrected check tolerates cool shadows and leaves mixed palettes unknown. It restored movement coverage on the held-out match without altering detector weights or filling observations; earlier intermediate outputs remain archived.

## Phase 3 — full study measurements and availability

Player A_blue_36 (reference 473125), B_white_9 (506463), and B_white_17 (506460) are present in both halves. A_blue_21 was an early-only pilot candidate and is excluded from the paired cohort. First-half blocks cover source 300–1800 s; second-half blocks cover 900–2400 s with a declared 2700-s match offset.

| Clip | Player | Ground coverage | Movement windows / 12 | Gait cycles | Gait windows / 12 |
|---|---|---:|---:|---:|---:|
| 117093_first | A_blue_36 | 81.3% | 10 | 20 | 0 |
| 117093_second | A_blue_36 | 84.9% | 11 | 39 | 0 |
| 128057_first | B_white_17 | 77.6% | 9 | 4 | 0 |
| 128057_first | B_white_9 | 87.0% | 12 | 0 | 0 |
| 128057_second | B_white_17 | 63.2% | 2 | 0 | 0 |
| 128057_second | B_white_9 | 26.1% | 0 | 0 | 0 |

Native poses use source-coordinate joints and at most eight crops per batch. Confidence holes remain missing. Only straight ≥4-second running bouts with source body height ≥40 px contribute gait. Cadence uses observed same-leg oscillations, periodicity support and explicit gap limits; events are not verified foot strikes. Stride length is ground path per cycle; knee excursion is projected; vertical motion is a dimensionless image proxy.

Movement windows require 70% valid ground exposure. Gait windows require 20 accepted-cycle seconds and ten same-leg cycles. Overlapping display windows and separate non-overlapping inference windows are saved. Missing values are never forced to meet a plot/analysis quota. Speed/body-scale/projected-motion strata support later comparable-bout review.

## Deliverables and validation

- [Runnable notebook](notebooks/pipeline_demo.ipynb), successfully executed through all cells; executed copy under `results/phase3/`.
- [Coverage table](results/metrics_csv/study_coverage.csv), [windowed measurements](results/metrics_csv/all_windowed_metrics_nonoverlap.csv), [gait cycles](results/metrics_csv/all_gait_cycles.csv), [units](results/metrics_csv/units.json).
- [Plots and minimaps](results/plots/), final native keypoints/skeleton audits under `results/phase3/study_identity_v3/`, and track/identity/gap/camera tables under `results/phase2/study/*_identity_v3/`.
- [Training comparison](results/phase2/training/comparison.csv), [autonomous tracking audit](results/phase2/autonomous_tracking/report.json), [runtime table](results/phase3/study_runtime.csv), and [machine-readable completion report](results/phase3/completion_report.json).

Encoded-video and independent synthetic tests cover PTS origin, known velocity/cadence, no movement across gaps, missing joints, camera translation/unobservable registration, gradual turns and spatial validation. Ruff and pytest are the required checks. Documentation includes [setup/commands](README.md) and [limitations](LIMITATIONS.md).

**Phase 4 remains:** supported baseline/trend/change-point analysis, contextual early/late validation and dashboard. No physiological fatigue or onset result is claimed by Phases 1–3. Many gait windows may lack enough reliable observations even when movement is measurable.
