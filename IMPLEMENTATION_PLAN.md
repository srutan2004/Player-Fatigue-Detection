# Implementation Plan: Vision-Based Player Movement and Fatigue-Related Decline Analysis

Prepared: 14 October 2026

Course: BCSE407L Computer Vision, RIL

Status: technical design retained; phases 1–4 have now been implemented and executed. See [FINAL_REPORT.md](FINAL_REPORT.md) for actual measurement availability and scientific limits.

Execution update: use [FOUR_PHASE_IMPLEMENTATION_PLAN.md](FOUR_PHASE_IMPLEMENTATION_PLAN.md) for the four delivery phases and minimal-training budget. This file retains the technical detail. Only about 400 selected images/tiles will be used for one small detector fine-tuning run; the match videos are primarily analysis inputs.

## 1. Recommendation and project outcome

Proceed with the project. It is feasible on your laptop using pretrained detection and pose models, provided that camera geometry, player visibility, and statistical interpretation are corrected before implementation.

Recommended title: **Vision-Based Player Movement and Fatigue-Related Decline Analysis from Football Match Video**.

The system will reconstruct a selected player's movement, measure gait on suitable running bouts, and identify sustained changes compatible with fatigue. Its output is evidence of performance decline, with measurement quality and contextual caveats. It cannot establish physiological fatigue from video alone.

Retain the substantial scope: detection, persistent identity, pose, metric field calibration, gait features, longitudinal analysis, first/second-half comparison, annotated video, and an interactive results dashboard. Use full-pitch match recordings for the main longitudinal experiment and broadcast footage for a separate robustness experiment.

A relevant April 2026 preprint already investigates fatigue-oriented kinematics from football videos. Therefore, present the contribution as a course implementation and evaluation of a quality-aware movement/gait pipeline, rather than claiming that video-based fatigue assessment itself is novel. [Related primary research](https://arxiv.org/abs/2604.05636).

## 2. Validation of the original brief

| Original proposal | Assessment | Implementation decision |
|---|---|---|
| Pretrained person detection plus ByteTrack | Practical starting point | Keep; distinguish a temporary track ID from a persistent player identity. |
| Track one player continuously through broadcast footage | Visibility and cuts prevent uninterrupted observation | Process visible tracklets, detect cuts, and expose missing intervals. Use panoramic footage for the principal experiment. |
| Sample everything at 5 fps | Suitable for a coarse overview, weak for foot-cycle timing and fast motion | Default to 12.5–15 fps detection/tracking and native 25/30 fps pose on selected bouts. Benchmark higher tracking rates if IDs fail. |
| Transform hip centroid through field homography | Geometrically invalid as a ground position: hips are above the pitch | Use a ground-contact anchor; preserve hips for image-space gait measurements. |
| One homography per broadcast clip | Fails under pan, tilt, and zoom | Static-camera calibration for the main dataset; temporal calibration updates and quality gating for moving-camera shots. |
| A homography yields vertical oscillation in meters | Impossible with a ground-plane transform | Measure a normalized image-space hip-motion proxy; explicitly distinguish it from physical vertical displacement. |
| Ankle x peaks always yield stride frequency | Camera direction, turns, occlusion, and aliasing can dominate the signal | Restrict analysis to suitable bouts and combine limb-angle/relative-ankle signals with periodicity checks. |
| Ankle y minima identify foot strikes | Image y increases downward; extrema alone are unreliable | Estimate gait-cycle timing first; require corroboration before calling an event ground contact. |
| Lower speed or knee range proves fatigue | Tactics, running intensity, view, and opportunity are confounders | Compare similar movement regimes and report fatigue-related changes as proxies. |
| Detect one change point per metric | Forces a result even for stable/noisy data | Return zero or more supported candidates, or an explicit insufficient-data status. |
| Paired significance test on many rolling windows | Overlapping windows are dependent; frames/windows are not independent players | Infer from non-overlapping summaries; keep player-level paired comparisons exploratory. |
| Substitution is fatigue ground truth | Substitutions have multiple causes | Use substitution timing only as contextual proxy validation. |

Running output can vary with pacing and match circumstances even without a uniquely identifiable fatigue cause. This is supported by primary match-analysis research and should guide the interpretation throughout. [Bradley and Noakes, match-running fluctuations](https://pubmed.ncbi.nlm.nih.gov/23808376/).

## 3. Your hardware and whether training is possible

### Verified local information

Read-only inspection of this machine found:

| Component | Observed value | Consequence |
|---|---|---|
| Laptop | OMEN by HP Gaming Laptop 16-xd0xxx | Use plugged-in performance mode for sustained runs. |
| CPU | AMD Ryzen 7 7840HS with Radeon 780M Graphics | Suitable for decoding, tracking, metrics, and statistical analysis. |
| System memory visible to the OS | About 15.29 GiB, consistent with a nominal 16 GB system | Stream video; avoid keeping decoded clips or whole training datasets in RAM. |
| Discrete GPU | NVIDIA GeForce RTX 4060 Laptop GPU | CUDA inference and small-model fine-tuning are realistic. |
| GPU memory | 8,188 MiB reported by NVIDIA management tools | Use small batches and process detector/pose stages separately when needed. |
| NVIDIA driver | 616.56 | Verify compatibility using an actual PyTorch CUDA smoke test. |
| Workspace storage | About 249.7 GiB available on D: at inspection time | Enough for a selected subset, subject to actual file sizes and cache limits. |
| Existing Python configuration | .python-version = 3.12; pyproject.toml requires >=3.12 | Start with this environment and verify package compatibility before changing it. |

The NVIDIA query initially failed inside the restricted environment and succeeded during the subsequent hardware check. CUDA matrix multiplication, torchvision NMS and pretrained inference subsequently passed. One bounded 15-epoch detector experiment completed in approximately 258 seconds including evaluations. Power limits and thermals were not profiled; training/inference measurements are recorded in the delivered phase reports.

### Direct answer about training

**Use one small detector fine-tuning experiment; keep pose, tracking, and fatigue analysis free of additional model training.** Follow [the four-phase roadmap](FOUR_PHASE_IMPLEMENTATION_PLAN.md) for the execution order and limits.

Downloading long match recordings is different from training. The 40–80 GB project storage allowance covers footage, environments, caches, and outputs; it is not a training dataset.

Prepare about 600 diverse labeled images or native-resolution tiles: approximately 400 training, 100 validation, and 100 held-out test examples from a different match. Split by source scene/time before creating tiles; prevent neighboring frames or tiles from the same frame leaking across splits. Label every visible person for the single person class.

Fine-tune pretrained yolov8n.pt for 15 epochs initially, capped at 20, with 640-pixel tiles, batch 4, mixed precision, freeze=10, patience=5, workers=0, cache=False, and a fixed seed. Verify frozen/trainable layers and memory on the installed version. Keep compressed training-subset images under 1 GB; roughly 50–300 MB for the full 600-image subset is a planning estimate that depends on encoding and content.

Budget roughly 30–90 minutes for one small training run, then replace this unmeasured estimate with pilot timings. Label review, downloads, inference over long clips, and model/environment storage are separate costs. No automatic expansion to larger datasets or longer training is planned. [Official training settings](https://docs.ultralytics.com/modes/train).

Compare pretrained and fine-tuned YOLOv8n on the same validation set, lock the chosen detector, and evaluate on the untouched match. Pretrained YOLOv8s remains an untrained quality fallback. Retain pretrained weights if adaptation degrades validation performance; the small experiment need not improve accuracy to be honestly reported.

No pose-model, re-identification-model, or fatigue-classifier training is required. Time of match and substitution are not fatigue ground truth, so they must not become artificial fatigue labels.


### Open-source alternatives

| Option | Use in this project | Tradeoff |
|---|---|---|
| Ultralytics YOLOv8 n/s plus YOLOv8-Pose n/s | Recommended reproducible baseline, consistent with the brief | Review code and weight licenses; pin versions. |
| MMPose RTMPose, small top-down configuration | Alternate pose backend if the baseline fails the crop audit | More installation effort; isolate its environment and verify CUDA/package compatibility. |
| MediaPipe Pose Landmarker | Lightweight optional comparison on clean single-player crops | Do not assume Python GPU acceleration on Windows or reliable depth from model-inferred world landmarks. |
| ByteTrack | Default tracklet association | Long disappearances and same-kit players still need identity handling. |
| SoccerNet-GSR/TVCalib components | Calibration/benchmark reference for broadcast shots | Integrate only the required components rather than the whole research stack. |

MMPose provides pretrained pose configurations; MediaPipe exposes image and model-inferred world landmarks. Neither should be treated as a source of measured field position or validated monocular vertical motion. [MMPose repository](https://github.com/open-mmlab/mmpose), [MediaPipe Pose documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker).

Cloud GPUs are an optional overflow path, not a prerequisite. No paid API is required.

## 4. Dataset selection and access plan

### Primary: SoccerTrack v2

Use **SoccerTrack v2** for the main long-duration study. Its official release describes ten full matches with panoramic video, pitch-coordinate trajectories, player identifiers, and event annotations. This is a better fit for observing the same player across early and late match periods than short broadcast benchmarks. [Official project](https://atomscott.github.io/SoccerTrack-v2/).

The current canonical Hugging Face card identifies release v1.2, 20 half-match videos, a separate four-minute MOT subset, and calibration assets. Access requires logging in and accepting dataset conditions; the published license is CC BY 4.0. The complete repository is approximately 145 GB, so download selected matches. Do not confuse the four-minute MOT subset with the full-half videos. [Dataset and current release information](https://huggingface.co/datasets/atomscott/soccertrack-v2).

Important intake risk: the repository's format documentation explicitly records differences between documented and shipped annotation layouts, including nested attributes and image/frame metadata. Some pages disagree on timing and coordinate conventions. Inspect actual v1.2 files, pin the revision, and validate alignment on decoded frames rather than copying a schema example uncritically. [Maintainer's schema divergence notice](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md).

Implementation choices:

- Inspect one validation-match sample before downloading a second match.
- Confirm frame rate, true image dimensions, timestamps, annotation indexing, pitch orientation, and physical-team identity across halves.
- Verify whether the supplied image boxes cover the selected frames; annotations differ by task.
- Use supplied calibration/distortion assets where applicable. A stitched panoramic image may not obey one global pinhole homography.
- Assess local or per-view projection error; use calibrated views or local mappings if a single homography fails.
- Keep pose extraction on native-resolution target crops or suitable camera-view crops. Do not stretch a pitch-plane rectification into a human pose input.
- Treat GSR positions as reference annotations with their own error, not ground-truth fatigue or sensor-quality speed.

The official calibration workflow includes undistortion, camera mappings, and projection validation; adopt its coordinate spaces explicitly. [Calibration documentation](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/calibration.md).

### Secondary: SoccerNet Game State Reconstruction

Use 5–10 annotated SoccerNet-GSR broadcast clips to evaluate detection, short-term identity, and calibration under realistic camera motion. The original benchmark is built around 30-second sequences; these cannot independently support a 20–30-minute fatigue trend. [Dataset paper](https://arxiv.org/abs/2404.11335), [official development kit](https://github.com/SoccerNet/sn-gamestate).

This preserves the broadcast-video aspect as an explicit transfer/robustness experiment. Do not concatenate unrelated 30-second clips into a fake continuous player timeline.

For a genuinely broadcast-only longitudinal experiment, obtain continuous full-half video separately and use per-shot/temporal calibration and reviewed identity links. Budget additional development time; “no cuts anywhere for 30 minutes” is generally an unrealistic broadcast selection criterion.

### Fallbacks and their purpose

| Source | Appropriate use | Restriction |
|---|---|---|
| SoccerTrack original release | Movement validation with available video and GNSS annotations | Confirm synchronized duration; handle fisheye distortion and large files. |
| SoccerNet full-match video | Longitudinal broadcast extension | Access permissions, camera motion, resolution, and visibility must be audited. |
| SportsMOT football subset | Detector/tracker testing | It is a tracking benchmark, not fatigue-labeled longitudinal data. |
| Permissioned university/local-match recording | Long fixed-view experiment if primary access is delayed | Obtain consent and usable early/late footage; assess pose resolution before committing. |

Original SoccerTrack includes wide-view/drone video and GNSS, which could validate movement estimates where synchronized assets are available. [Original dataset repository](https://github.com/changjo/SoccerTrack). SportsMOT supplies multi-sport tracking annotations. [Official SportsMOT repository](https://github.com/MCG-NJU/SportsMOT). SoccerNet publishes separate access instructions for its video/task assets. [Official data access](https://www.soccer-net.org/data).

### Sampling that retains the intended scope

Target **three outfield players across two matches and four 25-minute segments**, approximately 100 minutes of distinct footage:

- Match A: early segment around 05:00–30:00 and late segment around 60:00–85:00; two selected players.
- Match B: corresponding early/late segments; one selected player.
- Adjust endpoints for recording coverage, substitutions, and player availability before analysis.
- Prefer players present in both periods. Record playing role and known substitution/context information.
- Add a fifth segment or additional match only after the principal four segments pass quality checks.

These are early-versus-late samples from different halves, not complete first-half/second-half averages. Label them accordingly. Full-half summaries can be added using additional processed footage; annotation-derived full-half summaries must be labeled as reference-only.

For true change-point localization, process a **continuous 25–30-minute interval** with enough valid windows. Early/late segments separated by an unprocessed interval permit a paired comparison but cannot reveal when a decline occurred inside that gap. A late segment alone cannot establish whole-match fatigue onset.

The pilot must establish that sufficient usable gait bouts exist. If panoramic crops are too small, preserve the longitudinal movement experiment and add closer-view, continuous running bouts for evaluating gait extraction separately. Report the different evidence sources explicitly.

## 5. Architecture, dependencies, and repository layout

The core pipeline:

video + manifest -> quality/camera segmentation -> detections -> tracklets -> reviewed player identity -> pose + calibrated ground positions -> frame/bout metrics -> time windows -> trends/change points -> validation and dashboard.

Separate expensive vision extraction from cheap analysis so that changing a window size does not rerun the GPU models.

Proposed structure:

~~~text
configs/
  pilot.yaml
  study.yaml
data/
  raw_videos/
  annotations/
  calibration/
  manifests/
  processed/
src/
  __init__.py
  cli.py
  video_io.py
  quality.py
  detect_track.py
  identity.py
  pose_extract.py
  calibrate.py
  metrics.py
  fatigue_analysis.py
  validation.py
  visualize.py
app/
  dashboard.py
scripts/
  check_environment.py
  prepare_dataset.py
  benchmark.py
notebooks/
  pipeline_demo.ipynb
tests/
results/
  metrics_csv/
  plots/
  annotated_videos/
  reports/
requirements.txt
requirements-dev.txt
pyproject.toml
README.md
LIMITATIONS.md
IMPLEMENTATION_PLAN.md
~~~

Dependency groups:

- Core vision: torch, torchvision, ultralytics, supervision, opencv-python.
- Analysis/storage: numpy, pandas, scipy, statsmodels, ruptures, pyarrow.
- Output: matplotlib, plotly, pyyaml, tqdm.
- Notebook/quality checks: jupyterlab, ipykernel, pytest.
- Optional adapters: huggingface_hub, SoccerNet, mediapipe.
- Optional UI: streamlit.

Install the appropriate CUDA-enabled PyTorch build using the official selector; do not assume plain pip installation selected CUDA. The setup checker must execute a small tensor operation on cuda:0 and report versions, device, and allocated memory. [PyTorch installation guide](https://pytorch.org/get-started/locally/).

Start with the repository's Python 3.12 setting. If an optional backend requires a different supported version, isolate it rather than disrupting the core environment. Lock a tested dependency set after the pilot; avoid unbounded installations in the final notebook. Install a single OpenCV distribution to avoid conflicting cv2 packages.

## 6. Implementation stages and intermediate outputs

### A. Dataset intake and time alignment

Build a manifest with dataset revision, source checksum, match ID, half, clip bounds, player identity, pitch dimensions/convention, calibration files, and known context.

Store source frame index, video timestamp, clip-relative seconds, half-relative seconds, and mapped match time. Use presentation timestamps when available; validate constant-frame-rate assumptions. Halftime is not physical movement time. Preserve stoppage-time labels and half identifiers instead of naively concatenating halves.

Stream frames. Save a frame index and selected QA images, not millions of full-size decoded JPEGs. Decode sequentially and retain the required samples.

Outputs: manifest.json, frame_index.parquet, intake_report.json.

### B. Camera, visibility, and quality assessment

Detect shot boundaries using image-change statistics, and mark close-ups, replays, overlays, and unsuitable views for review. Where available, use a pretrained shot detector as an optional adapter.

Estimate camera drift on pitch lines/static background features. Exclude players, spectators, graphics, and moving shadows from calibration feature matching. Never estimate camera movement from the player's own trajectory.

Outputs: camera_segments.csv, quality_events.csv, calibration/keyframe overlays.

### C. Detection, tracking, and persistent identity

Use pretrained yolov8s.pt as the initial detector; compare yolov8n.pt if needed. Detect people, filter off-pitch detections, and select outfield targets manually. The person class includes referees and staff, so it is not a player label.

Start with 12.5 fps on 25-fps video or 15 fps on 30-fps video. Test 25/30 fps if rapid motion creates association failures. Configure tracker age/buffers in seconds converted to the actual processing rate.

Use a low enough detection threshold to retain ByteTrack's low-confidence association candidates; expose higher thresholds for starting/displaying tracks. A single fixed 0.5 threshold can discard detections the tracker needs.

Preserve both tracklet_id and player_id. Use jersey-color descriptors as supporting evidence only. Team color does not identify a specific teammate. Combine plausible field motion, appearance, and optional pretrained re-identification features for candidate links.

After a shot cut, long disappearance, or ambiguous association, require a reviewed identity link or mark identity unknown. A small review queue is a core feature. Initial target selection alone cannot guarantee correct unattended identity over arbitrary broadcast video.

Outputs: detections.parquet, tracks.parquet, identity_links.csv, gaps.csv, annotated audit video.

### D. Pose extraction

Run yolov8s-pose.pt on padded target crops, in small batches. YOLOv8 pose provides COCO-format joints; keep shoulders, hips, knees, and ankles with individual confidence values. [YOLOv8 model documentation](https://docs.ultralytics.com/models/yolov8), [pose interface](https://docs.ultralytics.com/tasks/pose).

Use the native 25/30 fps during eligible running bouts. Prefer processing continuous bout spans to irregular sparse samples for cadence. At intermediate frames, verify the crop through a lightweight target-tracker or detection refresh; crop propagation must stop if identity is uncertain.

Map crop coordinates back to the source image, undoing crop offsets, resizing, and letterboxing. Associate multiple pose outputs with the selected box; do not assume the first pose belongs to the target.

Record low-confidence keypoints as missing. Upsampling a small crop does not create anatomical information. Select resolution/confidence gates from manually audited examples and report pose availability.

Outputs: player_keypoints.parquet and a readable per-clip keypoints.csv, pose_quality.csv, skeleton audit video.

### E. Calibration and position reconstruction

For static conventional camera views, select at least four non-collinear correspondences; prefer 8–12 well-spread visible line intersections. Use actual pitch dimensions when known, otherwise record the 105 x 68 m approximation. A center circle has no literal corners.

Fit image-to-pitch mapping with RANSAC and maintain inverse mapping for overlays. Report held-out line/point reprojection error and validity over time. Four fitting points alone cannot validate generalization.

For panoramic inputs, use the supplied camera model/remapping or a locally validated mapping. Keep distorted source pixels, calibrated view pixels, and metric pitch coordinates separate. Never apply the same homography to mismatched image spaces.

Ground anchor hierarchy:

1. Reliable visible support-foot/contact estimate.
2. A validated feet midpoint when contact is uncertain, with lower confidence.
3. Bounding-box bottom center as a fallback, explicitly tagged.

All anchors are approximate. Flight phases, partial occlusions, and cropping can displace them. Compare anchor alternatives against reference positions and smooth only within valid segments.

For broadcast shots, initialize calibration at a keyframe and propagate/re-estimate using pitch-line/background correspondences. Reinitialize at cuts or failed checks. Invalid calibration means unavailable metric speed, not zero speed. Pretrained TVCalib components can be added if this simpler propagation is insufficient.

Outputs: calibration assets, calibration_quality.csv, positions_meters.parquet.

### F. Frame and running-bout features

Identify continuous valid running bouts, initially at least 4 seconds, excluding turns, jumps, tackles, and severe occlusion from gait analysis. Retain those activities in movement summaries when position quality permits.

A speed threshold alone cannot reliably label activity; combine speed, continuity, heading stability, and pose checks. Compare gait at similar speeds and view directions.

Extract metrics using the definitions in Section 7. Persist metric-specific quality flags and valid durations. Do not smooth across identity switches, camera cuts, halftime, or missing observations.

Outputs: frame_metrics.parquet, running_bouts.csv, gait_cycles.csv.

### G. Window aggregation

Use 120-second windows stepped every 60 seconds for visual exploration. Also create non-overlapping 120-second windows for primary trend/change-point analyses and 5-minute summaries for sensitivity analysis.

Initial eligibility defaults, to revise only on pilot/development data:

- Locomotion metrics: at least 70% valid calibrated time in a window.
- Gait metrics: at least 20 seconds of usable running, distributed across valid bouts, and at least 10 accepted same-leg cycles.
- Baseline: early 10-minute elapsed interval with adequate observations of the relevant activity/view; never silently extend it until enough favorable samples appear.
- Missing or near-zero baselines: suppress percentage normalization for that metric and explain why.
- Per-metric eligibility: usable movement metrics do not imply usable joint angles.

Record valid_seconds, running_seconds, valid_cycles, camera_quality, pose_quality, and metric availability alongside each window. Missing measurements remain missing.

Outputs: {clip}_{player}_windowed_metrics.csv and eligibility summaries.

### H. Longitudinal analysis and visualization

Run descriptive trends, autocorrelation-aware uncertainty, and supported change-point searches. Distinguish within-segment change from early/late comparisons. Export source window IDs for every detected candidate.

Produce the requested plots and a lightweight dashboard after CSVs and static figures are reliable. The notebook orchestrates existing modules; it should not duplicate pipeline logic.

## 7. Metrics and biomechanical interpretation

| Metric | Definition and unit | Interpretation and safeguards |
|---|---|---|
| Average movement speed | Smoothed ground-path distance / valid observed time, m/s | Measures observed work rate, with coverage disclosed. |
| Running speed envelope | Median and 90th percentile within qualified running bouts, m/s | More informative than a noisy single-frame maximum; still influenced by opportunity. |
| Peak speed | Highest sustained approximately 1-second mean, m/s | Keep the requested peak but reject isolated tracking spikes. |
| High-intensity distance rate | Distance above a declared speed threshold / valid observed minutes | Adds intensity information; report observed exposure and threshold sensitivity. |
| Cadence | Accepted same-leg cycles per running second, strides/s; step cadence reported separately as steps/min | Reduced cadence is not universally fatigue; compare at similar speeds. |
| Stride-length proxy | Integrated ground-path distance between consecutive same-leg cycle events, m/stride | Requires reliable event timing and position; avoid mapping swing ankles through the pitch transform. |
| Vertical-motion proxy | Detrended hip image-height residual relative to a locally stable ground anchor, divided by robust body/torso scale | Dimensionless projected oscillation, affected by view and contact-anchor noise; not vertical displacement in meters. |
| Knee flexion excursion | 180 degrees minus hip-knee-ankle internal angle; robust 95th–5th percentile excursion per cycle, degrees | A 2D projection proxy; compare matched speed/view and report left/right separately. |
| Left/right cycle asymmetry | Absolute difference between leg summaries divided by their mean, % | Exploratory; perspective and occlusion can imitate asymmetry. |

Speed computation:

- Use actual timestamp differences, not an assumed universal frame interval.
- Apply an offline robust position smoother over uninterrupted segments before differentiation.
- Integrate path increments over valid observed intervals; do not join positions across gaps.
- Reject implausible jumps with recorded reasons. Do not silently clip speeds to make plots look reasonable.
- Check sensitivity to smoothing strength because smoothing can remove real peaks.
- Label offline centered filters as offline; a future live mode would need causal processing.

Cadence and stride events:

- Start with side-view leg-angle signals and ankle positions relative to hips; choose a usable signal per bout.
- Remove low-frequency translation and assess periodicity using peaks plus autocorrelation/spectral agreement.
- Enforce minimum sample rate, cycle count, and continuity.
- Count same-leg cycles as strides. Alternating left/right events are steps; they differ by approximately a factor of two in regular gait.
- Treat event timestamps as estimated cycle/contact proxies. Image-y extrema alone do not establish foot strike.
- Median speed divided by stride frequency is an alternate stride-length estimate, but is mathematically coupled to speed and cadence. Do not count it as independent corroborating evidence.

For vertical motion and knee excursion, use cycle summaries rather than the extrema of an entire two-minute window. Turning or one bad keypoint can otherwise dominate the result.

Retain shoulder/torso lean and acceleration/deceleration as additional experimental features only after position and pose validation. They are sensitive to projection and differentiation noise.

## 8. Decline detection and uncertainty

### Baseline and confounding controls

For each player, use an early-match reference interval with documented valid coverage. If recording begins late, call it an observed early-segment baseline, not a fresh/unfatigued baseline.

Percentage change is 100 * (window_value - baseline_value) / baseline_value. Use matched movement/view strata for gait; skip percentages when the reference is close to zero. Maintain units alongside normalized values.

Show work-rate change and speed-matched gait change separately. Reduced sprint exposure can reflect fewer attacking opportunities. A gait change observed only because the player slowed down is not independent evidence of fatigue.

### Trend analysis

- Fit an interpretable linear slope per metric on eligible non-overlapping windows.
- Use robust/HAC standard errors or a justified time-block bootstrap where enough contiguous observations exist.
- Report sample count, elapsed span, slope, uncertainty interval, and missingness.
- Mark very short series as descriptive only; asymptotic p-values with a handful of windows are unreliable.
- If multiple significance tests are reported, identify them as exploratory and apply an appropriate multiple-testing correction.
- A negative slope is not an automatic fatigue decision. Direction for gait changes must be prespecified or treated as a two-sided change.

### Change-point analysis

Use ruptures.Pelt with a simple mean-shift cost as the baseline. Normalize scale, enforce minimum segment length, and evaluate a declared penalty range on development data.

Initial gate: at least 12 eligible non-overlapping windows in an uninterrupted analyzable run, and at least four windows on each side of a candidate. Increase sample length if the pilot supports it. Do not compress scattered valid windows into an apparently continuous series.

Check candidate robustness to window size, smoothing, and penalty. Exclude candidates explained by camera/identity-quality changes. Convert the window boundary to a match-time interval with the window's temporal uncertainty; do not claim frame-accurate onset.

Explicit statuses:

- candidate_decline
- stable_or_no_supported_change
- insufficient_data
- quality_confounded
- baseline_unavailable

A terminal ruptures endpoint is not a detected internal change point. Reject end-of-series artifacts.

### Explainable evidence summary

Provide a movement-decline flag and, when available, separate gait-change evidence. An optional composite score can summarize standardized changes with fixed documented weights, but:

- It is a heuristic index, not a probability of fatigue.
- Quality is shown separately, rather than interpreted as physiology.
- Missing inputs remain visible; do not silently change score weights per window.
- Coupled quantities such as speed and speed/cadence-derived stride length cannot count as independent evidence.
- Require persistence across non-overlapping windows and agreement across genuinely distinct evidence families before describing a multi-feature fatigue-related candidate.

A sustained 10% change can be explored as a demonstration threshold; it is not a clinically validated cutoff. Lock choices before evaluating the final held-out match.

## 9. Validation design

### Layer 1: Verify the computer vision measurements

- Detection: precision/recall on an audited annotated subset, stratified by target size.
- Identity: target identity purity, switches, fragmentation, and visible-time coverage. Calculate IDF1/HOTA where labels and adapters support them.
- Calibration: held-out reprojection error, pitch-line overlay alignment, and position error against matched reference annotations.
- Motion: trajectory error and speed sensitivity to smoothing/frame rate. Reference speed derived from annotations also contains differentiation error.
- Pose: annotate a small, separately held-out crop set and report normalized keypoint error/PCK for hips, knees, and ankles.
- Cadence: manually count cycles on held-out running bouts and report strides/s error plus accepted-bout coverage.
- Quality: disclose how much footage was rejected for each metric.

Suggested audit budget: 100–200 target crops across several sizes/views and at least 20 running bouts spread across players and early/late periods. Keep development examples separate from the final audit. Do not select evaluation crops solely because their skeletons look clean.

### Layer 2: Test the analysis independently

Use synthetic time series with known no-change, gradual-decline, and abrupt-change cases to verify behavior under noise and missingness. These validate analysis logic, not real fatigue detection.

Required cases include no forced change point on a constant series, no speed jump across an identity switch, and no interpolation across a camera cut. Include a synthetic camera-motion case in calibration checks.

### Layer 3: Perform proxy match validation

- Pair each player's early and late observations within the same match.
- Report valid exposure, absolute difference, percentage difference, and uncertainty.
- Use player/match-level summaries as the pairing unit; do not pretend hundreds of frames are hundreds of subjects.
- With only three players, make descriptive effect estimates the main result. An exact two-sided signed-rank test with three nonzero pairs cannot reach p < 0.05; small-sample significance is not a success criterion.
- If paired t-tests are shown, discuss their fragile assumptions at this sample size. Repeated appearances of one player do not create independent subjects.
- Use a substitution lead-time comparison only when identity and timing are verified and the candidate lies within the analyzed continuous interval.
- Report known stoppages, role changes, and match context that might explain a decline.
- Label the result proxy validation: none of these datasets supplies physiological fatigue labels for this task.

If an academic supervisor requires a true fatigue-validation extension, design a separate consented field study with an independently chosen reference such as standardized performance testing or exertion assessment. This is a new data-collection study and should not be assumed available.

## 10. Valuable features to include

### Core additions that materially improve the project

1. **Measurement-quality timeline:** tracking, identity, pose, calibration, and missingness shown beside every metric.
2. **Player identity review queue:** inspect uncertain reappearances instead of silently assigning a teammate's movements.
3. **Pitch minimap and observed-distance trail:** visually explains homography and reveals impossible jumps.
4. **Annotated video:** player box, skeleton, confidence, current valid speed, match time, and explicit unavailable states.
5. **Speed/view-matched gait comparison:** reduces the strongest confounders in early/late mechanics.
6. **Per-player baseline profile:** supports individual normalization without requiring a trained fatigue classifier.
7. **Reproducible run manifest:** configuration, dataset revision, weight hashes, package versions, and excluded intervals.
8. **Sensitivity report:** change estimates under alternate sampling rates, window sizes, smoothing, and change-point penalties.

### Dashboard after the core outputs work

A local Plotly/Streamlit view with clip/player selection, synchronized metric timelines, candidate boundaries, baseline percentage changes, half/period comparisons, a minimap, and links to audit frames/video. Use explicit labels for proxy metrics and insufficient-data states.

### Stretch features with clear entry conditions

- Complete the bounded detector experiment in Phase 2; additional training is outside the agreed budget.
- Optional pretrained re-identification only if audited identity breaks are a major error source.
- Left/right asymmetry and torso lean only when pose reliability supports them.
- Acceleration-speed envelopes only with validated dense ground trajectories; related recent work motivates this direction, but it is not required for delivery.
- Jersey-number recognition as supporting identity evidence, never an unattended identity oracle.

Avoid adding a fatigue LSTM/Transformer trained on time-of-match labels. That would add model complexity without resolving label validity.

## 11. GPU, memory, and runtime budget

Use two cached passes: detect/track the whole selected interval, then process target poses only on eligible bout spans at native frame rate. Keep a simple option to process all visible target poses for the small demo and comparison.

For four 25-minute segments:

- At 15 fps, approximately 90,000 detector frames.
- With two simultaneously selected targets, 30 fps pose, and 30% eligible-bout coverage, approximately 108,000 target-crop pose evaluations. This is an upper planning example; the proposed study has only one target in Match B.
- At 25 fps, the corresponding pose count is lower. Variable coverage must come from measured footage, not this estimate.

Benchmark 60–120 seconds of representative video and record decoding time, detection time, pose time, peak VRAM, CPU RAM, and accepted-bout rate. Estimate total runtime from the measured stage counts and stage throughput. Do not promise real-time performance or fixed training hours in advance.

Initial resource settings:

- Detector batch 1–4, pose batch 4–8, adjusted after a memory smoke test.
- Start at a detector size that preserves player visibility; compare full-frame inference with overlapping native-resolution tiles when distant people become too small.
- Panoramic 4K frames should not simply be shrunk to 640 pixels if target recall collapses.
- Sequential GPU stages, mixed precision where supported, bounded decoder queues.
- No full-dataset RAM cache on the 16 GB system.
- Parquet for dense tables; readable CSV exports for deliverables.
- Preserve original compressed video; cap debug frame/video caches.
- Reserve roughly 40–80 GB for the selected study plus environments/caches, then revise from actual file sizes. This is a planning allowance, not a measured subset size.

If memory pressure occurs, reduce batch size or process tiles sequentially before reducing spatial detail or gait sampling. That preserves the scientific scope.

## 12. Output contracts and reproducibility

Required data tables:

| Artifact | Essential fields |
|---|---|
| frame_index | clip, source frame, timestamp, half, mapped match time |
| tracks | frame, box, detector confidence, tracklet ID, player ID, identity confidence |
| keypoints | frame, player, joint, source-pixel x/y, confidence, crop transform, quality flag |
| positions | timestamp, player, metric x/y, anchor type, calibration ID, validity |
| gaps | player, start/end frame/time, reason, affected metrics |
| running_bouts / gait_cycles | player, bounds, speed/view class, cycle events, confidence |
| windowed_metrics | original requested metrics plus units, observed durations, cycle counts, eligibility |
| trends / change_points | metric, slope/uncertainty, candidate interval, settings, evidence status |
| period_comparison | player/match, early/late means, exposure, differences, optional exploratory tests |

Keep the original requested column names where useful, but add explicit unit/proxy columns. For example, vertical_oscillation must carry unit=normalized_image_proxy; do not label it meters. Include stable source IDs so CSV and dashboard rows can be traced to footage.

Stage reruns should reuse valid caches keyed by source/config/model versions. A changed homography invalidates positions and downstream metrics; it need not invalidate detector outputs.

Docstrings should explain the purpose, formula, units, assumptions, missing-data behavior, and relation to fatigue-related interpretation. Do not attach biomechanical claims to trivial utility functions.

## 13. Proposed configuration and user workflow

Illustrative config; all interfaces below are planned, not existing commands:

~~~yaml
dataset: soccertrack_v2
dataset_revision: "<pinned-v1.2-commit>"
video: data/raw_videos/match_A_first_half.mp4
clip:
  start_seconds: 300
  duration_seconds: 1500
  half: 1
  match_time_offset_seconds: 0
targets:
  - player_id: match_A_team1_number8
    initial_tracklet_id: 7
sampling:
  tracking_fps: 12.5
  pose_fps: native
  pose_mode: running_bouts
models:
  detector: yolov8s.pt
  pose: yolov8s-pose.pt
  device: cuda:0
calibration:
  mode: dataset_camera_model
  reference: data/calibration/match_A.json
analysis:
  window_seconds: 120
  display_step_seconds: 60
  inference_step_seconds: 120
  baseline_elapsed_seconds: 600
quality:
  min_position_coverage: 0.70
  min_running_seconds: 20
  min_valid_cycles: 10
~~~

Workflow to implement:

~~~powershell
python scripts/check_environment.py
python -m src.cli inspect --config configs/pilot.yaml
python -m src.cli select-player --config configs/pilot.yaml
python -m src.cli calibrate --config configs/pilot.yaml
python -m src.cli extract --config configs/pilot.yaml
python -m src.cli review-identity --config configs/pilot.yaml
python -m src.cli analyze --config configs/pilot.yaml
python -m src.cli report --config configs/pilot.yaml
streamlit run app/dashboard.py
~~~

After setup, reviewed identity links, and calibration are saved, rerunning the same segment should be unattended and deterministic within the documented model/runtime limits. Arbitrary new broadcast footage may require additional identity/calibration review.

## 14. Four-phase implementation schedule

Execution follows [FOUR_PHASE_IMPLEMENTATION_PLAN.md](FOUR_PHASE_IMPLEMENTATION_PLAN.md). The numbered sections in this document are technical references, not additional delivery phases.

| Phase | Effort estimate | Work | Completion evidence |
|---|---|---|---|
| 1. Setup, data, and feasibility | 3–5 working days | CUDA checks, one-half intake, timestamps, crop audit, 1–2 minute benchmark | Usable detection/pose and a viable calibration path |
| 2. Minimal fine-tuning, tracking, and calibration | 7–10 working days | About 400 training images plus held-out splits; one detector run capped at 20 epochs; tracking, identity, gaps, calibration | Baseline comparison and an auditable ten-minute trajectory pilot |
| 3. Pose and movement/gait metrics | 5–7 working days | Pretrained pose, running bouts/cycles, metrics, quality windows, full study extraction | Held-out crop/bout checks and traceable metric tables |
| 4. Analysis, validation, and final delivery | 5–7 working days | Baselines, trends/change points, early/late comparison, robustness, plots, dashboard, notebook, documentation | Reproducible reports with valid uncertainty and no-change support |

Allow approximately 5–6 focused weeks overall, including integration/correction buffer; dataset approval/download delays are external. The single small detector experiment is the entire planned training workload, not training on the full video collection.

Dependencies: confirm footage/pose suitability before expanding downloads; verify identity and calibration before speed; verify cycles before stride length; validate measurements before interpreting decline. Increase data volume after the ten-minute pilot passes.


## 15. Acceptance criteria

The project is complete when:

- One configured sample segment runs end to end after initial calibration/selection and any required identity review has been saved.
- Three selected players across four long segments have exported window tables, including explicit missing/invalid measurements.
- Each player has early/late comparisons where sufficient paired evidence exists, with observed exposure and appropriate labels.
- At least four metric-over-time plots are saved for adequately measured signals, with gap and quality annotations. Missing metric panels explain insufficiency instead of fabricating values.
- Validated gait bouts demonstrate cadence, stride-length proxy, vertical-motion proxy, and knee excursion extraction; their long-clip availability is quantified.
- Every analyzed metric has an explicit change-analysis status. A supported candidate is marked when present; a no-change result is acceptable.
- A synthetic known-change series produces a localized change under documented settings, and a constant/noisy no-change control is not forced to produce one.
- Detection/identity/calibration/pose/cadence quality is quantified on held-out audited samples.
- A short broadcast robustness experiment documents what degrades relative to the main full-pitch study.
- A minimap, annotated video, period-comparison table, dashboard or interactive HTML report, and runnable notebook are delivered.
- README and LIMITATIONS document setup, actual hardware, metric meanings, coordinate assumptions, missingness, confounding, small sample size, and proxy-only validation.
- No claims of measured 3D vertical oscillation, physiological fatigue accuracy, or population-level significance are made without corresponding evidence.

A scientifically valid negative result still satisfies the project: reliable measurement with no supported decline is better evidence than a forced fatigue timestamp.

## 16. Main risks and responses

| Risk | Response that preserves useful scope |
|---|---|
| Dataset access delayed | Begin on public benchmark samples and analysis controls; arrange permissioned fixed-view footage in parallel. |
| Too few pixels for ankles/knees | Use original-resolution crops, tiles, or closer-view gait clips; keep long-duration movement study and quantify gait coverage. |
| Panoramic geometry fails a global homography | Use supplied distortion/camera mappings or locally validated views; compare held-out projection errors. |
| Broadcast camera movement creates false speed | Segment shots, update calibration, gate invalid frames, and show the movement-quality timeline. |
| Same-kit identity switches | Reviewed identity links and unknown states; no silent reassignment. |
| Few valid gait windows | Report bout-level gait comparisons and coverage; do not interpolate gait through missing play. |
| No fatigue labels | Evaluate measurement accuracy, synthetic change recovery, and contextual proxy comparisons separately. |
| No clear decline | Report stable/uncertain results and sensitivity analysis; do not lower thresholds after seeing final outcomes. |
| Laptop memory/thermal limits | Cache stages, stream video, batch conservatively, and run long jobs plugged in; use cloud only if measured needs justify it. |

## 17. Immediate next actions

1. Create an isolated environment and verify an actual CUDA tensor operation and one detector/pose forward pass.
2. Request/access the canonical SoccerTrack v2 release and pin its revision; inspect one validation match before bulk download.
3. Audit source dimensions, time alignment, camera model, cross-half identity, and 20–30 target crops.
4. Run a 1–2 minute benchmark and manually count gait cycles in several running bouts.
5. Lock the four-segment/three-player selection and preprocessing settings after the pilot.
6. Implement and validate trajectory extraction, followed by pose/bouts, then analysis and dashboard.
7. In Phase 2, run the single bounded detector experiment on about 400 training images, starting at 15 epochs and capped at 20; keep pose and analysis pretrained/statistical.

The recommended balance is a substantial, locally runnable computer-vision system whose measurements and uncertainty are inspectable, with a defensible fatigue-related interpretation and a realistic path to stronger validation.
