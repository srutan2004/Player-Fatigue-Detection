# Four-Phase Implementation Plan: Player Movement and Fatigue-Related Decline

Updated: 14 October 2026

Course: BCSE407L Computer Vision, RIL

Status: all four phases implemented and executed. See [FINAL_REPORT.md](FINAL_REPORT.md), including unavailable baselines, insufficient gait exposure and no supported fatigue-onset claim. Software completion is separate from physiological validation.

This is the execution roadmap. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) remains the detailed technical reference.

## Project scope and the small training budget

Keep three outfield players across two matches and four approximately 25-minute segments. Retain player tracking, pose estimation, field calibration, movement/gait metrics, trend and change-point analysis, early/late comparison, annotated video, a minimap, and a dashboard.

**Downloading footage and training a model are different activities.** The long match recordings are mainly inputs to analyze. The earlier 40–80 GB allowance covered video, environments, caches, and outputs; it was never a required training dataset.

The revised plan includes one small detector fine-tuning experiment:

| Item | Revised limit |
|---|---|
| Total selected images/tiles | About 600 |
| Actual training split | About 400 labeled images/tiles |
| Validation split | About 100 separate images/tiles |
| Final test split | About 100 images/tiles from a different match |
| Model to fine-tune | Pretrained YOLOv8n person detector |
| Training length | Start at 15 epochs; hard limit of 20 |
| Pose model | Pretrained; no fine-tuning |
| Tracker | ByteTrack; no training |
| Fatigue analysis | Statistical rules/trends/change points; no fatigue classifier training |
| Image dataset storage | Plan for roughly 50–300 MB for compressed 640-pixel tiles; measure actual size, and keep under 1 GB |
| Training time | Budget roughly 30–90 minutes for the one small run; an unmeasured planning allowance, not a hardware benchmark |

Model weights, checkpoints, and Python environments consume additional storage; they are separate from the image dataset. Label preparation is also separate from GPU training time and can take longer than the run.

A small dataset is sufficient for a modest domain-adaptation experiment, not a guarantee of improved accuracy. Choose the final detector using validation results, and retain pretrained weights if fine-tuning makes detection worse. Do not increase the training dataset or run budget automatically.

The previously verified laptop has an RTX 4060 Laptop GPU with approximately 8 GB VRAM, a Ryzen 7 7840HS, and approximately 16 GB RAM. It is a suitable target for this bounded experiment.

## Phase 1 — Setup, footage selection, and a short feasibility pilot

**Goal:** prove that the footage contains enough detail and that the environment can run the models.

**Estimated implementation effort:** 3–5 working days, excluding external dataset-access delays.

### Work

1. Set up an isolated Python environment and test a CUDA tensor operation and pretrained detector/pose forward passes.
2. Access SoccerTrack v2, pin the downloaded dataset revision, and inspect its real annotation schema.
3. Download one half initially and watch it normally as an MP4. Match 117093's first half was listed at approximately 3.35 GB; its two halves together were approximately 6.83 GB at research time. File sizes are examples, not a mandatory selection. [Video files](https://huggingface.co/datasets/atomscott/soccertrack-v2/tree/main/videos/117093).
4. Select a 1–2 minute pilot with running, some crowding, and near/far players. Audit 20–30 target crops for usable hips, knees, and ankles.
5. Check source resolution, timestamps, frame rate, calibration coordinate spaces, and annotation/video alignment. Inspect the actual release instead of trusting an unverified example schema.
6. Run pretrained baselines and record runtime, VRAM/RAM use, detection misses, and pose availability.
7. Select the early/late segments and three players after the pilot. Download the second match only when needed for held-out evaluation and the longitudinal study.

The principal experiment uses full-pitch panoramic footage. Short SoccerNet-GSR broadcast clips are a separate robustness experiment; do not join them into a fake continuous player timeline.

### Outputs

- Environment report and pinned dependencies.
- Source manifest with match, half, timestamps, clip bounds, dataset revision, and player identity.
- Pilot audit images/video and a runtime estimate.
- Chosen study segments and train/validation/test source blocks.

### Completion check

One representative sample produces visible detections and usable target poses, and the camera geometry has a viable calibration path. If distant joints are too small, test native-resolution crops/tiles or closer-view gait footage before processing the long study.

## Phase 2 — Minimal detector training, reliable tracking, and calibration

**Goal:** follow the correct player and obtain trustworthy ground positions, while keeping training small.

**Estimated implementation effort:** 7–10 working days, including label review. GPU training occupies only a small part of this phase.

### A. Prepare the small labeled dataset

1. Select about 600 diverse source images or native-resolution tiles rather than every frame of a video.
2. Use about 400 for training and 100 for validation from disjoint temporal blocks. Put about 100 test examples in a different match.
3. Sample different player sizes, lighting, locations, poses, and occlusions. Use available annotations to seed boxes, then inspect and correct them.
4. Split source frames/scenes before generating tiles or augmentations. Keep all tiles from one frame in one split. Leave a temporal buffer, initially at least 30 seconds, between development splits where possible.
5. Use a single class named person and label every visible person in selected tiles, including referees. Apply the pitch/player filters afterward; otherwise unlabeled people become misleading negatives.
6. Crop or tile the native image before resizing. Shrinking an entire panoramic frame to 640 pixels can make distant players disappear.
7. Save a deterministic split manifest and annotation-quality audit.

The detector learns football-domain person appearance. It is not learning fatigue or individual player identity.

### B. Run one bounded fine-tuning experiment

Starting configuration used for the completed bounded experiment:

~~~yaml
model: yolov8n.pt
data: data/detector_subset/dataset.yaml
epochs: 15
imgsz: 640
batch: 4
device: 0
amp: true
freeze: 10
patience: 5
workers: 0
cache: false
seed: 42
~~~

Start from pretrained weights. Freezing early layers limits how much is adapted. Confirm the freeze setting and trainable parameter count in the installed version. Begin with a small memory smoke test; reduce batch if necessary.

Use 15 epochs initially, with an absolute ceiling of 20. If a wall-clock limit is implemented, stop at a completed epoch once the approximately 90-minute budget is reached. Do not treat a time budget as a promised runtime. No model search or repeated long experiments are required.

Compare the fine-tuned YOLOv8n against the same pretrained YOLOv8n using identical validation settings. A pretrained YOLOv8s can remain a quality fallback without being trained. Report recall/precision and detection error by target size. Choose the model on validation; evaluate the locked choice and baseline on the held-out match without tuning to test results.

[Official training settings, including freezing, precision, early stopping, and workers](https://docs.ultralytics.com/modes/train).

### C. Build tracking and identity handling

- Detect at about 12.5 fps for 25-fps input or 15 fps for 30-fps input; increase if the pilot reveals association failures.
- Use ByteTrack with low-confidence candidates available for association.
- Select the target and maintain temporary tracklet IDs separately from persistent player identity.
- Log losses, low-confidence intervals, and ambiguous reappearances.
- Review uncertain identity links. Same-colored jerseys are supporting evidence, not individual identity.
- Detect cuts and unsuitable camera views; do not bridge them with invented movement.

### D. Calibrate the field

- Use supplied camera/distortion mappings for panoramic footage where appropriate.
- For suitable conventional views, fit a homography using distributed field correspondences and validate on held-out points/lines.
- Use a reliable ground-contact/feet anchor or tagged box-bottom fallback for pitch position.
- Keep hips in image space for gait; projecting elevated hips through a field homography does not give a correct ground position.
- Update calibration under camera motion, or mark metric positions unavailable when calibration fails.
- Produce a pitch minimap and overlay to catch impossible jumps.

### Outputs

- A small training subset, corrected labels, and split manifest.
- Fine-tuned weights and a baseline-versus-fine-tuned report.
- Detection/track tables, reviewed identity links, gaps log.
- Calibration assets, quality scores, metric positions, and minimap.
- An auditable approximately ten-minute pilot.

### Completion check

Identity and position errors are inspectable; calibration failures and tracking gaps do not become false speed spikes. The training experiment is documented even if pretrained weights remain the best choice.

## Phase 3 — Pose extraction and movement/gait metrics

**Goal:** turn valid observations into interpretable measurements without additional model training.

**Estimated implementation effort:** 5–7 working days.

### Work

1. Run pretrained YOLOv8-Pose on padded target crops. Restore source-image coordinates correctly after resizing/cropping.
2. Use native 25/30 fps on suitable continuous running bouts. Keep low-confidence joints missing rather than inventing positions.
3. Identify straight, sufficiently long running bouts. Exclude severe turns, jumps, tackles, and occlusions from gait measurements.
4. Compute the following:

| Metric | Meaning | Required qualification |
|---|---|---|
| Average and sustained peak speed | Movement rate on the field | Calibrated ground positions and valid elapsed time |
| High-intensity running rate | Fast-running distance per valid observed minute | Declared threshold and observed coverage |
| Cadence | Same-leg running cycles per second | Continuous high-rate pose and credible periodicity |
| Stride-length proxy | Ground-path distance per same-leg cycle | Reliable cycle events and ground trajectory |
| Knee excursion | Robust knee flexion change per cycle | Reliable 2D joints and comparable viewing conditions |
| Vertical-motion proxy | Normalized projected hip oscillation | Image-space proxy; never label as vertical meters |

5. Compare gait at similar speeds and view directions. Do not interpret a slower gait alone as deteriorating mechanics.
6. Smooth only within valid uninterrupted segments. Do not compute displacement across gaps or identity switches.
7. Aggregate into 120-second windows: 60-second steps for display, non-overlapping windows for primary inference.
8. Add per-metric eligibility: initially 70% valid calibrated coverage for movement; 20 seconds of usable running and ten accepted same-leg cycles for gait.
9. Analyze the full four-segment/three-player study only after the ten-minute pilot is verified.

### Outputs

- Source-coordinate keypoints and per-frame quality flags.
- Running-bout and gait-cycle tables.
- Per-frame movement features and tidy windowed metric CSVs.
- Skeleton audit video, coverage report, and exclusion reasons.

### Completion check

Manually check representative joints and count cycles on held-out running bouts. Every metric has units, validity, and observed exposure. A measurable speed signal does not automatically make gait reliable.

## Phase 4 — Decline analysis, validation, dashboard, and final delivery

**Goal:** produce an explainable result with trustworthy limitations.

**Estimated implementation effort:** 5–7 working days, followed by a small correction buffer.

### A. Analyze sustained changes

- Compare eligible observations against the player's early ten-minute elapsed baseline.
- Mark unavailable or near-zero baselines explicitly.
- Fit descriptive trends with uncertainty appropriate to the available, autocorrelated data.
- Run change-point detection on eligible continuous sequences with enough non-overlapping windows, not across unobserved intervals.
- Require adequate pre/post evidence and inspect sensitivity to smoothing, window size, and penalty.
- Return supported candidate, no supported change, insufficient data, quality-confounded, or unavailable baseline.

Early and late clips permit a paired comparison; they cannot reveal exactly when a change occurred in the unprocessed time between them. Do not force a fatigue timestamp for every metric.

### B. Validate each layer

- Report detection, identity, calibration, pose, and cadence quality separately.
- Test known-change and no-change synthetic series.
- Compare each player's early and late summaries, including observed exposure.
- Use substitution timing only as contextual evidence when verified.
- Keep small-sample statistics exploratory; three players cannot support strong population-level conclusions.
- Include 5–10 short SoccerNet-GSR clips for broadcast robustness if accessible, without expanding the training budget.
- State clearly that movement decline can reflect tactics, pacing, and opportunities; physiological fatigue is not directly measured.

### C. Deliver the complete project

- At least four valid metric-over-time plots with gaps, quality, and supported candidate boundaries.
- Early/late comparison table and summary plots.
- Annotated video with box, skeleton, valid speed, and unavailable states.
- Pitch minimap and observed trajectory.
- Local Plotly/Streamlit dashboard or interactive HTML report.
- Runnable end-to-end notebook and command-line workflow.
- README, LIMITATIONS, configuration files, and reproducibility manifest.
- Compact training report: selected data volume, settings, runtime, and baseline comparison.

### Completion check

The notebook reruns after initial player/calibration choices and saved identity review. Results remain honest about missingness and proxy measurements. No supported decline is an acceptable scientific result.

## Timeline and data boundaries

| Phase | Effort estimate | Training? | Main milestone |
|---|---|---|---|
| 1. Setup and pilot | 3–5 working days | None | Footage, models, and geometry are viable |
| 2. Tracking, calibration, minimal fine-tuning | 7–10 working days | One small detector run only | Correct identity and reliable ground positions |
| 3. Pose and movement metrics | 5–7 working days | None | Valid keypoint, bout, and window tables |
| 4. Analysis and delivery | 5–7 working days | None | Validated report, plots, notebook, dashboard |

Allow approximately 5–6 focused weeks overall, including integration/correction time and excluding external approval delays. Most work is programming, annotation review, and measurement validation rather than training.

The complete study still analyzes about 100 minutes of selected footage. Only about 400 selected images/tiles update model weights. Long-video processing time is a separate cost and should be estimated from the pilot.

The user authorized all four phases, and the scoped execution is complete. The retained roadmap records the design and its original effort estimates. Delivered code, measured results, limitations and reproducibility hashes are linked from the README and final report.
