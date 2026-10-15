# Player Fatigue Detection — 12-Slide Presentation Content

Each slide includes text to paste into the presentation, a suggested visual and evidence links for presenter notes. Replace the name and roll-number placeholders on Slide 1. Suggested visuals refer to existing project outputs; evaluation tables can also be converted into charts in PowerPoint.

---

## Slide 1 — Player Fatigue Detection Using Computer Vision

**Subtitle:** Movement and gait analysis from soccer match footage

- BCSE407L Computer Vision — RIL Mini-Project
- Presented by: [Your Name]
- Roll number: [Your Roll Number]
- Python • YOLOv8 • ByteTrack • Pose Estimation

**Suggested visual:** A still from the annotated demonstration showing the player box, skeleton and pitch minimap.

**Visual file:** `results/phase4/annotated_demo_preview.png` — available in the local project.

---

## Slide 2 — Project Objective and Motivation

- Analyze changes in a soccer player's movement using match video.
- Follow selected players and estimate their positions on the field.
- Measure running speed, high-intensity movement and projected gait mechanics.
- Compare early and late match segments and examine sustained movement changes.
- Present measurements through graphs, an annotated video and an interactive dashboard.

**Project contribution:** Combining player tracking, field geometry, feature engineering and time-series analysis into one reusable Python pipeline.

**Suggested visual:** A simple sequence: **Match Video → Player Tracking → Movement Measurements → Trend Analysis → Dashboard**.

**Source for presenter notes:** [Project methodology](IMPLEMENTATION_PLAN.md), [final report](FINAL_REPORT.md).

---

## Slide 3 — Dataset and Study Design

**Primary dataset:** SoccerTrack v2 — panoramic soccer videos with player annotations and camera reference information.

| Study component | Scope |
|---|---|
| Matches | 2: 117093 and 128057 |
| Selected players | Blue #36, White #9 and White #17 |
| Analyzed segments | 4 × 25 minutes |
| Distinct footage analyzed | 100 minutes |
| Source resolution and frame rate | 4096 × 1080 pixels, 25 fps |
| Supplementary broadcast audit | 5 SoccerNet-GSR clips, 300 sampled images |

- Main source files: approximately **25.29 GB**, including about **184 minutes of video** and supporting annotations.
- Supplementary broadcast samples: approximately **175 MB** of extracted files.
- Players were selected before inspecting their longitudinal movement results.

**Suggested visual:** A table showing Match A → Blue #36 and Match B → White #9 / White #17, each with an early and late segment.

**Sources for presenter notes:** [SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2), [SoccerNet-GSR](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024), [locked study manifest](data/manifests/study.json), [download manifests](data/manifests/downloads/).

---

## Slide 4 — End-to-End Processing Pipeline

1. **Read video:** Extract frames with their original timestamps.
2. **Detect people:** YOLOv8n identifies candidate player boxes.
3. **Track the target:** ByteTrack links detections; study annotations assist player identity.
4. **Map field positions:** Convert detected ground anchors into coordinates in meters.
5. **Estimate pose:** Extract body joints during suitable running passages.
6. **Calculate features:** Summarize movement in two-minute windows and gait in accepted cycles.
7. **Analyze and visualize:** Produce comparisons, trends, quality tables and dashboard views.

**Processing scale:** 225,000 target-player frame records across the study. These include position and observation-status records; pose inference is applied to selected running frames.

**Suggested visual:** A horizontal pipeline with the seven stages above.

**Source for presenter notes:** [Final workflow](FINAL_REPORT.md), [tracking implementation](src/detect_track.py), [pose extraction](src/pose_pass.py).

---

## Slide 5 — Models, Tracking and Training Experiment

| Component | Role |
|---|---|
| YOLOv8n | Pretrained person detection |
| YOLOv8m-Pose | Pretrained body-joint estimation |
| ByteTrack | Association of detections into track IDs |
| Geometric field mapping | Image-to-field coordinate conversion |
| Regression and PELT | Movement-trend and change-point analysis |

- One YOLOv8n adaptation experiment: **387 training, 100 validation and 100 held-out test images**.
- Training settings: **15 epochs**, batch size **4**, mixed precision and frozen early layers.
- Experiment and evaluations completed in approximately **258 seconds** on the **RTX 4060 Laptop GPU with 8 GB VRAM**.
- Validation selected the **original pretrained YOLOv8n** for the final pipeline; YOLOv8m-Pose uses pretrained weights.

**Suggested visual:** Model-role table with a small hardware caption: RTX 4060 Laptop / Ryzen 7 7840HS / 16 GB RAM.

**Source for presenter notes:** [Detector experiment](results/phase2/training/bounded_run.json), [training comparison](results/phase2/training/comparison.csv), [hardware and pose selection](PHASE_1_3_REPORT.md).

---

## Slide 6 — Field Calibration and Movement Metrics

- Known pitch reference points establish the relationship between image pixels and field meters.
- The panoramic study cameras use a **quadratic rational ground mapping** suited to their image geometry.
- A detected box-bottom anchor represents the player's ground position.
- Smoothed ground displacement and elapsed time provide movement measurements.

**Movement features:**

- **Average speed:** Observed distance divided by valid observed time, in m/s.
- **Sustained peak speed:** Highest supported approximately one-second average speed.
- **Running-speed p90:** The 90th percentile of observed speeds during running at ≥2 m/s.
- **High-intensity distance rate:** Distance at ≥5.5 m/s per observed minute.

**Calibration evaluation:** Held-out field-point median error was **0.654 m for Match A** and **0.615 m for Match B**, with 13 tested reference points per camera.

**Suggested visual:** Observed pitch trajectory from `results/plots/117093_first_A_blue_36_trajectory_coverage.png`.

**Source for presenter notes:** [Match A calibration](results/phase2/calibration/117093.json), [Match B calibration](results/phase2/calibration/128057.json), [movement calculations](src/metrics.py). Calibration errors concern reference points on the field.

---

## Slide 7 — Pose Estimation and Gait Features

- YOLOv8m-Pose estimates shoulders, hips, knees and ankles from padded player crops.
- Joint coordinates are restored to the original video frame.
- Gait calculations use native **25-fps observations** from suitable straight-running passages.

| Gait feature | Measurement |
|---|---|
| Cadence | Same-leg running cycles per second |
| Stride-length proxy | Ground distance traveled per same-leg cycle |
| Knee excursion | Robust range of projected knee flexion, in degrees |
| Vertical-motion proxy | Projected hip oscillation normalized by body height |

**Demonstration output:** **63 accepted same-leg cycles**, with cycle-level plots and comparisons against running speed.

**Suggested visual:** `results/plots/cycle_stride_frequency.png` alongside `results/plots/speed_vs_knee_angle_range.png`.

**Source for presenter notes:** [Accepted gait-cycle table](results/metrics_csv/all_gait_cycles.csv), [pose extraction](src/pose_pass.py), [metric definitions](results/metrics_csv/units.json).

---

## Slide 8 — Time-Series Analysis and Graphs

- Primary movement summaries use **120-second nonoverlapping windows**; display summaries also use a **60-second step**.
- Own-player percentage changes use an eligible early ten-minute baseline.
- Regression describes the direction and rate of change over time.
- PELT examines sustained changes within eligible continuous sequences.
- Sensitivity checks vary smoothing, window length and change-point penalty.

**Graph views:** Movement over match time, early/late comparison, pitch trajectories, individual gait cycles and speed-versus-gait relationships.

**Suggested visual:** `results/phase4/plots/avg_speed_decline.png` and `results/phase4/plots/high_intensity_distance_per_minute_decline.png`.

**Presenter note:** Explain the physical-value panel first. The percentage-change panel applies where the early baseline qualifies. These graphs display measurements; the numerical model evaluation is on Slides 9–10.

**Source for presenter notes:** [Analysis implementation](src/fatigue_analysis.py), [analysis report](results/phase4/analysis_report.json), [sensitivity results](results/phase4/sensitivity.csv).

---

## Slide 9 — Detector Evaluation Results

**Selected model:** Pretrained YOLOv8n, evaluated on **100 held-out images from a separate match**.

| Evaluation metric | Result |
|---|---:|
| Precision | **94.68%** |
| Recall | **89.98%** |
| mAP@0.50 | **93.10%** |
| mAP@0.50:0.95 | **72.32%** |

- **Precision:** How often reported detections correspond to labeled people.
- **Recall:** How many labeled people are detected at the evaluation operating point.
- **mAP:** Detection performance across confidence thresholds, with stricter localization assessed by mAP@0.50:0.95.

**Suggested visual:** A labeled four-bar chart using the values above. Keep each bar's metric name visible.

**Source for presenter notes:** `chosen_test` in [bounded_run.json](results/phase2/training/bounded_run.json). These numbers evaluate person detection on the held-out image subset.

---

## Slide 10 — Broadcast and Pose Evaluation

**Separate broadcast audit:** Five SoccerNet-GSR test clips, **60 sampled frames per clip**.

| Clip | Labeled-player detection recall at IoU ≥0.50 |
|---|---:|
| SNGS-116 | **85.26%** |
| SNGS-117 | **91.75%** |
| SNGS-118 | **87.90%** |
| SNGS-119 | **94.15%** |
| SNGS-120 | **92.70%** |

- Across all sampled frames: **3,178 / 3,530 labeled player boxes matched — 90.03% pooled recall**.
- Pose availability: **98 / 100 selected player crops** passed the six-joint confidence gate.
- Independent 60-second tracking audit: **79.39% known-player box recall** at IoU ≥0.50, without identity hints.

**Suggested visual:** A five-bar recall chart from the table, with the pose result shown as “98/100 crops passed the confidence gate.”

**Source for presenter notes:** [Broadcast clip results](results/phase4/broadcast_audit/clips.csv), [frame counts](results/phase4/broadcast_audit/frames.csv), [pose audit](results/phase4/broadcast_audit/poses.csv), tracking result documented in [PHASE_1_3_REPORT.md](PHASE_1_3_REPORT.md). Pose crops were the two tallest matched players every sixth sampled frame; this measures confidence availability. Broadcast recall and the separate tracking audit use different evaluation data.

---

## Slide 11 — Movement Results and Interactive Demo

**Case study:** A_blue_36, comparing qualified observations from the selected early and late 25-minute segments.

| Feature | Early | Late | Relative change |
|---|---:|---:|---:|
| Average speed | 2.207 m/s | 1.930 m/s | **−12.5%** |
| Sustained peak speed | 6.538 m/s | 5.873 m/s | **−10.2%** |
| Running-speed p90 | 5.609 m/s | 5.287 m/s | **−5.8%** |
| High-intensity distance rate | 17.698 m/min | 11.142 m/min | **−37.0%** |

**Observed result:** Lower recorded movement intensity in the late segment.

**Demo features:** Player selection, metric plots, measured pitch trajectories, comparison tables and a continuous video showing the player box, skeleton, valid speed and minimap.

**Suggested visual:** `results/phase4/plots/early_late_comparison.png`. For the live demonstration, open `results/phase4/dashboard.html` and play the local `results/phase4/annotated_demo.mp4`.

**Source for presenter notes:** [Qualified segment comparison](results/phase4/early_late_comparison.csv), [dashboard](results/phase4/dashboard.html). Means are weighted by valid observed seconds in eligible windows; changes compare selected segments rather than complete halves.

---

## Slide 12 — Limitations and Future Work

- Movement changes are fatigue proxies; physiological fatigue requires independent validation. No confirmed fatigue-onset time is claimed.
- The reported early/late movement comparison is a descriptive single-player case study; a larger cohort would strengthen evaluation.
- Gait demonstrations use accepted individual cycles; longer continuous running observations are needed for longitudinal gait conclusions.
- Pose measurements depend on player visibility and viewpoint; moving cameras require dynamic field calibration.

**Future work:** Broader match evaluation, automatic calibration updates, stronger player re-identification and validation against wearable or physiological measurements.

**Repository:** https://github.com/srutan2004/Player-Fatigue-Detection

**Source for presenter notes:** [LIMITATIONS.md](LIMITATIONS.md), [FINAL_REPORT.md](FINAL_REPORT.md).
