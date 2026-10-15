# Player Fatigue Detection: Complete Working Explanation

This document explains the implemented project from first principles through its models, datasets, measurements, statistical analysis and React dashboard. Numerical results describe the saved study in this repository. Use [README.md](README.md) for setup and commands, [FINAL_REPORT.md](FINAL_REPORT.md) for the findings, and [PRESENTATION_CONTENT.md](PRESENTATION_CONTENT.md) for slide content.

**Reading guide:** Start with [the simple explanation](#1-the-idea-in-simple-words) and [model roles](#3-which-models-and-algorithms-are-used), then review [datasets](#4-dataset-downloaded-data-versus-used-data), [test splits](#5-what-exactly-is-the-training-validation-and-test-data), [feature equations](#7-feature-engineering-the-measurements-and-equations), [statistics](#9-statistics-what-is-computed-and-how-to-present-it), [results](#11-actual-findings-in-this-repository) and [dashboard operation](#13-the-react-liquid-glass-dashboard).

## 1. The idea in simple words

Imagine watching one football player during an early part and a late part of a match. You want to measure whether the player moves more slowly, performs less high-intensity running, or changes their running mechanics. Watching alone is subjective. This project turns selected observations into numbers and graphs.

The program finds people in the video, follows selected players, estimates their positions on the pitch, and measures how those positions change over time. During suitable running passages, it also locates body joints and measures repeated leg movements. It then summarizes observations in time windows, compares early and late periods, and looks for persistent changes.

The result is a **video-based movement and gait analysis pipeline for investigating fatigue-related decline**. A lower movement workload can be consistent with fatigue, but tactics, match events and running opportunities also affect it. The implementation therefore reports the measured change and the evidence available for interpreting it.

### What happens in the current demo?

The demo contains **three selected players**, not just one:

| Display identity | Source match | Reference identity | Selected periods |
|---|---|---|---|
| `A_blue_36` | 117093, Match A | 473125 | Early and late, 25 minutes each |
| `B_white_9` | 128057, Match B | 506463 | Early and late, 25 minutes each |
| `B_white_17` | 128057, Match B | 506460 | Early and late, 25 minutes each |

All three have tracking and movement records. Their usable coverage differs. Only `A_blue_36` qualifies for the final paired early/late movement comparison. Accepted individual gait cycles are also available for inspection, although their exposure is insufficient for longitudinal gait-window inference.

The dashboard opens saved results. Changing a player or graph filters those results; it does not retrain a model or process a new video in the browser.

### A short vocabulary

| Term | Meaning here |
|---|---|
| Frame | One image in the video; the study video supplies 25 each second |
| Detection | A rectangle and confidence score around a person in one image |
| Tracking | Connecting detections over successive images into a temporary track ID |
| Player identity | The actual selected person, maintained across reviewed tracklets |
| Pose/keypoints | Estimated image locations of joints such as hips, knees and ankles |
| Calibration | Converting an image location on the ground into pitch coordinates in meters |
| Feature/metric | A calculated measurement, such as speed or stride frequency |
| Window | A fixed interval used to summarize observations, normally 120 seconds |
| Baseline | The player's own qualified early observations used as a reference |
| Change point | A possible boundary between different levels of a time series |
| Inference | Running an existing model to obtain predictions |
| Fine-tuning | Updating pretrained neural-network weights using additional labeled examples |

## 2. The complete pipeline

```text
Permitted match video + timestamps + camera reference points
                         |
                  Decode source frames
                         |
                 YOLOv8n person boxes
                         |
                 ByteTrack tracklets
                         |
       Selected-player association + identity/gap checks
                         |
            Detected ground anchor -> field mapping
                         |
          Smoothed ground positions -> movement metrics
                         |
           Suitable native-rate straight-running bouts
                         |
           YOLOv8m-Pose -> confidence-qualified joints
                         |
          Repeated leg cycles -> projected gait metrics
                         |
        Quality-qualified windows + own-player baselines
                         |
     Trends + change-point search + sensitivity + comparisons
                         |
      CSV/JSON evidence + PNG plots + offline React dashboard
```

CSV/Parquet files, flags and reports are saved between stages. This makes it possible to inspect an intermediate mistake or rebuild the dashboard without repeating expensive inference.

## 3. Which models and algorithms are used?

| Component | Selected implementation | What it produces | Training in this project |
|---|---|---|---|
| Person detector | Pretrained **YOLOv8n**, COCO person class 0 | Boxes and detection scores | One adaptation experiment; original pretrained weights retained |
| Pose estimator | Pretrained **YOLOv8m-Pose** | 17 joint locations and confidences | None |
| Tracker | Supervision **ByteTrack** | Temporary IDs associated across frames | None; association/motion algorithm |
| Study identity association | Reference-assisted global assignment with appearance/geometric checks | Selected-player links and ambiguity flags | None |
| Ground calibration | Validated quadratic rational warp for these panoramas; homography available for appropriate views | Pitch X/Y in meters | Geometric parameter fitting, not neural training |
| Temporal analysis | OLS regression, HAC uncertainty, PELT and sensitivity checks | Slopes, uncertainty, candidate changes and statuses | Statistical fitting, no fatigue classifier |
| Presentation | React, Plotly and CSS | Interactive local HTML report | None |

### 3.1 YOLOv8n: finding people

YOLO is an object detector: it learns visual patterns and predicts object locations in an image. The selected `n` variant is the small nano model. Its pretrained COCO weights already recognize people; only the person class is used here. Detecting a person does not identify their jersey number or establish which player they are.

The study uses a 1920-pixel inference setting and a 0.15 detection-confidence threshold to retain small/difficult player candidates for subsequent checks. These are the actual study settings, rather than the approximate 0.5 threshold in the original proposal. See the [official YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/) and [study configuration](configs/study_117093_first.yaml).

### 3.2 ByteTrack: linking boxes through time

ByteTrack predicts track motion and associates new detections with existing tracks. Its handling of lower-scoring detections can help retain an existing track when a person becomes less clear. A track ID is temporary: occlusion or crowding can cause it to disappear or switch. The original method is described in the [ByteTrack paper](https://arxiv.org/abs/2110.06864).

The study detects/tracks at **12.5 fps**, while maintaining records on the native **25-fps** time grid. Intermediate short box propagation is explicitly tagged. Fresh extraction uses a ByteTrack loss buffer corresponding to approximately one second. Historical study caches retain their original shorter buffer; the record is preserved rather than silently changed.

The long study uses dataset references to assist identity association. It matches independently detected ground anchors to plausible reference identities, with global assignment, kit-color checks and rejection of ambiguous links. Reference coordinates are **not substituted for the measured trajectory**. A bounded 0.8-second continuity rule applies only to a still-observed, recently verified tracklet; it does not fill missing detections or approve a new ID automatically.

For a new video without reference identities, the user selects the initial tracklet and reviews later reappearances. A new match therefore needs configuration and identity/calibration setup before analysis.

### 3.3 YOLOv8m-Pose: locating joints

The pose model predicts 17 body keypoints from padded player crops. Hips, knees and ankles support the gait calculations; other joints help inspection. A joint-confidence threshold of **0.5** masks unreliable coordinates.

The model runs on selected suitable running passages at the native 25-fps rate. It is not applied densely to every one of the 225,000 target-frame records. Padded crop origins are added back to keypoint coordinates so joints are stored in the original full-frame image space.

A fixed ten-minute model-choice audit compared smaller and medium pose models: the six-joint confidence criterion passed on **173/255 crops for YOLOv8s-Pose** and **249/255 for YOLOv8m-Pose**. Overlays were inspected, and the medium model was selected. These fractions measure confident-output availability, not anatomical keypoint accuracy. No pose model was trained. MediaPipe is not used in the final implementation.

### 3.4 What was actually fine-tuned?

One bounded YOLOv8n detector adaptation was run on the laptop:

| Setting | Recorded value |
|---|---|
| Training images | 387 |
| Validation images | 100 |
| Held-out test images | 100 |
| Epochs | 15 |
| Batch size | 4 |
| Frozen early layers | 10 |
| Optimizer / mixed precision / seed | AdamW / enabled / 42 |
| Hardware | RTX 4060 Laptop GPU, 8 GB VRAM |
| Experiment time including evaluations | Approximately 258 seconds |

The model-selection criterion was validation mAP@0.50:0.95:

| Weights | Validation result |
|---|---:|
| Original pretrained YOLOv8n | **0.7105** |
| Adapted YOLOv8n | 0.6803 |

The original pretrained model won and is used by the final pipeline. Thus, the accurate description is: **we evaluated a detector fine-tuning experiment, retained the better pretrained detector, and built the player-level measurement and statistical pipeline around pretrained models**. There is no separately trained neural fatigue model. Evidence: [experiment record](results/phase2/training/bounded_run.json) and [comparison table](results/phase2/training/comparison.csv).

## 4. Dataset: downloaded data versus used data

### 4.1 Main source: SoccerTrack v2

The main dataset is [SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2), release v1.2, pinned to revision `325bd4c57ee8e135ab652f3921d274526ada496b`. It supplies panoramic match videos, player annotations, curated tracking clips and field/camera assets.

Both halves of matches **117093** and **128057** were downloaded. Their panoramic videos are **4096 x 1080 pixels at 25 fps** and total approximately **184.05 minutes**. The project did not download all of SoccerTrack v2.

| Downloaded main-source category | Approximate size, decimal GB | Purpose |
|---|---:|---|
| Four full-half panoramic videos | 13.59 | Source images and time |
| Four GSR annotation JSON files | 11.06 | Reference player identity and observation context |
| Curated MOT clips and labels | 0.61 | Detector subset and independent tracking audit |
| Field/camera reference assets | 0.03 | Ground calibration |
| **Main downloaded source total** | **25.29** | Excludes environments, weights and derived caches |

GB here means one billion bytes. SQLite indexes, processed measurements and training tiles are derived local files, not additional independent footage. Exact source-file hashes and sizes are retained in [download manifests](data/manifests/downloads/).

### 4.2 Longitudinal study selection

| Segment | Source-video interval | Configured match interval | Players |
|---|---|---|---|
| Match A, first half | 05:00-30:00 | 05:00-30:00 | Blue #36 |
| Match A, second half | 15:00-40:00 of half video | 60:00-85:00 | Blue #36 |
| Match B, first half | 05:00-30:00 | 05:00-30:00 | White #9, White #17 |
| Match B, second half | 15:00-40:00 of half video | 60:00-85:00 | White #9, White #17 |

This is **four distinct 25-minute segments = 100 footage minutes**. Each player has 50 selected minutes, making **150 player-minutes** when player observations are counted separately. Two players from the same Match B frames do not create extra independent video minutes.

At 25 fps, those player streams yield **225,000 native target-frame rows**, including missing/invalid observations. They produce **72 nonoverlapping two-minute analysis windows** and **144 overlapping display windows**. The final one-minute remainder of each 25-minute player segment is not treated as a complete two-minute window. Players were locked before inspecting longitudinal decline; see [study.json](data/manifests/study.json).

Match time uses a configured 2700-second second-half offset. It does not reconstruct first-half stoppage time. The unprocessed interval between the selected early and late segments contains no evidence about when a change occurred.

### 4.3 Separate broadcast audit: SoccerNet-GSR

A supplementary source is [SoccerNet/SN-GSR-2024](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024), pinned to `3cc710eb6d53a23350a3d2863311c1c0a2e645d7`.

Five public test clips, **SNGS-116 through SNGS-120**, supply **300 sampled images: 60 per clip, approximately 2 fps**. Each original clip is about 30 seconds. Selected JPEGs and five annotation files were fetched by checked archive byte ranges, rather than downloading the entire approximately 8.85-GB test ZIP. Extracted files occupy approximately **175.25 MB**. Main source plus these extracted samples totals approximately **25.46 GB**, before derived data and software.

This audit asks whether person detection, selected-crop pose availability and fixed-camera assumptions hold on broadcast examples. It is not training data, a continuous fatigue sequence, or a cadence dataset. The [official SoccerNet game-state project](https://github.com/soccernet/sn-gamestate) describes the source task.

## 5. What exactly is the training, validation and test data?

Different components require different checks. The project does not have one shared "accuracy" for everything.

### 5.1 Detector split

| Split | Source | Purpose | Final tile count |
|---|---|---|---:|
| Training | Match 117093 curated MOT clip, source seconds 0-150 | Update detector weights in the single experiment | 387 |
| Validation | Same curated clip, seconds 180-240 | Choose pretrained versus adapted weights | 100 |
| Test | Match 128057 curated MOT clip, seconds 0-240 | Evaluate the selected detector on a held-out match | 100 |

Source frames were assigned to splits before tiling, preventing crops from the same frame from entering different splits. Training and validation have a 30-second source gap. The initial 600-tile preparation lost 13 questionable training images to quarantine, leaving **587 final images**. Extra staff/referee proposals were visually reviewed; some labels are model-assisted, so the evaluation remains an internal study rather than an independently curated benchmark.

Match B is held out **from detector development**. It is also part of the later movement study; it should not be described as a third unseen match for every component. See [source split specification](configs/detector_sources.json); completed counts and selected weights are established by the experiment report.

### 5.2 Other evaluation data

| Component | Evaluation observations | What the result can establish |
|---|---|---|
| Raw tracking | First 60 seconds of Match B curated MOT, 750 updates at 12.5 fps, no identity hints | Known-player box recall and consecutive observed ID switches |
| Field mapping | 13 held-out field reference points per camera | Ground-reference projection error |
| Broadcast detector | 300 SoccerNet-GSR sampled frames | Recall against labeled players at IoU 0.5 |
| Broadcast pose | 100 deliberately selected large matched crops | Six-joint confidence availability |
| Gait algorithms | Known-signal automated tests and inspected study cycles | Software behavior and plausible descriptive measurements |
| Change-point code | Known downward step, stationary noise, constant synthetic signals | Expected algorithm behavior on controls |
| Fatigue interpretation | Qualified early/late periods and optional verified substitutions | Proxy comparison; no sensor-labeled fatigue test set exists here |

## 6. Technical working: video, identity, geometry and pose

### 6.1 Decode and align time correctly

`src/video_io.py` uses PyAV to decode sequentially and preserve native presentation timestamps. Time in the file is separated from configured match time. Some source streams begin at a presentation timestamp of one second; subtracting that origin prevents a systematic annotation shift.

The annotation conventions were checked against overlays: the GSR image-name suffix is one-based, whereas the curated MOT frame numbering is zero-based. Multi-GB GSR JSON is streamed into SQLite indexes for bounded-memory lookups. Source annotations assist association; their quantized field positions are not used as the measured movement trace.

Sparse 2-fps pilots are suitable for image-quality and runtime checks. They cannot reliably resolve alternating leg motion. This is why the implemented gait stage uses native 25-fps passages rather than the proposal's approximate 5-fps setting.

### 6.2 Measure a ground position

For each accepted player box, a consistent box-bottom ground anchor represents its contact with the field. The calibrated mapping converts this image point into `(world_x, world_y)` meters.

A homography is a 3 x 3 projective matrix `H` for a planar scene:

```text
[a, b, c]^T = H [u, v, 1]^T
X = a/c; Y = b/c
```

The field is approximately planar, but the study images are curved panoramas. A single raw-pixel homography failed the checks. The implementation compares candidate mappings and selects the simplest passing method; both study cameras select a **quadratic rational warp**.

Conceptually, this uses normalized image coordinates and quadratic terms such as `1, u, v, u^2, uv, v^2` in the coordinate numerators, divided by a shared fitted denominator. Held-out points evaluate generalization, while convex-hull and denominator checks reject unreliable extrapolation. Ordinary suitable camera views can still use the provided homography functions.

The hip is above the ground plane. Projecting a hip through a ground map does not produce its true physical ground position or height. Consequently ground speed uses the detected ground anchor, while vertical body motion remains a normalized image proxy.

Camera registration checks flag movement. They do not automatically solve a new per-frame calibration for arbitrary camera pans and zooms.

### 6.3 Keep usable running observations

Gait analysis uses approximately straight running at **2-9 m/s**, at least **four seconds** continuously, and source body-box height of at least **40 pixels**. Heading spread is limited to **30 degrees over two seconds**. Severe overlaps and ambiguous identity/calibration observations are excluded upstream.

For pose, a crop receives about 10% padding, clipped to the image boundary. YOLO's restored crop-space joint coordinates are offset by the actual clipped crop origin. Invalid or low-confidence joints remain missing. Crop origin handling matters: otherwise apparently sensible joints would be attached to the wrong full-frame location.

## 7. Feature engineering: the measurements and equations

### 7.1 Ground motion and smoothing

Raw detector boxes jitter. Directly differentiating them can create large false speeds. Valid positions are therefore processed inside uninterrupted clip/player/identity segments using a three-point median filter and an approximately **0.8-second Savitzky-Golay smoother**. Position gaps longer than 0.15 seconds and identity changes break the smoothing segment.

```text
speed(t) = sqrt((dX/dt)^2 + (dY/dt)^2)
distance_increment = distance between consecutive smoothed ground positions
```

Speeds above **12 m/s** are marked invalid, rather than clipped down to the threshold. A distance/time interval is included only when both ends have plausible measurements. Neither smoothing nor distance calculation bridges an identity break or missing segment.

### 7.2 The eight metrics

| Metric | Definition and aggregation | Unit / interpretation |
|---|---|---|
| Average speed | Sum of valid distance increments / valid observed seconds | m/s; observed work rate |
| Sustained peak speed | Highest approximately one-second rolling mean, with about 0.9 seconds of observations required | m/s; reduces isolated jitter spikes |
| Running-speed p90 | 90th percentile of finite speed samples at or above 2 m/s | m/s; upper running-speed distribution |
| High-intensity distance rate | `60 * distance_at_speed>=5.5 / valid_seconds` | Meters per observed minute; work intensity |
| Same-leg cadence | Inverse time between accepted repeated same-leg landmarks | Cycles/s; different from alternating steps/s |
| Stride-length proxy | Ground-path distance between successive accepted same-leg landmarks | Meters/cycle; coupled to running speed and cadence |
| Vertical-motion proxy | Standard deviation of detrended relative hip image height within a cycle | Fraction of source body-box height; not meters |
| Knee excursion | 95th minus 5th percentile of projected knee flexion within a cycle | Degrees; robust 2D range of motion |

Windowed gait features are **medians of qualified cycle values**. Gait-cycle scatter plots retain individual accepted cycles rather than pretending every cycle forms a qualified time window.

### 7.3 How a leg cycle is detected

For each leg, the ankle's horizontal position is expressed relative to that leg's hip and divided by source body-box height:

```text
leg_signal(t) = (ankle_x(t) - same_side_hip_x(t)) / body_box_height(t)
```

A linear trend is removed. Peak detection finds repeated landmarks in the oscillation. A complete signal uses a five-point smoother and autocorrelation to check repetition. A signal with small confidence holes uses Lomb-Scargle periodicity on the observed samples; joint coordinates are not interpolated.

Acceptance requires native sampling of at least 20 fps, at least 80% observed signal, no observation hole over 0.16 seconds, at least four credible peaks, median period between 0.28 and 2 seconds, period coefficient of variation at most 0.25, and periodicity support of at least 0.3. If both legs qualify, their frequency estimates must agree within 25%.

For successive same-leg landmarks `t1, t2`:

```text
cycle_frequency = 1 / (t2 - t1)
cycle_stride_length_proxy = sum(valid ground distance within that cycle)
```

These landmarks are not independently validated foot strikes. Their purpose is a reproducible image-based cycle proxy.

### 7.4 Knee angle and vertical motion

For joint vectors `a = hip - knee` and `b = ankle - knee`:

```text
internal_angle = acos(dot(a,b) / (norm(a)*norm(b)))
knee_flexion = 180 degrees - internal_angle
knee_excursion = percentile95(flexion) - percentile5(flexion)
```

Degenerate vectors and unconfident joints have no angle. Percentile range reduces sensitivity to a single extreme prediction. The result is a camera-projected angle, not a reconstructed 3D joint angle.

For the vertical proxy, the midpoint of the two hips is compared with the image ground-anchor Y, divided by body-box height, and detrended. The within-cycle standard deviation is retained when at least 80% of cycle values are finite. A change can reflect mechanics, viewpoint or body-scale effects, so its direction is descriptive.

## 8. Window quality, missing data and baselines

### 8.1 Why there are two window tables

Both use **120-second windows**. Inference windows advance by **120 seconds**, so adjacent observations do not share footage. Display windows advance by **60 seconds** for a denser visual view. Display windows overlap and are not treated as additional independent evidence in hypothesis tests.

A movement window needs at least **70% valid observed time**, i.e. 84 seconds out of 120. A gait window needs at least **20 accepted running seconds**, at least **10 cycles from one leg**, and at least **10 finite cycle values for the specific feature**. Left/right cycle intervals are unioned when calculating exposure, preventing two legs from doubling observed running time.

Unusable measurements are represented as missing values with status/rejection information. Missing speed is not zero speed, and a missing gait window is not evidence of no leg movement.

### 8.2 Own-player baseline normalization

Each metric uses the first **600 elapsed seconds of the selected early clip** as its candidate reference interval. Here that corresponds to configured match minutes 5-15, not minutes 0-10 from kickoff.

For qualified baseline windows, calculate an exposure-weighted mean:

```text
baseline = sum(window_metric * exposure_seconds) / sum(exposure_seconds)
percentage_change = 100 * (window_metric / baseline - 1)
```

At least three eligible windows and **420 movement seconds** are required. Gait baselines instead require at least 60 accepted running seconds. A zero/near-zero reference cannot support a percentage change.

| Player | Baseline evidence | Movement normalization |
|---|---|---|
| A_blue_36 | 4 eligible windows; 389.88 seconds | Unavailable: exposure below 420 seconds |
| B_white_17 | 2 eligible windows; 203.00 seconds | Unavailable: too few windows and seconds |
| B_white_9 | 5 eligible windows; 465.96 seconds | Available |

Baseline eligibility and early/late comparison eligibility are different questions. A_blue_36 can have adequate full-period coverage for a physical-unit comparison while lacking sufficient coverage in the particular first-ten-minute baseline interval.

## 9. Statistics: what is computed and how to present it

### 9.1 Descriptive statistics

The project uses means, exposure-weighted means, medians, standard deviations, percentiles, counts, coverage proportions and relative changes. These answer questions such as how much qualified time was observed, how fast the player moved, and how much the late segment differs from the early segment.

An early/late change is `100 * (late_mean / early_mean - 1)`. A period's reported mean weights qualified window values by their valid exposure. Consequently an early/late summary of window peak speeds is a weighted mean of **window peaks**, not the single fastest instant in the whole period. The same distinction applies to window p90 summaries.

### 9.2 Linear regression: gradual change

For each available normalized metric, the implementation fits a line:

```text
metric_change_percentage_points = intercept + slope * elapsed_minutes + residual
```

At least **six contiguous eligible nonoverlapping windows** are needed. Missing windows, clip boundaries and the early/late gap split the series. A slope of -0.5 means an estimated fall of 0.5 baseline percentage points per minute within that fitted sequence.

Fitting uses statsmodels OLS with **lag-1 HAC covariance**. HAC adjusts uncertainty for heterogeneous residual variance and short-range temporal dependence; it does not change the fitted OLS slope or turn a small sequence into a large experiment. The output includes slope, **95% confidence interval** and a slope p-value. Available trend p-values receive **Holm correction per player**, reducing the multiple-comparison problem when several trends are examined. See the [OLS documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html).

A p-value concerns compatibility with a zero-slope model under its assumptions. It is not the probability the player is fatigued. An interval crossing zero indicates that a clear nonzero slope is not established by that fit.

### 9.3 PELT: sustained level changes

PELT searches for boundaries that improve a piecewise-constant description of the series while penalizing extra boundaries. The project uses an L2 cost and at least three observations per segment. A search requires at least **eight contiguous eligible windows** and a metric with a pre-specified downward interpretation. See the [official ruptures PELT guide](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/).

Before searching, local noise is estimated robustly from first differences using median absolute deviation. The normalization scale is at least two baseline percentage points. Three penalties, **2, 3 and 4 times log(n)**, are checked. A candidate needs agreement from at least two settings within one window.

The medians of three windows immediately before and after the boundary must show a downward shift of at least **10 baseline percentage points** and at least **twice the local noise**. The recorded time resolution/uncertainty is one analysis window, normally 120 seconds. This is a design resolution, not a clinically calibrated confidence interval.

Average speed, sustained peak, running p90, high-intensity rate, stride length and knee excursion have a specified downward interpretation. Cadence and vertical motion can change in different directions; they are not forced into the same decline rule.

### 9.4 Sensitivity: does a candidate survive reasonable settings?

Primary candidates are checked using movement smoothing lengths **0.6/0.8/1.2 seconds** and nonoverlapping window lengths **120/180/240 seconds**. This yields nine setting combinations; each change-point search also checks its three penalties. The primary own-player baseline is held fixed for these comparisons.

A candidate must be corroborated across at least two smoothing values, two window widths and at least half the assessable combinations, within the allowed window timing tolerance. Gait poses stay fixed; sensitivity does not train a model or rerun pose inference. Candidates that depend heavily on processing choices remain quality-confounded.

### 9.5 Paired early/late comparison

The comparison unit is a **player**, not an individual frame or window. Each movement period needs at least four eligible windows and at least 70% full-period movement coverage. Gait comparisons need four eligible windows and at least 80 accepted running seconds per period.

With at least three qualified player pairs for a metric, the implementation attempts an exploratory **Wilcoxon signed-rank test**, using exact mode, then Holm correction across available metric tests. Very small samples, ties and zeros still constrain interpretation. All-zero differences return p = 1. The current study has only **one qualified movement pair**, so no population paired-test p-value is reported.

This avoids counting many correlated windows from one person as many independent people. A paired t-test was an option in the original proposal; **Wilcoxon is the method implemented**, with the stated sample-size gate.

### 9.6 Optional substitution comparison and software controls

A verified substitution time can be supplied with its source in the study manifest. If a supported candidate exists, its timing can be compared with that event. No verified substitutions are supplied for this cohort, and a substitution alone would still be proxy evidence.

Three synthetic controls test the analysis code: a known -25-percentage-point step, stationary noise, and a constant sequence. All three pass their expected behavior. This is software verification, not a fatigue detection accuracy score.

## 10. Evaluation numbers and their correct meaning

### 10.1 Selected detector on 100 held-out Match B tiles

| Metric | Value | Plain-language interpretation |
|---|---:|---|
| Precision | **94.68%** | Fraction of reported detections that match labeled people at the evaluated operating point |
| Recall | **89.98%** | Fraction of labeled people recovered at that operating point |
| mAP@0.50 | **93.10%** | Average precision with box matching at IoU 0.50 |
| mAP@0.50:0.95 | **72.32%** | Average precision averaged over stricter IoU thresholds from 0.50 to 0.95 |

Intersection over union, `IoU = overlap_area / union_area`, measures how well a predicted box overlaps a labeled box. AP summarizes a precision-recall curve as confidence changes; it is not identical to precision at one threshold. These are **person detector metrics**, not fatigue accuracy or pose accuracy.

### 10.2 Independent audits

| Check | Recorded result | Interpretation |
|---|---|---|
| No-hint tracking audit | 79.39% known-player box recall; 21 consecutive observed ID switches | 60 seconds, 750 tracker updates; not a full IDF1/HOTA score |
| Match A field calibration | Median 0.654 m; p95 1.933 m | Error on 13 held-out field references |
| Match B field calibration | Median 0.615 m; p95 2.101 m | Error on 13 held-out field references |
| Broadcast detection pooled | 3178 / 3530 labeled player boxes = **90.03% recall** | Across 300 sampled broadcast images at IoU 0.50 |
| Broadcast selected-crop pose | **98 / 100** pass six-joint confidence gate | Output availability in a large-player sample |

Per-clip broadcast recalls are **85.26%, 91.75%, 87.90%, 94.15% and 92.70%** for SNGS-116 through SNGS-120. Pooled recall weights clips by their number of labeled boxes; it is different from an unweighted mean of the five percentages.

The 100 pose crops are selected as the two tallest matched players every sixth sampled image. This favors larger, visible players. There is no manually labeled anatomical test set establishing a pose-error metric here. Similarly, field-reference error is not a direct ground-truth error measurement for the moving player's ground anchor.

### 10.3 What should be said about "graph accuracy"?

Graphs visualize calculated measurements. They do not have a classification accuracy. Present the detector's precision/recall/AP, calibration errors and observation coverage separately. Label confidence availability as availability. Use the movement graphs to explain observed change, and explain the statistical conditions supporting any trend or candidate boundary.

An appropriate presentation sentence is: "The selected detector achieved 93.10% mAP@0.50 on the held-out image subset, and the qualified Blue #36 case showed a 12.5% lower observed average speed in the selected late segment."

## 11. Actual findings in this repository

For A_blue_36, qualified early and late window summaries are:

| Metric | Early | Late | Relative change |
|---|---:|---:|---:|
| Average speed | 2.207 m/s | 1.930 m/s | **-12.5%** |
| Sustained peak speed | 6.538 m/s | 5.873 m/s | **-10.2%** |
| Running-speed p90 | 5.609 m/s | 5.287 m/s | **-5.8%** |
| High-intensity distance rate | 17.698 m/observed min | 11.142 m/observed min | **-37.0%** |

These compare selected periods, rather than complete first and second halves. The coverage report shows approximately **78.30% early and 82.35% late valid movement time** for this player. The corresponding figures are 74.90%/57.16% for B_white_17 and 84.22%/23.45% for B_white_9; their late periods do not qualify for paired inference.

There are **63 accepted same-leg cycles**, but **zero eligible nonoverlapping gait windows** under the required exposure rules. Cycle-level methodology can be demonstrated; a longitudinal cadence/stride/knee/vertical decline is not established.

Across the 24 player/metric analyses, the recorded statuses are:

| Status | Count |
|---|---:|
| Insufficient gait data | 12 |
| Unavailable baseline | 8 |
| No supported change | 3 |
| Quality-confounded candidate | 1 |
| Supported onset candidate | **0** |

B_white_9's initial average-speed boundary fails the sensitivity requirement and is not accepted as an onset. No onset line is invented for an unavailable or unsupported result. The main observed finding is reduced qualified movement workload for Blue #36, with its source and statistical scope explicitly shown. See [early/late comparison](results/phase4/early_late_comparison.csv) and [analysis report](results/phase4/analysis_report.json).

## 12. Graphs: what each one shows

### 12.1 Final static panels

The canonical final plot directory is [results/phase4/plots/](results/phase4/plots/). It contains eight metric panels and one early/late summary:

| PNG | Read it as |
|---|---|
| `avg_speed_decline.png` | Qualified observed work rate over match time |
| `peak_speed_decline.png` | Sustained window peak speed over time |
| `p90_running_speed_decline.png` | Upper running-speed distribution over time |
| `high_intensity_distance_per_minute_decline.png` | High-intensity running distance per observed minute |
| `stride_frequency_decline.png` | Qualified same-leg cadence windows, or explicit insufficient-data status |
| `stride_length_decline.png` | Qualified stride-length proxy windows/status |
| `vertical_oscillation_decline.png` | Qualified projected vertical-motion windows/status |
| `knee_angle_range_decline.png` | Qualified projected knee-excursion windows/status |
| `early_late_comparison.png` | Player-level qualified early/late summaries |

Time-series panels separate physical-unit observations from available baseline-normalized changes. Breaks represent unavailable evidence. An accepted onset would be marked only if supported; the current study has none.

### 12.2 Additional measurement plots

[results/plots/](results/plots/) retains distinct measurement-stage outputs: combined windowed metrics, pitch trajectory/coverage views for the six player-period streams, four individual-cycle plots, and speed-versus-stride/knee scatter plots.

A trajectory explains where the player was observed. A coverage plot explains where observations are missing. A cycle plot shows descriptive gait samples. A speed-versus-gait scatter helps reveal that changed gait measurements may simply accompany different running speeds. These answer different questions and should be shown with their units and observation counts.

## 13. The React liquid-glass dashboard

### 13.1 How it works

Python assembles saved metrics, reports and display trajectories into an evidence payload. React provides the interface; Plotly draws interactive charts. CSS supplies translucent panels, blur, gradients, reflections, dark/light themes and responsive layouts.

The exporter embeds the **React bundle, Plotly, CSS and JSON in one HTML file**. Opening it requires no server, GPU, Node.js installation or network connection. GitHub displays HTML source; download/open the file locally to interact with it. There is no browser upload/inference service in this implementation.

### 13.2 Six dashboard sections

| Section | Main interactions and evidence |
|---|---|
| Overview | Player/period selection, qualified summaries, timeline, early/late bars and coverage |
| Movement | Eight metric choices, physical/baseline mode, point inspector and analysis status |
| Gait explorer | Individual accepted cycles, leg/speed filters, speed relationships and source body scale |
| Pitch map | Observed trajectory, gaps, time scrub, playback and sampled occupancy |
| Evaluation | Detector results, broadcast recall, pose availability, mapping errors, tracker audit and controls |
| Study & methods | Study scope, models, thresholds, baselines, sensitivity and eligibility tables |

Tables support search and data export; views preserve useful state in their URL. The coverage indicator reports observed movement time, not a fatigue probability. The map uses roughly one-second display samples while preserving intervening invalid intervals and identity breaks. Occupancy is sampled observed occupancy, not a full-match heatmap.

### 13.3 Open or rebuild

From this Windows workspace:

```powershell
Set-Location 'D:\Computer Vision'
Start-Process '.\results\phase4\dashboard.html'
```

To update frontend code and regenerate the HTML from existing results:

```powershell
npm --prefix dashboard ci
npm --prefix dashboard run build
uv run --no-sync python scripts/render_dashboard.py
```

The checked-in production bundle allows Python to export the dashboard without Node.js when JSX/build source has not changed. `render_dashboard.py` changes presentation only; it does not recompute tracking, poses or statistical findings. See [dashboard/README.md](dashboard/README.md).

## 14. Running the pipeline and analyzing another match

### 14.1 Existing project

Use the locked Python setup in the README. To recompute statistical reports from the local final measurement caches:

```powershell
uv run --no-sync python -m src.cli analyze
```

The full sensitivity stage needs frame-level local caches; small published CSVs alone do not recreate every inference stage. To rerun the configured study after acquiring its source files and preparing its indexes:

```powershell
uv run --no-sync python scripts/run_study.py
uv run --no-sync python scripts/finalize_phase3.py
uv run --no-sync python -m src.cli analyze
```

Completed matching caches are reused. A fresh clone must remove the study YAMLs' `pipeline.cached_tracks` entries when those local caches are absent. It must download videos/annotations and create indexes first. The notebook [pipeline_demo.ipynb](notebooks/pipeline_demo.ipynb) walks through the same modules.

### 14.2 A new permitted local video

1. Copy an existing YAML into a new configuration and set the source video, output folders, selected interval, half and match-time offset. Choose continuous footage with adequately visible players.
2. Establish a ground map for that camera using distributed known field points. Reserve points for validation. Use `scripts/calibrate_click.py` on a local desktop or supply a checked mapping. A moving camera requires an appropriate dynamic-calibration solution before trustworthy field speeds can be claimed.
3. For footage without reference annotations, set `reference_assisted_identity: false`, run `select-player`, and set each target's `initial_tracklet_id`. Supply reviewed later tracklet intervals if needed.
4. Run `extract`, then `pose-pass`, then `metrics`, inspecting overlays, gaps and coverage between stages.
5. Register the new clip/player metadata in a study manifest and collect the generated per-clip windows/cycles/coverage into the analysis inputs. Adapt the existing study/finalization drivers to the new cohort rather than reusing their fixed three-player selection unchanged.
6. Run analysis and review the plots/statuses. Early observations are needed for a personal baseline; qualified paired periods are needed for an early/late comparison.

The reusable pretrained models do not require retraining for every match. However, simply replacing the video path is insufficient to guarantee player identity, field scale, usable gait or a fatigue conclusion.

## 15. Code and output map

| Location | Responsibility |
|---|---|
| `src/video_io.py` | Source decoding, frame timing and alignment |
| `src/detect_track.py` | Detection, tracklets, target association and gaps |
| `src/pose_pass.py` | Native-rate selected-bout pose extraction |
| `src/calibrate.py` | Ground mapping and held-out calibration checks |
| `src/metrics.py` | Motion smoothing, running gates, cycles and windows |
| `src/fatigue_analysis.py` | Baselines, regression, candidates and proxy comparisons |
| `src/fatigue_analysis.py` (`run_analysis`) | Movement processing-setting robustness checks |
| `src/decline_visualize.py` | Final static figures and dashboard delivery |
| `src/dashboard.py` | Safe embedded evidence and standalone HTML export |
| `dashboard/` | React source, styles, locked packages and prebuilt client |
| `data/manifests/` | Locked cohort, source provenance and download records |
| `results/metrics_csv/` | Combined windows, individual cycles and coverage |
| `results/phase2/training/` | Completed detector experiment and scores |
| `results/phase2/calibration/` | Selected fitted camera mappings and validation |
| `results/phase4/` | Baselines, normalized data, comparisons, statuses, sensitivity and dashboard |
| `tests/` | Numerical, timing, geometry, gap and output-behavior verification |

Raw videos, credentials, environments, pretrained binaries and large native-frame caches remain local and excluded from Git. Small derived evidence and the prebuilt offline dashboard are published. The local annotated demo uses real footage and is not redistributed in the repository.

The repository cleanup removes exact duplicate final plot copies, an explicitly superseded proposal manifest and obsolete one-off download helpers. Canonical figures, model/source data, useful reports, notebooks and experiment evidence are preserved. Local debug caches remain ignored and are not uploaded.

## 16. Explaining the contribution in a viva

**What did we build?** A reproducible pipeline connecting pretrained person detection and pose estimation to player tracking, validated field geometry, movement/gait features, quality-aware time-series analysis and interactive evidence presentation.

**What did we train?** One small detector adaptation experiment; the original pretrained detector performed better on validation and was retained. The pose model was not trained. There is no supervised fatigue classifier.

**Why is calibration needed?** Pixel distances depend on perspective and position in the image. A checked ground mapping provides metric displacement on the field.

**Why use 25 fps for gait?** Repeated leg motion needs sufficiently dense temporal observations. Sparse audit frames cannot resolve a reliable same-leg cycle.

**Why not output a fatigue score for every player?** A score would require a justified definition and validation target. This implementation reports observed metrics and supported/unsupported evidence states instead of assigning an unvalidated probability.

**What is the main result?** The qualified Blue #36 case has lower measured late-segment movement intensity. The detector and calibration have separately reported evaluation results. No supported fatigue-onset time is claimed by the current cohort.

**What would strengthen the next version?** More continuously visible running, more independently paired players, dynamic calibration for moving cameras, independently labeled pose/identity evaluation, and physiological or trusted workload ground truth. Detailed methodological boundaries are collected in [LIMITATIONS.md](LIMITATIONS.md).
