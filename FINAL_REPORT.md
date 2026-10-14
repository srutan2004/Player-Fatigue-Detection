# Final project report: football movement and fatigue-related decline

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

| Metric | Early | Late | Change |
|---|---:|---:|---:|
| avg_speed | 2.207 | 1.930 | -12.5% |
| peak_speed | 6.538 | 5.873 | -10.2% |
| p90_running_speed | 5.609 | 5.287 | -5.8% |
| high_intensity_distance_per_minute | 17.698 | 11.142 | -37.0% |

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

**Supported onset candidates: 0.** B_white_9's primary average-speed candidate
fails robustness and remains quality-confounded. Unsupported candidates are stored for
inspection, but are not drawn as accepted onset lines. The unprocessed 30–60-minute interval
cannot reveal onset. The configured second-half offset omits first-half stoppage time.

Across 24 player/metric analyses: `{'insufficient_data': 12, 'unavailable_baseline': 8, 'no_supported_change': 3, 'quality_confounded': 1}`.

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

Per-clip known-player recall at IoU 0.5 ranges from 85.3%
to 94.1%. The six-joint confidence gate passes in 98/100
selected crops, chosen as the two tallest matched players every sixth sampled frame.
This favors visible large players and is availability, not anatomical accuracy. Fixed
first-frame calibration remains valid in only 1.7–8.3% of sampled frames by the camera guard;
moving broadcast cameras require dynamic calibration beyond this scoped implementation.
No short clips were stitched into the longitudinal study or used for training.

## Verification and delivery

The full test suite and lint check pass; the locked dependency dry run requires no changes.
Known-step, stationary-noise and constant software controls all pass. They establish algorithm
behavior on synthetic data, not clinical performance. The end-to-end notebook executed all
10 code cells. The local standalone HTML dashboard rendered
without JavaScript errors and all three player-selector options were checked.
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
