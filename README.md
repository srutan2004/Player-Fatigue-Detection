# Football movement and gait analysis

BCSE407L Computer Vision mini-project. **All four phases are implemented and executed:** setup, dataset intake, one detector experiment, tracking, calibration, native-rate pose, quality-gated metrics, decline analysis, proxy validation and interactive reporting. The study covers **three outfield players, two matches and four 25-minute segments**, with a separate five-clip broadcast audit. See [FINAL_REPORT.md](FINAL_REPORT.md) for the final findings and [PHASE_1_3_REPORT.md](PHASE_1_3_REPORT.md) for measurement details.

The software is complete; physiological fatigue is not validated. Only one player has qualified early/late movement periods, and no nonoverlapping gait windows pass the exposure gate. Unsupported results remain missing. The original requirement to force a change point for every metric would produce misleading findings, so the revised roadmap allows insufficient data and no supported change.

The roadmap is [FOUR_PHASE_IMPLEMENTATION_PLAN.md](FOUR_PHASE_IMPLEMENTATION_PLAN.md); methodological detail is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

![Interactive dashboard preview](results/phase4/dashboard_preview.png)

The repository includes source, clean notebooks, configuration, fitted study calibration, small derived result tables, plots and the standalone HTML dashboard. Download [dashboard.html](results/phase4/dashboard.html) and open it locally; GitHub displays HTML source rather than running the dashboard. Raw dataset files, pretrained weight binaries, native-frame caches, executed notebooks and footage-based previews stay local. Their absence from a fresh clone is expected; the dataset and setup steps below are required to rerun inference. This repository does not redistribute match footage.

## Setup

Tested: RTX 4060 Laptop GPU (8 GB), Ryzen 7 7840HS, nominal 16 GB RAM, Python 3.12. CUDA PyTorch 2.8.0/torchvision 0.23.0 are pinned to the explicit CUDA 12.8 index. Runtime libraries ship with the wheels; a compatible NVIDIA driver is required.

```powershell
git clone https://github.com/srutan2004/Player-Fatigue-Detection.git
cd Player-Fatigue-Detection
uv sync --locked --group dev --group notebook
uv run --no-sync python -m src.cli check-environment --smoke-models
```

Alternatively use `scripts/setup.ps1`. Actual CUDA multiplication, torchvision NMS and pretrained forward passes are checked. `uv.lock` is authoritative; `requirements.txt` is a pip fallback after installing compatible CUDA torch/torchvision. YOLOv8-Pose supplies anatomy; MediaPipe is unnecessary.

## Dataset

The principal source is [SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2), release v1.2 pinned to `325bd4c57ee8e135ab652f3921d274526ada496b`. Accept its access conditions and authenticate in the terminal:

```powershell
uv run --no-sync hf auth login
uv run --no-sync python -m src.cli dataset-probe --match 117093
```

Browser login alone does not authenticate Python. Enter credentials privately in the terminal; no tokens belong in reports or configs. Selected files are both halves/GSR annotations of 117093 and 128057, camera assets and curated MOT clips/boxes. Files keep upstream paths under `data/hub/`, avoiding duplicate multi-GB video copies. Download manifests record hashes, sizes and the pinned commit.

```powershell
uv run --no-sync python -m src.cli dataset-probe --match 128057 --output results/phase1/dataset_128057.json
uv run --no-sync python -m src.cli dataset-download --probe results/phase1/dataset_128057.json --max-gib 12 `
  --file videos/128057/128057_panorama_1st_half.mp4 `
  --file videos/128057/128057_panorama_2nd_half.mp4 `
  --file gsr/128057/128057_1st.json `
  --file gsr/128057/128057_2nd.json
```

Downloads are explicitly bounded and resumable. Multi-GB JSON is streamed into SQLite. Source overlays verified GSR one-based suffix indices, MOT zero-based indices, and video PTS starting at one second. Subtracting the PTS origin avoids annotation shifts. GSR box semantics differ from anatomical MOT boxes; GSR boxes are not training labels. Inspect the actual release and [upstream schema notes](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md).

For a fresh clone, acquire the required videos and annotations for both matches. Build each reference index using `reference-index SOURCE_JSON --output data/processed/MATCH_first.sqlite --index-base 1` (use `MATCH_second.sqlite` for the second half). The exact selected upstream files and hashes are recorded in `data/manifests/downloads/`. Remove the `pipeline.cached_tracks` setting from the four study YAML files when those local tracker caches are absent, so extraction performs fresh detection. The published fitted calibration mappings can be reused for the corresponding unchanged source cameras. Study identity association requires the reference indexes. The included result tables let you inspect the final dashboard without downloading the footage; recomputing sensitivity requires local frame-level caches generated by the study workflow.

## Phase 1 and detector experiment

```powershell
uv run --no-sync python -m src.cli inspect --config configs/pilot.yaml
uv run --no-sync python -m src.cli pilot --config configs/pilot.yaml
```

The 60-second, 2-fps pilot audits image quality/runtime. Its IDs are frame-local, and sparse samples cannot support cadence. Manual crop judgments are saved separately from confidence scores.

One YOLOv8n experiment used **387 train / 100 validation / 100 held-out-match tiles**, after 13 doubtful training images were quarantined. Source frames were split before tiling, with a 30-second development gap. Staff/referee proposals were visually reviewed; original MOT alignment was sampled. Labels are partly model-assisted, not a new benchmark.

**15 epochs**, batch 4, freeze 10, AMP and seed 42 completed in approximately **258 seconds including evaluations**. Validation mAP50–95: pretrained **0.710**, adapted **0.680**. The baseline was therefore retained; selected held-out mAP50–95 was **0.723**. Do not run another adaptation automatically.

```powershell
# For a fresh dataset; inspect/correct labels before marking the manifest reviewed:
uv run --no-sync python -m src.cli prepare-detector configs/detector_sources.json
uv run --no-sync python -m src.cli train-detector --epochs 15
```

The single-run registry refuses repeat training. Scores depend on label quality. The principal long videos are analysis inputs, not millions of training frames.

## Calibration, tracking and poses

The curved panorama fails a single raw-pixel homography. `calibrate.py` tests homography, local affine interpolation and rational planar warps on withheld reference points. The simplest passing method is chosen, with convex-hull/denominator guards. Both study cameras select a quadratic rational warp. Ordinary planar views can use `pixel_to_world` with a validated homography. Ground-plane calibration cannot turn elevated hips into ground positions or vertical meters.

```powershell
uv run --no-sync python -m src.cli fit-field data/hub/raw/117093/117093_keypoints.json --output results/phase2/calibration/117093.json
uv run --no-sync python scripts/evaluate_tracking.py
uv run --no-sync python scripts/run_study.py
uv run --no-sync python scripts/finalize_phase3.py
```

The driver resumes completed stages from [study.json](data/manifests/study.json) and runs GPU work serially. Completed caches must match source/settings; partial attempts never count as complete. Inspect a failure and choose a fresh output directory before retrying vision stages.

ByteTrack runs at 12.5 fps; native observations are 25 fps. Temporary tracklet IDs and persistent player identities are separate. The principal study uses **annotation-assisted identity association**, constrained by kit color, independently detected ground anchors and global matching. Reference positions never replace measured coordinates. This accelerates review but is not autonomous identity accuracy. Independent raw tracker tables remain available.

The separate held-out MOT audit uses no hints. It reports known-player recall by size and consecutive observed switches, not full IDF1/HOTA or all-person precision, because staff labels are incomplete.

Losses, ambiguous associations, invalid crops and failed calibration remain gaps. Short 40-ms box propagation is tagged. Shadow-aware HSV checks use only dominant kit palettes; mixed palettes remain unknown. Recently verified, still-visible tracklets have a bounded 0.8-second continuity check with motion/overlap/reference limits. Missing detections and new tracklets are never filled. Background registration flags camera motion rather than recalibrating moving footage. The staged study retains one consistent detector box-bottom ground anchor.

Pretrained **YOLOv8m-Pose** is selected after a fixed ten-minute audit: joint-confidence coverage improved from 173/255 to 249/255 crops compared with YOLOv8s-Pose, with anatomical overlays inspected. No pose training is used. Native padded crops are batched at most eight, and coordinates are restored using their clipped source origins. Low-confidence anatomy remains missing.

```powershell
uv run --no-sync python -m src.cli extract --config configs/study_117093_second.yaml
uv run --no-sync python -m src.cli pose-pass results/phase2/study/117093_second_identity_v3/keypoints_positions.csv `
  --config configs/study_117093_second.yaml --output results/phase3/study_identity_v3/117093_second
uv run --no-sync python -m src.cli metrics results/phase3/study_identity_v3/117093_second/keypoints_positions.csv `
  --output results/metrics_csv/117093_second
```

Existing vision outputs are preserved; metrics can be recomputed cheaply. For a new permitted local video, edit the YAML paths, duration/offset and validated ground mapping. Run `select-player --config ...` to save initial ByteTrack IDs, then set `reference_assisted_identity: false` and each target's `initial_tracklet_id`. Later ambiguous reappearances require review.

For reviewed reappearances, each target can supply `reviewed_tracklets` with non-overlapping source-video intervals (`start_seconds`, `end_seconds`, `tracklet_id`, `reviewed: true`). Outside reviewed intervals the target remains missing. Field clicks on local Windows are supported by:

```powershell
uv run --no-sync python scripts/calibrate_click.py --config configs/my_video.yaml `
  --world-points configs/field_references.example.json --output data/calibration/my_points.json
```

Use known visible pitch features and at least eight distributed correspondences so some can be withheld for validation. The example assumes a 105×68-meter pitch. The supplied study assets make this interactive step unnecessary. The click UI is a local desktop helper, not a Colab widget.

## Metrics and eligibility

| Feature | Meaning and units |
|---|---|
| Average speed | Smoothed observed ground-path distance / valid time, m/s |
| Peak speed | Maximum supported approximately one-second average, m/s |
| High-intensity running rate | Distance at ≥5.5 m/s per valid observed minute |
| Cadence | Same-leg cycles/s, not alternating step frequency |
| Stride-length proxy | Ground-path distance per same-leg cycle, meters |
| Knee excursion | 95th–5th percentile 2D flexion per cycle, degrees |
| Vertical-motion proxy | Hip image oscillation / body-box height, dimensionless |

Smoothing and displacement never cross gaps, clip boundaries or tracklet switches. Speeds above 12 m/s are rejected. Gait requires running at 2–9 m/s for at least four seconds and ≤30° heading spread over two seconds; severe overlapping boxes are excluded. These guards do not classify every jump/tackle perfectly.

Cadence requires at least four credible same-leg peaks, amplitude/periodicity support and compatible left/right periods. Small confidence holes use Lomb–Scargle on observed samples; coordinates are never filled. At least 80% signal observations and no missing span over 0.16 s are required. Peaks are oscillation landmarks, not sensor-validated foot strikes.

120-second windows use 60-second display steps and 120-second inference steps. Movement needs 70% observed ground-position exposure. Gait needs 20 accepted-cycle seconds and ten same-leg cycles, plus feature-specific finite observations. Unsupported values stay NaN. Speed, source body scale and projected motion strata help later comparisons but do not remove all projection confounding.

## Outputs and verification

| Folder | Outputs |
|---|---|
| `results/phase1/` | Environment, dataset/source/schema audits and manually reviewed sparse pilot |
| `results/phase2/` | Training comparison, calibrated mappings, autonomous audit, study track/anchor CSVs, links, gaps, camera checks, previews |
| `results/phase3/study_identity_v3/` | Final source-coordinate keypoints, flags, native skeleton audit montage, runtime/VRAM; earlier runs remain archived separately |
| `results/metrics_csv/` | Frame Parquet, running bouts, gait cycles, window CSVs and comparability strata |
| `results/plots/` | Combined metric plots, pitch trajectories and coverage |
| `results/phase4/` | Baselines, normalized windows, trends, sensitivity, paired comparisons, status tables, eight metric panels, dashboard, annotated demo and broadcast audit |
| `results/completion_report.json` | Verified completion of phases 1–4, with scientific availability separate |
| `results/reproducibility.json` | Delivered source/artifact hashes and upstream provenance |

Source-frame/video time and configured match time remain separate. The second-half offset is explicitly 2700 seconds; it omits first-half stoppage time. Sequential decoding preserves annotation indices but creates prefix decode overhead. Sampled/montage previews are labeled honestly.

```powershell
uv run --no-sync jupyter lab notebooks/pipeline_demo.ipynb
uv run --no-sync pytest -q
uv run --no-sync ruff check src scripts tests main.py
```

[pipeline_demo.ipynb](notebooks/pipeline_demo.ipynb) runs one sample through the same modules and reuses complete caches. It uses the combined study for final analysis when available, and can produce a one-clip analysis otherwise. Tests use encoded video and independent synthetic controls for nonzero PTS origin, gaps, crop offsets, calibration, known speed/cadence, camera motion and missing observations. See [LIMITATIONS.md](LIMITATIONS.md). Video, credentials, weights and generated outputs are excluded from Git.

## Phase 4: analyze and deliver

```powershell
# Recompute analysis from final metrics; CPU only, no repeat training:
uv run --no-sync python -m src.cli analyze
# Optional rerun of the completed separate broadcast audit (GPU):
uv run --no-sync python -m src.broadcast_audit
# Render the continuous reviewed study excerpt from cached joints:
uv run --no-sync python scripts/create_annotated_demo.py
uv run --no-sync python scripts/execute_notebook.py
```

Open [dashboard.html](results/phase4/dashboard.html) in a browser. It is standalone and embeds Plotly locally; no server or CDN is required. Choose a player to inspect physical values, own-baseline changes, availability/status, early/late summaries and observed ground trajectories. The native 25-fps [annotated demo](results/phase4/annotated_demo.mp4) shows boxes, projected skeleton, valid speed and a minimap on a continuous ten-second excerpt.

The own-player baseline uses the first 600 elapsed seconds of the early clip, at least three eligible windows and 420 eligible movement seconds. A_blue_36 (389.88 seconds) and B_white_17 (two eligible windows) fail this gate; their normalized movement trends are unavailable. B_white_9 has an available baseline. Zero baselines never produce percentage changes.

Descriptive trends use six or more contiguous nonoverlapping windows and lag-1 HAC uncertainty, with Holm correction per player. PELT requires eight windows, three before/after, a material downward change, and penalty agreement. Sensitivity checks smoothing 0.6/0.8/1.2 seconds, window sizes 120/180/240 seconds, and penalties 2/3/4 times log(n). Missing windows and clip boundaries break sequences. A candidate must survive smoothing and window changes; no supported onset remains in this study. Cadence and vertical motion have no universally negative fatigue direction and stay descriptive when available.

Only A_blue_36 qualifies for early/late comparison: average speed **−12.5%**, sustained peak **−10.2%**, running-speed p90 **−5.8%**, high-intensity distance rate **−37.0%**. These are selected segment changes, not full-half averages or fatigue labels. With one eligible paired player, no population significance test is run. Verified substitutions can be supplied in `study.json` under `verified_substitutions`, keyed by player ID, with `match_seconds`, `verified: true`, and a nonempty `source`; none are provided for this cohort.

The five independent [SoccerNet-GSR test clips](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024) provide a short broadcast stress audit: 300 frames at about 2 fps, detection recall 85.3–94.1%, and pose availability on a deliberately bounded large-player sample. These clips cannot estimate cadence/fatigue. Camera pan/zoom invalidates a fixed map for most sampled frames. The downloader selects compressed archive members using checked byte ranges and CRC/SHA checks instead of copying the entire archive. See the [official upstream project](https://github.com/soccernet/sn-gamestate).

`scripts/finalize_project.py` verifies tests, lint, lock, completed notebook, rendered dashboard QA, controls, video and plots before writing the completion and reproducibility reports. Run dashboard QA with `uv run --isolated --no-project --with playwright python scripts/dashboard_qa.py`; it uses the installed Edge browser and is separate from runtime dependencies. Then run `uv run --no-sync python scripts/finalize_project.py`. The delivered QA report is in `results/phase4/dashboard_qa.json`.
