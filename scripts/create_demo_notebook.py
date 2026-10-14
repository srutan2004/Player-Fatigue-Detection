"""Write a portable notebook that orchestrates all four tested project phases."""
# ruff: noqa: E501

import json

from src.common import ROOT


def create_notebook():
    """Keep the notebook free of duplicate inference/metric implementations."""
    cells = []

    def markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text):
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": text.splitlines(True),
            }
        )

    markdown(
        "# Football movement: completed Phases 1–4\n\nThis notebook runs intake → tracking → native-rate pose bouts → quality-gated measurements → decline analysis. Cached complete outputs are reused. The final study covers four 25-minute segments. No additional detector training occurs. See `FINAL_REPORT.md` and `LIMITATIONS.md`; movement decline is not a physiological fatigue diagnosis.\n"
    )
    code(
        "from pathlib import Path\nimport sys, json\nROOT = Path.cwd()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nassert (ROOT / 'src').is_dir(), 'Open this notebook from the project directory'\nsys.path.insert(0, str(ROOT))\nfrom src.common import load_config\nCONFIG = ROOT / 'configs/study_117093_first.yaml'\nconfig = load_config(CONFIG)\nconfig['pipeline']['targets'] = [{'player_id': 'A_blue_36', 'reference_player_id': '473125'}]\nconfig['pipeline']['pose_mode'] = 'running_bouts'\nconfig['pilot']['pose_model'] = 'yolov8m-pose.pt'\nTRACKS = config['output_dir']\nPOSES = config['pipeline'].get('pose_output_dir', ROOT / 'results/phase3/study/117093_first')\nMETRICS = ROOT / 'results/metrics_csv/117093_first'\n"
    )
    markdown(
        "## Environment and input\n\nInstall once outside the notebook: `uv sync --locked --group dev --group notebook`. Authenticate using `uv run --no-sync hf auth login` and the dataset's accepted access conditions. Videos are in `data/hub/`, preserving upstream filenames. A fresh machine needs the files listed by the checked download manifests. Initial ID selection and field correspondences are configurable; the study uses supplied annotations to assist identity only.\n"
    )
    code(
        "from src.environment import check_environment\nreport_path = ROOT / 'results/phase1/environment.json'\nenvironment = json.loads(report_path.read_text()) if report_path.exists() else check_environment(report_path, True, 'cuda:0')\nassert environment['status'] == 'passed'\nfrom src.video_io import probe_video\nprobe_video(config['video']['path'])\n"
    )
    markdown(
        "## Audited detector and calibration\n\nOne 15-epoch detector experiment was run. Validation selected the original YOLOv8n. A direct homography fails on the curved panoramic pixel space; a quadratic rational ground warp passed a withheld field-point check. Ordinary planar views can use `pixel_to_world` with a validated homography. Elevated hips remain image-space joints.\n"
    )
    code(
        "training = json.loads((ROOT / 'results/phase2/training/bounded_run.json').read_text())\nassert training['status'] == 'completed'\nfrom src.calibrate import GroundMapping\nmapping = GroundMapping.load(config['pipeline']['ground_mapping'])\nassert mapping.validated\n{'chosen_detector': training['selection'], 'field_method': mapping.method, 'validation': mapping.payload['validation']}\n"
    )
    markdown(
        "## Track selected player\n\nTracklets, ambiguous links, camera checks and gaps are saved separately. The stored first-half run includes an extra early-only pilot player; it is excluded from this cohort before pose extraction. Existing partial runs are never treated as complete.\n"
    )
    code(
        "from src.detect_track import extract_segment\nif not (TRACKS / 'extraction_report.json').exists():\n    extract_segment(config)\nimport pandas as pd\ntracks = pd.read_csv(TRACKS / 'keypoints_positions.csv')\ntracks = tracks[tracks.player_id == 'A_blue_36']\nPOSES.mkdir(parents=True, exist_ok=True)\nselected = POSES / 'selected_tracks.csv'\ntracks.to_csv(selected, index=False)\ntracks.groupby('observation_status').size()\n"
    )
    markdown(
        "## Native 25-fps pose extraction\n\nOnly eligible running bouts spend pose compute. Missing/occluded joints remain missing; each crop origin is restored to the original source coordinates. Confidence is a gate, not anatomical ground truth. Gait cycles and vertical motion remain 2D proxies.\n"
    )
    code(
        "from src.pose_pass import pose_from_tracks\nif not (POSES / 'pose_report.json').exists():\n    pose_from_tracks(config, selected, POSES, batch_size=8)\njson.loads((POSES / 'pose_report.json').read_text())\n"
    )
    markdown(
        "## Movement, cycles, windows and plots\n\n120-second windows need 70% ground-position exposure for movement. Gait needs 20 accepted running seconds and ten same-leg cycles. Ineligible values are NaN. Display windows overlap; separate non-overlapping windows support later inference. Compare speed, source body scale and projected image-motion strata before discussing mechanics.\n"
    )
    code(
        "from src.process_metrics import process_metrics\nsource = POSES / 'keypoints_positions_camera_qualified.csv'\nif not source.exists(): source = POSES / 'keypoints_positions.csv'\nmeasurements = process_metrics(source, METRICS)\nwindows = pd.read_csv(METRICS / 'windowed_metrics_nonoverlap.csv')\ndisplay(windows)\nmeasurements\n"
    )
    code(
        "from IPython.display import Image, display\nfor name in ['avg_speed', 'stride_frequency', 'stride_length', 'knee_angle_range']:\n    display(Image(filename=str(METRICS / 'plots' / f'{name}.png')))\n"
    )
    markdown(
        "## Full study and decline analysis\n\n`uv run --no-sync python scripts/run_study.py` processes/resumes all four 25-minute segments. `uv run --no-sync python scripts/finalize_phase3.py` qualifies camera registration and combines the cohort tables. The following stage uses these completed tables. Missing baseline exposure, gait or between-half periods cannot become fabricated fatigue timestamps.\n"
    )
    code(
        "from src.fatigue_analysis import run_analysis\nANALYSIS = ROOT / 'results/phase4'\nSTUDY = ROOT / 'data/manifests/study.json'\nTABLES = ROOT / 'results/metrics_csv'\nif not (TABLES / 'all_windowed_metrics_nonoverlap.csv').exists():\n    # A fresh one-clip demo can finish without requiring the other three clips.\n    import shutil\n    from src.common import write_json\n    TABLES = ANALYSIS / 'single_clip_input'\n    TABLES.mkdir(parents=True, exist_ok=True)\n    manifest = json.loads(STUDY.read_text())\n    manifest['clips'] = [c for c in manifest['clips'] if c['clip_name'] == windows.clip_name.iloc[0]]\n    STUDY = TABLES / 'study.json'\n    write_json(STUDY, manifest)\n    windows.to_csv(TABLES / 'all_windowed_metrics_nonoverlap.csv', index=False)\n    shutil.copytree(METRICS, TABLES / windows.clip_name.iloc[0], dirs_exist_ok=True)\n    frame_data = pd.read_parquet(METRICS / 'frame_metrics.parquet')\n    rows = []\n    for player, part in frame_data.groupby('player_id'):\n        rows.append({'player_id': player, 'half': config['video']['half'], 'valid_movement_seconds': part.valid_dt.sum(), 'movement_frame_coverage': part.movement_valid.mean()})\n    pd.DataFrame(rows).to_csv(TABLES / 'study_coverage.csv', index=False)\n    ANALYSIS = ANALYSIS / 'single_clip_demo'\nfinal_report = run_analysis(STUDY, TABLES, ANALYSIS)\ndisplay(pd.read_csv(ANALYSIS / 'analysis_status.csv'))\ndisplay(pd.read_csv(ANALYSIS / 'early_late_comparison.csv'))\n"
    )
    code(
        "for name in ['avg_speed', 'peak_speed', 'p90_running_speed', 'high_intensity_distance_per_minute']:\n    display(Image(filename=str(ANALYSIS / 'plots' / f'{name}_decline.png')))\nfrom IPython.display import FileLink\ndisplay(FileLink(str(ANALYSIS / 'dashboard.html')))\njson.loads((ANALYSIS / 'synthetic_controls.json').read_text())\n"
    )
    markdown(
        "## Independent broadcast robustness and annotated evidence\n\nFive short public SoccerNet-GSR test clips are sampled at about 2 fps to audit detection, pose availability and camera motion. These samples cannot support native-rate cadence or long-term fatigue. `uv run --no-sync python -m src.broadcast_audit` reruns the separate audit; `uv run --no-sync python scripts/create_annotated_demo.py` renders a continuous native-rate study excerpt from saved joints without additional model inference.\n"
    )
    code(
        "broadcast = ANALYSIS / 'broadcast_audit/report.json'\nif broadcast.exists():\n    display(pd.read_csv(ANALYSIS / 'broadcast_audit/clips.csv'))\nelse:\n    print('Run the optional broadcast audit command above for this separate stress test.')\nfrom IPython.display import Video\npreview = ANALYSIS / 'annotated_demo.mp4'\nif preview.exists(): display(Video(str(preview), embed=False))\n"
    )
    for index, cell in enumerate(cells):
        cell["id"] = f"phase1234-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "notebooks/pipeline_demo.ipynb").write_text(
        json.dumps(notebook, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    create_notebook()
