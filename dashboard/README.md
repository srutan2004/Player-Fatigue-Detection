# STRIDE — Player Movement Lab

React interface for the project's measured soccer movement and gait evidence. The exported dashboard is a single offline HTML file with embedded React, Plotly, styles and study data. It supports local `file://` access without fetching scripts, fonts or dataset files.

## Build and view

From the project root, with Node.js and the Python environment installed:

```powershell
npm --prefix dashboard ci
npm --prefix dashboard run build
uv run --no-sync python scripts/render_dashboard.py
```

Open `results/phase4/dashboard.html` in a browser. A prebuilt client is committed at `dashboard/assets/app.js`; Node.js is needed only after editing the JSX/build source. The normal Python analysis stage also calls the exporter.

## Source files

| File | Purpose |
|---|---|
| `app.jsx` | Components, navigation, filters, scientific chart views and exports |
| `styles.css` | Glass surfaces, gradients, reflections, themes and responsive layouts |
| `build.mjs` | esbuild production bundle with React included |
| `package-lock.json` | Locked frontend dependency graph |
| `assets/app.js` | Committed offline production client |
| `../src/dashboard.py` | Evidence payload and self-contained HTML export |
| `../scripts/render_dashboard.py` | Rebuild presentation from existing saved measurements |
| `../scripts/dashboard_qa.py` | Headless browser interaction, export and responsive checks |

## Dashboard sections

- **Overview:** Qualified movement summaries, selected metric timeline, early/late bars and coverage.
- **Movement:** All eight metric choices, physical/baseline modes, analysis statuses and window details.
- **Gait explorer:** Accepted cycle measurements, speed/leg filters, running-speed comparisons and source body size.
- **Pitch map:** Gapped measured trajectories, time scrubbing, playback and sampled occupancy.
- **Evaluation:** Held-out detector metrics, five-clip broadcast recall, pose confidence availability, field-reference error, independent tracker audit and software controls.
- **Study & methods:** Dataset, models, calibration assumptions, thresholds, baseline exposure, HAC uncertainty, sensitivity, candidate robustness and paired-test eligibility.

## Evidence behavior

The interface changes presentation only. Missing measurements stay null, map thinning preserves native gaps and identity switches, and chart lines do not bridge excluded windows. Individual gait-cycle summaries remain descriptive. Accepted onset lines appear only for candidates passing the recorded analysis checks. The coverage ring reports observed movement time, not fatigue probability. Detection AP, tracking recall, pose availability and calibration error are separately labeled.

Period means and baseline normalization use the existing scientific outputs. Exports preserve the selected player/period or the complete recorded tables. Cursor positions are unavailable during unprocessed intervals; trajectory playback skips to actual observed samples. Occupancy uses approximately one-second display samples rather than full-match positions.

## Checks

```powershell
npm --prefix dashboard run format:check
uv run --no-sync pytest -q
uv run --isolated --no-project --with playwright python scripts/dashboard_qa.py
```

The browser QA uses installed Microsoft Edge. It verifies all three player selections, measured values, missing-baseline states, chart-point inspection, CSV export, gait filters, pitch playback, all six mobile views, both themes, deep links and zero external runtime requests. Screenshots and the QA report are saved under `results/phase4/`.
