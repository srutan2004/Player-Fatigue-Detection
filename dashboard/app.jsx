import { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";

const DATA = JSON.parse(document.getElementById("study-data").textContent);
const MOVEMENT = DATA.metrics.filter((m) => m.family === "movement");
const GAIT = DATA.metrics.filter((m) => m.family === "gait");
const PLAYERS = [...new Set(DATA.windows.map((w) => w.player_id))].sort();
const COLORS = { early: "#55c9fa", late: "#aa9bff", accent: "#76e5b0" };
const palette = (theme) =>
  theme === "light"
    ? { early: "#1788b5", late: "#7863cc", accent: "#258463" }
    : COLORS;
const LABELS = {
  supported_candidate: "Supported candidate",
  no_supported_change: "No supported change",
  quality_confounded: "Sensitivity not supported",
  unavailable_baseline: "Baseline unavailable",
  insufficient_data: "Insufficient exposure",
  descriptive_only: "Descriptive only",
  candidate_pending_sensitivity: "Under sensitivity review",
};
const DESCRIPTIONS = {
  avg_speed: "Smoothed observed ground-path distance divided by valid time.",
  peak_speed: "Highest supported approximately one-second average speed.",
  p90_running_speed:
    "90th percentile of observed speeds during running at ≥2 m/s.",
  high_intensity_distance_per_minute:
    "Distance at ≥5.5 m/s per valid observed minute.",
  stride_frequency:
    "Same-leg running cycles per second, rather than alternating steps.",
  stride_length: "Measured ground-path distance per accepted same-leg cycle.",
  vertical_oscillation:
    "Projected hip oscillation divided by body height; a dimensionless image proxy.",
  knee_angle_range:
    "Robust projected knee-flexion excursion (95th–5th percentile) per cycle.",
};
const NAV = [
  ["overview", "Overview", "grid"],
  ["movement", "Movement", "activity"],
  ["gait", "Gait explorer", "runner"],
  ["pitch", "Pitch map", "pitch"],
  ["evaluation", "Evaluation", "chart"],
  ["methods", "Study & methods", "layers"],
];
const fmt = (value, digits = 2) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
const pct = (value, digits = 1) =>
  value == null ? "—" : `${fmt(value * 100, digits)}%`;
const human = (text) => String(text ?? "—").replaceAll("_", " ");
const jersey = (player) => player.split("_").at(-1);
const halfOf = (clip) => (clip.endsWith("first") ? 1 : 2);
const allows = (half, period) =>
  period === "both" || half === (period === "early" ? 1 : 2);
const finite = (values) => values.filter(Number.isFinite);
const median = (values) => {
  const a = finite(values).sort((a, b) => a - b);
  return a.length
    ? (a[Math.floor((a.length - 1) / 2)] + a[Math.ceil((a.length - 1) / 2)]) / 2
    : null;
};
const byPlayer = (rows, player) =>
  rows.filter((row) => row.player_id === player);
const metricInfo = (metric) => DATA.metrics.find((m) => m.id === metric);

function Icon({ name, size = 20, ...props }) {
  const paths = {
    grid: "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z",
    activity: "M2 12h4l3-8 5 16 3-8h5",
    runner:
      "M15 5a2 2 0 1 0 0-4 2 2 0 0 0 0 4M9 10l4-3 3 4 5 1M13 8l-2 6 5 3 1 5M11 14l-4 5H2",
    pitch:
      "M3 4h18v16H3zM12 4v16M3 9h4v6H3M21 9h-4v6h4M14 12a2 2 0 1 1-4 0 2 2 0 0 1 4 0",
    chart: "M4 3v18h17M8 16v-5M13 16V7M18 16v-9",
    layers: "m12 3 10 5-10 5L2 8zM2 12l10 5 10-5M2 16l10 5 10-5",
    arrow: "M4 12h16M14 6l6 6-6 6",
    download: "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5",
    sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5",
    moon: "M20 15A9 9 0 0 1 9 3a9 9 0 1 0 11 12",
    check: "m5 12 4 4L19 6",
    info: "M12 10v7M12 7h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
    chevron: "m8 4 8 8-8 8",
    search: "M17 17l5 5M19 10a9 9 0 1 1-18 0 9 9 0 0 1 18 0",
    play: "m8 4 12 8-12 8z",
    pause: "M8 4v16M16 4v16",
    expand: "M8 3H3v5M16 3h5v5M3 16v5h5M16 21h5v-5",
    shield: "m12 2 9 4v6c0 5-9 10-9 10S3 17 3 12V6zM8 12l3 3 5-6",
    clock: "M12 7v5l3 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
    external: "M14 3h7v7M21 3l-11 11M10 3H3v18h18v-7",
    close: "m5 5 14 14M19 5 5 19",
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d={paths[name] || paths.info} />
    </svg>
  );
}
function Pill({ children, tone = "neutral" }) {
  return <span className={`pill ${tone}`}>{children}</span>;
}
function Empty({ title, text, action }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Icon name="activity" size={27} />
      </div>
      <h3>{title}</h3>
      <p>{text}</p>
      {action}
    </div>
  );
}
function SectionHead({ eyebrow, title, text, children }) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
        {text && <p>{text}</p>}
      </div>
      {children}
    </div>
  );
}
function Glass({ className = "", children, ...props }) {
  return (
    <section className={`glass ${className}`} {...props}>
      {children}
    </section>
  );
}
function Segments({ value, onChange, options, label }) {
  return (
    <div className="segments" role="group" aria-label={label}>
      {options.map(([id, text]) => (
        <button
          key={id}
          type="button"
          aria-pressed={value === id}
          className={value === id ? "selected" : ""}
          onClick={() => onChange(id)}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

function Plot({ traces, layout = {}, theme, height = 310, id, onPoint }) {
  const ref = useRef(null),
    callback = useRef(onPoint);
  const [error, setError] = useState(null);
  callback.current = onPoint;
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    let cancelled = false;
    const light = theme === "light";
    const axis = {
      gridcolor: light ? "rgba(36,58,90,.08)" : "rgba(179,200,255,.07)",
      zeroline: false,
      tickfont: { size: 11 },
      showline: false,
    };
    const full = {
      autosize: true,
      height,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: {
        family: "Segoe UI, sans-serif",
        color: light ? "#4e6380" : "#9bacc7",
        size: 12,
      },
      margin: { l: 52, r: 22, t: 25, b: 45 },
      hovermode: "closest",
      hoverlabel: {
        bgcolor: light ? "#fff" : "#1c2945",
        font: { color: light ? "#122642" : "#eef3ff" },
        bordercolor: "rgba(140,169,210,.3)",
      },
      showlegend: false,
      ...layout,
      xaxis: { ...axis, ...layout.xaxis },
      yaxis: { ...axis, ...layout.yaxis },
    };
    window.Plotly.react(element, traces, full, {
      responsive: true,
      displaylogo: false,
      displayModeBar: "hover",
      modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
      toImageButtonOptions: {
        filename: `stride-${id || "chart"}`,
        format: "png",
        scale: 2,
      },
    })
      .then(() => {
        if (cancelled || !element.isConnected) return;
        element.dataset.ready = "true";
        element.removeAllListeners?.("plotly_click");
        if (callback.current)
          element.on("plotly_click", (event) =>
            callback.current(event.points[0]),
          );
      })
      .catch((e) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [traces, layout, theme, height, id]);
  useEffect(() => {
    const element = ref.current;
    let ticket;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(ticket);
      ticket = requestAnimationFrame(() => {
        if (element?.isConnected && element._fullLayout)
          window.Plotly.Plots.resize(element);
      });
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(ticket);
      window.Plotly.purge(element);
    };
  }, []);
  return error ? (
    <Empty title="Chart unavailable" text={error} />
  ) : (
    <div
      ref={ref}
      className="plot"
      id={id}
      style={{ minHeight: height }}
      aria-label={id?.replaceAll("-", " ")}
      role="img"
    />
  );
}

function download(text, name, type = "text/csv;charset=utf-8") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function csv(rows) {
  if (!rows.length) return "";
  const keys = [...new Set(rows.flatMap(Object.keys))];
  const cell = (v) => {
    const s =
      v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v);
    return /[",\n\r]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
  };
  return [
    keys.join(","),
    ...rows.map((r) => keys.map((k) => cell(r[k])).join(",")),
  ].join("\n");
}
function ExportMenu({ player, period }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const close = (e) => {
      if (!ref.current?.contains(e.target)) setOpen(false);
    };
    const esc = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", esc);
    };
  }, []);
  return (
    <div className="export-wrap" ref={ref}>
      <button
        className="button primary"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-controls="export-menu"
      >
        <Icon name="download" size={16} /> Export data
      </button>
      {open && (
        <div id="export-menu" className="export-menu glass">
          <span className="menu-label">CURRENT SELECTION</span>
          <button
            onClick={() => {
              download(
                csv(
                  byPlayer(DATA.windows, player).filter((w) =>
                    allows(w.half, period),
                  ),
                ),
                `${player}-${period}-windows.csv`,
              );
              setOpen(false);
            }}
          >
            Player window metrics <Icon name="download" size={15} />
          </button>
          <span className="menu-label">COMPLETE STUDY</span>
          {Object.keys(DATA.exports).map((name) => (
            <button
              key={name}
              onClick={() => {
                download(
                  DATA.exports[name],
                  name,
                  name.endsWith(".json")
                    ? "application/json"
                    : "text/csv;charset=utf-8",
                );
                setOpen(false);
              }}
            >
              {human(name.replace(/\.(csv|json)$/, ""))}
              <Icon name="download" size={15} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
function Table({
  rows,
  columns,
  empty = "No observations for this selection",
  filename,
}) {
  const [query, setQuery] = useState(""),
    [page, setPage] = useState(0);
  const filtered = rows.filter(
    (row) =>
      !query ||
      columns.some((c) =>
        String(row[c.key] ?? "")
          .toLowerCase()
          .includes(query.toLowerCase()),
      ),
  );
  useEffect(() => setPage(0), [rows, query]);
  const pages = Math.max(1, Math.ceil(filtered.length / 10)),
    safePage = Math.min(page, pages - 1);
  return (
    <div className="data-table">
      <div className="table-tools">
        <label className="search">
          <Icon name="search" size={16} />
          <input
            aria-label="Search table"
            placeholder="Search observations…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <span>{filtered.length} records</span>
        {filename && (
          <button
            className="button subtle small"
            disabled={!filtered.length}
            onClick={() => download(csv(filtered), filename)}
          >
            <Icon name="download" size={14} /> CSV
          </button>
        )}
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c.key}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.slice(safePage * 10, (safePage + 1) * 10).map((r, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c.key}>
                    {c.render
                      ? c.render(r[c.key], r)
                      : r[c.key] == null
                        ? "—"
                        : String(r[c.key])}
                  </td>
                ))}
              </tr>
            ))}
            {!filtered.length && (
              <tr>
                <td className="table-empty" colSpan={columns.length}>
                  {empty}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {pages > 1 && (
        <div className="pagination">
          <span>
            Page {safePage + 1} of {pages}
          </span>
          <div>
            <button
              className="button subtle small"
              disabled={safePage === 0}
              onClick={() => setPage(safePage - 1)}
            >
              Previous
            </button>
            <button
              className="button subtle small"
              disabled={safePage === pages - 1}
              onClick={() => setPage(safePage + 1)}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricChart({
  player,
  period,
  metric,
  mode,
  theme,
  onMode,
  onSelectWindow,
}) {
  const info = metricInfo(metric);
  const rows = byPlayer(DATA.windows, player).filter((w) =>
    allows(w.half, period),
  );
  const field = mode === "normalized" ? `${metric}_pct` : metric;
  const baseline = DATA.baselines.find(
    (b) => b.player_id === player && b.metric === metric,
  );
  const traces = useMemo(
    () =>
      [...new Set(rows.map((w) => w.clip_name))].map((clip) => {
        const part = rows
          .filter((r) => r.clip_name === clip)
          .sort((a, b) => a.window_start_time - b.window_start_time);
        return {
          type: "scatter",
          mode: "lines+markers",
          name: halfOf(clip) === 1 ? "Early segment" : "Late segment",
          x: part.map((r) => (r.window_start_time + r.window_end_time) / 120),
          y: part.map((r) => r[field]),
          connectgaps: false,
          line: {
            color:
              halfOf(clip) === 1 ? palette(theme).early : palette(theme).late,
            width: 2.6,
            shape: "linear",
          },
          marker: {
            size: 7,
            color:
              halfOf(clip) === 1 ? palette(theme).early : palette(theme).late,
            line: { color: theme === "light" ? "#fff" : "#142038", width: 2 },
          },
          customdata: part.map((r) => [
            r.position_coverage,
            r.valid_seconds,
            r.running_seconds,
            r.valid_cycles,
            r.window_start_time,
            r.window_end_time,
          ]),
          hovertemplate: `%{x:.1f} match min<br><b>%{y:.3f} ${mode === "normalized" ? "%" : info.unit}</b><br>Valid movement %{customdata[0]:.1%}<br>Observed %{customdata[1]:.1f}s<br>Accepted running %{customdata[2]:.1f}s · cycles %{customdata[3]}<extra>%{fullData.name}</extra>`,
        };
      }),
    [player, period, metric, mode, theme],
  );
  const shapes = [];
  const early = byPlayer(DATA.windows, player).filter((r) => r.half === 1),
    late = byPlayer(DATA.windows, player).filter((r) => r.half === 2);
  if (period === "both" && early.length && late.length)
    shapes.push({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0:
        Math.max(...early.map((r) => r.clip_start_time + r.clip_duration)) / 60,
      x1: Math.min(...late.map((r) => r.clip_start_time)) / 60,
      y0: 0,
      y1: 1,
      line: { width: 0 },
      fillcolor:
        theme === "light" ? "rgba(60,83,120,.06)" : "rgba(142,161,207,.055)",
      layer: "below",
    });
  const analysis = DATA.analysis.metrics.find(
    (r) => r.player_id === player && r.metric === metric,
  );
  (analysis?.candidates || [])
    .filter((c) => c.supported && allows(halfOf(c.clip_name), period))
    .forEach((c) =>
      shapes.push({
        type: "line",
        x0: c.boundary_time / 60,
        x1: c.boundary_time / 60,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: palette(theme).accent, dash: "dash", width: 2 },
      }),
    );
  const layout = {
    xaxis: { title: { text: "Configured match minute", font: { size: 11 } } },
    yaxis: {
      title: {
        text:
          mode === "normalized" ? "Change from own baseline (%)" : info.unit,
        font: { size: 11 },
      },
    },
    shapes,
    hovermode: "closest",
  };
  const valid = rows.some((r) => r[field] != null);
  return (
    <>
      <div className="chart-header">
        <div>
          <h3>{info.label}</h3>
          <p>{DESCRIPTIONS[metric]}</p>
        </div>
        <Segments
          label="Chart value mode"
          value={mode}
          onChange={onMode}
          options={[
            ["physical", "Measured"],
            ["normalized", "Baseline %"],
          ]}
        />
      </div>
      <div className="legend">
        <span>
          <i style={{ background: palette(theme).early }} />
          Early segment
        </span>
        <span>
          <i style={{ background: palette(theme).late }} />
          Late segment
        </span>
        <span className="quiet">2-minute nonoverlapping windows</span>
      </div>
      {valid ? (
        <Plot
          id="movement-chart"
          traces={traces}
          layout={layout}
          theme={theme}
          height={340}
          onPoint={(p) => {
            if (p.customdata)
              onSelectWindow(
                rows.find(
                  (r) =>
                    r.window_start_time === p.customdata[4] &&
                    r.window_end_time === p.customdata[5],
                ),
              );
          }}
        />
      ) : (
        <Empty
          title={
            mode === "normalized"
              ? "Own-player baseline unavailable"
              : "No qualified window estimates"
          }
          text={
            mode === "normalized"
              ? `Baseline requires three eligible windows and adequate observed exposure. This metric has ${baseline?.eligible_windows ?? 0} eligible baseline windows and ${fmt(baseline?.observed_seconds, 1)} observed seconds.`
              : "This selection does not meet the required observation exposure. Individual accepted gait cycles remain available in the gait explorer where measured."
          }
          action={
            mode === "normalized" && (
              <button
                className="button subtle"
                onClick={() => onMode("physical")}
              >
                View measured values <Icon name="arrow" size={16} />
              </button>
            )
          }
        />
      )}
      <div className="chart-foot">
        <Icon name="info" size={14} />
        <span>
          Missing observations remain missing. Shaded time was not processed.
          Select a point to inspect its exposure.
        </span>
      </div>
    </>
  );
}

function PairCards({ player, period }) {
  return (
    <div className="kpi-grid">
      {MOVEMENT.map((info) => {
        const pair = DATA.pairs.find(
          (p) => p.player_id === player && p.metric === info.id,
        );
        const value = period === "early" ? pair?.early_mean : pair?.late_mean;
        return (
          <Glass className="kpi" key={info.id}>
            <div className="kpi-top">
              <span>{info.label}</span>
              <Icon
                name={
                  info.id === "high_intensity_distance_per_minute"
                    ? "runner"
                    : "activity"
                }
                size={16}
              />
            </div>
            <div className="kpi-number">
              {fmt(value)}
              <small>{info.unit}</small>
            </div>
            <div className="kpi-bottom">
              {pair?.paired_eligible ? (
                <span className="change">
                  {pair.change_pct > 0 ? "+" : ""}
                  {fmt(pair.change_pct, 1)}% <span>late vs early</span>
                </span>
              ) : (
                <span className="quiet">
                  {value == null
                    ? "Period does not qualify"
                    : "Qualified period mean"}
                </span>
              )}
              <span className="mini-label">
                {period === "early" ? "EARLY" : "LATE"}
              </span>
            </div>
          </Glass>
        );
      })}
    </div>
  );
}

function CoveragePanel({ player, theme, period }) {
  const rows = byPlayer(DATA.coverage, player).filter((r) =>
    allows(r.half, period),
  );
  const traces = [
    {
      type: "bar",
      x: rows.map((r) => (r.half === 1 ? "Early" : "Late")),
      y: rows.map((r) => (r.valid_movement_seconds / 1500) * 100),
      marker: {
        color: rows.map((r) =>
          r.half === 1 ? palette(theme).early : palette(theme).late,
        ),
        cornerradius: 6,
      },
      width: 0.4,
      customdata: rows.map((r) => [
        r.valid_movement_seconds,
        r.movement_eligible_windows,
        r.nonoverlap_windows,
      ]),
      hovertemplate:
        "%{x}<br>Valid observed time %{y:.1f}%<br>%{customdata[0]:.1f}s<br>%{customdata[1]}/%{customdata[2]} eligible windows<extra></extra>",
    },
  ];
  // Duration comes from the actual configured clip; do not assume a 25-minute new input.
  traces[0].y = rows.map(
    (r) =>
      (r.valid_movement_seconds /
        (DATA.windows.find((w) => w.clip_name === r.clip_name)?.clip_duration ||
          1500)) *
      100,
  );
  return (
    <Glass className="coverage-card">
      <SectionHead
        eyebrow="OBSERVATION QUALITY"
        title="Coverage matters"
        text="Valid movement time in each selected segment."
      />
      <Plot
        id="coverage-chart"
        traces={traces}
        layout={{
          height: 215,
          margin: { l: 40, r: 15, t: 12, b: 35 },
          yaxis: { range: [0, 100], ticksuffix: "%", dtick: 25 },
          shapes: [
            {
              type: "line",
              xref: "paper",
              x0: 0,
              x1: 1,
              y0: 70,
              y1: 70,
              line: { color: "#dcaf74", width: 1, dash: "dot" },
            },
          ],
        }}
        theme={theme}
        height={215}
      />
      <div className="coverage-key">
        <span>
          <i />
          70% period coverage gate
        </span>
        <span>+ at least 4 eligible windows</span>
      </div>
      {rows.map((r) => (
        <div className="coverage-row" key={r.clip_name}>
          <div>
            <strong>{r.half === 1 ? "Early" : "Late"} segment</strong>
            <small>{fmt(r.valid_movement_seconds / 60, 1)} valid minutes</small>
          </div>
          <Pill tone={r.movement_eligible_windows >= 4 ? "mint" : "amber"}>
            {r.movement_eligible_windows}/{r.nonoverlap_windows} windows
          </Pill>
        </div>
      ))}
    </Glass>
  );
}
function Interpretation({ player, metric }) {
  const status = DATA.analysis.metrics.find(
    (r) => r.player_id === player && r.metric === metric,
  );
  const pair = DATA.pairs.find(
    (r) => r.player_id === player && r.metric === metric,
  );
  const base = DATA.baselines.find(
    (r) => r.player_id === player && r.metric === metric,
  );
  return (
    <Glass className="interpretation">
      <div className="panel-icon">
        <Icon name="shield" />
      </div>
      <SectionHead eyebrow="READ THE EVIDENCE" title="Movement, with context" />
      <Pill tone={status?.status === "supported_candidate" ? "mint" : "amber"}>
        {LABELS[status?.status] || "Not available"}
      </Pill>
      <p>
        {pair?.paired_eligible
          ? `This player's qualified late-period ${metricInfo(metric).label.toLowerCase()} is ${fmt(Math.abs(pair.change_pct), 1)}% ${pair.change_pct < 0 ? "lower" : "higher"} than the early period.`
          : "A paired early/late estimate requires qualified observations in both periods."}
      </p>
      <div className="fact-row">
        <span>Early baseline</span>
        <strong>
          {base?.baseline_status === "available"
            ? `${fmt(base.baseline)} ${metricInfo(metric).unit}`
            : "Unavailable"}
        </strong>
      </div>
      <div className="fact-row">
        <span>Physical windows</span>
        <strong>{status?.qualified_physical_windows ?? 0}</strong>
      </div>
      <div className="fact-row">
        <span>Supported onset</span>
        <strong>
          {status?.candidates?.some((c) => c.supported)
            ? "Candidate detected"
            : "None established"}
        </strong>
      </div>
      <div className="soft-note">
        Observed changes can also reflect tactics, opportunities and pacing.
        This study does not diagnose physiological fatigue.
      </div>
    </Glass>
  );
}

function PairComparison({ player, theme, metric }) {
  const info = metricInfo(metric),
    rows = DATA.pairs.filter((r) => r.metric === metric);
  const traces = ["early", "late"].map((p, i) => ({
    type: "bar",
    name: p === "early" ? "Early" : "Late",
    x: rows.map(
      (r) =>
        `${r.player_id.startsWith("A_") ? "Blue" : "White"} #${jersey(r.player_id)}`,
    ),
    y: rows.map((r) => r[`${p}_mean`]),
    marker: {
      color: i ? palette(theme).late : palette(theme).early,
      cornerradius: 5,
    },
    customdata: rows.map((r) => r.player_id),
    hovertemplate: `%{customdata}<br>%{y:.3f} ${info.unit}<extra>%{fullData.name}</extra>`,
  }));
  return (
    <Glass className="comparison-card">
      <SectionHead
        eyebrow="EARLY / LATE"
        title="Qualified segment comparison"
        text="Period means weighted by valid observed seconds."
      />
      <div className="legend">
        <span>
          <i style={{ background: palette(theme).early }} />
          Early
        </span>
        <span>
          <i style={{ background: palette(theme).late }} />
          Late
        </span>
      </div>
      <Plot
        id="comparison-chart"
        traces={traces}
        theme={theme}
        height={250}
        layout={{
          barmode: "group",
          xaxis: { type: "category" },
          yaxis: { title: { text: info.unit, font: { size: 11 } } },
          margin: { l: 52, r: 15, t: 15, b: 45 },
        }}
      />
      <div className="chart-foot">
        Missing bars indicate a period that does not qualify. Each player,
        rather than each window, is the paired statistical unit.
      </div>
    </Glass>
  );
}
function WindowInspector({ window, onClose, metric }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section
        className="glass modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="window-title"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === "Tab") {
            e.preventDefault();
            e.currentTarget.querySelector("button")?.focus();
          }
        }}
      >
        <div className="modal-head">
          <span className="eyebrow">WINDOW INSPECTOR</span>
          <button
            autoFocus
            className="icon-button"
            onClick={onClose}
            aria-label="Close window inspector"
          >
            <Icon name="close" />
          </button>
        </div>
        <h2 id="window-title">
          {fmt(window.window_start_time / 60, 0)}–
          {fmt(window.window_end_time / 60, 0)} match min
        </h2>
        <p>
          {window.player_id} · {window.half === 1 ? "early" : "late"} segment
        </p>
        <div className="inspector-grid">
          {[
            [
              "Measured value",
              `${fmt(window[metric], 3)} ${metricInfo(metric).unit}`,
            ],
            ["Valid movement time", `${fmt(window.valid_seconds, 1)}s`],
            ["Movement coverage", pct(window.position_coverage)],
            ["Accepted running", `${fmt(window.running_seconds, 1)}s`],
            ["Accepted cycles", window.valid_cycles],
            ["Movement eligible", window.movement_eligible ? "Yes" : "No"],
            ["Gait eligible", window.gait_eligible ? "Yes" : "No"],
            ["Clip", window.clip_name],
          ].map(([k, v]) => (
            <div key={k}>
              <span>{k}</span>
              <strong>{v}</strong>
            </div>
          ))}
        </div>
        <p className="soft-note">
          Window estimates are available only when the metric's exposure and
          confidence requirements pass.
        </p>
      </section>
    </div>
  );
}

function MovementPage({
  player,
  period,
  theme,
  metric,
  setMetric,
  mode,
  setMode,
  onSelectWindow,
  overview,
}) {
  const rows = byPlayer(DATA.windows, player).filter((r) =>
    allows(r.half, period),
  );
  const stats = byPlayer(DATA.analysis.metrics, player);
  return (
    <>
      <PairCards player={player} period={period} />
      <div className="analysis-grid">
        <Glass className="main-chart">
          <div className="panel-toolbar">
            <span className="eyebrow">MOVEMENT TIMELINE</span>
            <label className="select-label">
              <span className="sr-only">Metric</span>
              <select
                id="metric"
                aria-label="Movement metric"
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
              >
                {DATA.metrics.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <MetricChart
            {...{ player, period, metric, mode, theme, onSelectWindow }}
            onMode={setMode}
          />
        </Glass>
        <Interpretation player={player} metric={metric} />
      </div>
      <div className="lower-grid">
        <PairComparison {...{ player, theme, metric }} />
        <CoveragePanel {...{ player, theme, period }} />
      </div>
      {!overview && (
        <>
          <Glass className="table-card">
            <SectionHead
              eyebrow="PER-METRIC QUALIFICATION"
              title="Analysis status"
              text="Physical exposure and normalized analysis are reported separately."
            />
            <Table
              rows={stats}
              filename={`${player}-analysis-status.csv`}
              columns={[
                {
                  key: "metric",
                  label: "Metric",
                  render: (v) => metricInfo(v).label,
                },
                {
                  key: "status",
                  label: "Conclusion",
                  render: (v) => (
                    <Pill
                      tone={v === "no_supported_change" ? "neutral" : "amber"}
                    >
                      {LABELS[v]}
                    </Pill>
                  ),
                },
                {
                  key: "qualified_physical_windows",
                  label: "Measured windows",
                },
                { key: "eligible_windows", label: "Normalized windows" },
                {
                  key: "longest_continuous_windows",
                  label: "Longest eligible sequence",
                },
              ]}
            />
          </Glass>
          <Glass className="table-card">
            <SectionHead
              eyebrow="INTERMEDIATE OUTPUTS"
              title="Window measurements"
              text="Full exposure and eligibility accompany every estimate."
            />
            <Table
              rows={rows}
              filename={`${player}-${period}-window-details.csv`}
              columns={[
                {
                  key: "window_start_time",
                  label: "Match minute",
                  render: (v) => fmt(v / 60, 1),
                },
                {
                  key: "clip_name",
                  label: "Segment",
                  render: (v) => (halfOf(v) === 1 ? "Early" : "Late"),
                },
                {
                  key: metric,
                  label: metricInfo(metric).label,
                  render: (v) => fmt(v, 3),
                },
                {
                  key: "position_coverage",
                  label: "Coverage",
                  render: (v) => pct(v),
                },
                {
                  key: "running_seconds",
                  label: "Running s",
                  render: (v) => fmt(v, 1),
                },
                { key: "valid_cycles", label: "Cycles" },
                {
                  key: "movement_eligible",
                  label: "Movement",
                  render: (v) => (v ? "Eligible" : "Insufficient"),
                },
                {
                  key: "gait_eligible",
                  label: "Gait",
                  render: (v) => (v ? "Eligible" : "Insufficient"),
                },
              ]}
            />
          </Glass>
        </>
      )}
    </>
  );
}

function GaitPage({ player, period, theme }) {
  const [metric, setMetric] = useState("stride_frequency"),
    [leg, setLeg] = useState("both"),
    [minSpeed, setMinSpeed] = useState(2),
    [maxSpeed, setMaxSpeed] = useState(9),
    [axis, setAxis] = useState("time");
  const rows = byPlayer(DATA.cycles, player).filter(
    (r) =>
      allows(halfOf(r.clip_name), period) &&
      (leg === "both" || r.leg === leg) &&
      r.avg_speed >= minSpeed &&
      r.avg_speed <= maxSpeed,
  );
  const info = metricInfo(metric);
  const traces = ["first", "second"].map((half, i) => {
    const part = rows.filter((r) => r.clip_name.endsWith(half));
    return {
      type: "scatter",
      mode: "markers",
      name: i ? "Late cycles" : "Early cycles",
      x: part.map((r) =>
        axis === "speed" ? r.avg_speed : (r.start_time + r.end_time) / 120,
      ),
      y: part.map((r) => r[metric]),
      marker: {
        color: i ? palette(theme).late : palette(theme).early,
        size: 8,
        opacity: 0.8,
        symbol: part.map((r) => (r.leg === "left" ? "diamond" : "circle")),
        line: { width: 1, color: theme === "light" ? "#fff" : "#1c2945" },
      },
      customdata: part.map((r) => [
        r.leg,
        r.avg_speed,
        r.body_scale_source_pixels,
        r.start_time,
        r.end_time,
      ]),
      hovertemplate: `<b>%{y:.3f} ${info.unit}</b><br>%{customdata[0]} leg<br>Speed %{customdata[1]:.2f} m/s<br>Body scale %{customdata[2]:.0f}px<extra>%{fullData.name}</extra>`,
    };
  });
  return (
    <>
      <div className="page-intro">
        <div>
          <span className="eyebrow">PROJECTED RUNNING MECHANICS</span>
          <h2>Look closer at each stride.</h2>
          <p>
            Native-rate accepted cycles, compared at similar running speeds.
          </p>
        </div>
        <Pill tone="mint">{rows.length} accepted cycles in selection</Pill>
      </div>
      <div className="cycle-kpis">
        {[
          ["stride_frequency", "Median cadence"],
          ["stride_length", "Median stride length"],
          ["knee_angle_range", "Median knee excursion"],
        ].map(([m, label]) => (
          <Glass className="kpi" key={m}>
            <div className="kpi-top">
              {label}
              <Icon name="runner" size={16} />
            </div>
            <div className="kpi-number">
              {fmt(median(rows.map((r) => r[m])))}
              <small>{metricInfo(m).unit}</small>
            </div>
            <span className="quiet">Descriptive accepted-cycle median</span>
          </Glass>
        ))}
      </div>
      <Glass className="gait-chart">
        <div className="panel-toolbar">
          <label>
            <span className="sr-only">Gait metric</span>
            <select
              aria-label="Gait metric"
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
            >
              {GAIT.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </label>
          <Segments
            label="Gait chart horizontal axis"
            value={axis}
            onChange={setAxis}
            options={[
              ["time", "Match time"],
              ["speed", "Running speed"],
            ]}
          />
        </div>
        <div className="gait-filters">
          <div className="range-field">
            <label htmlFor="minimum-speed">
              Min. running speed <strong>{fmt(minSpeed, 1)} m/s</strong>
            </label>
            <input
              id="minimum-speed"
              type="range"
              min="2"
              max="9"
              step="0.1"
              value={minSpeed}
              onChange={(e) => setMinSpeed(Math.min(+e.target.value, maxSpeed))}
            />
          </div>
          <div className="range-field">
            <label htmlFor="maximum-speed">
              Max. running speed <strong>{fmt(maxSpeed, 1)} m/s</strong>
            </label>
            <input
              id="maximum-speed"
              type="range"
              min="2"
              max="9"
              step="0.1"
              value={maxSpeed}
              onChange={(e) => setMaxSpeed(Math.max(+e.target.value, minSpeed))}
            />
          </div>
          <Segments
            label="Cycle leg filter"
            value={leg}
            onChange={setLeg}
            options={[
              ["both", "Both legs"],
              ["left", "Left"],
              ["right", "Right"],
            ]}
          />
        </div>
        {rows.length ? (
          <Plot
            id="gait-chart"
            traces={traces}
            theme={theme}
            height={370}
            layout={{
              xaxis: {
                title: {
                  text:
                    axis === "speed"
                      ? "Average running speed (m/s)"
                      : "Configured match minute",
                },
              },
              yaxis: { title: { text: info.unit } },
            }}
          />
        ) : (
          <Empty
            title="No accepted cycles in this selection"
            text="Try a different player, period or speed range. Missing cycles are not interpolated."
          />
        )}
        <div className="chart-foot">
          {DESCRIPTIONS[metric]} · Individual cycles are descriptive;
          window-level gait decline requires greater accepted exposure.
        </div>
      </Glass>
      <Glass className="table-card">
        <SectionHead
          eyebrow="CYCLE RECORDS"
          title="Every accepted observation"
          text="Left/right leg, running speed and source body size remain inspectable."
        />
        <Table
          rows={rows}
          filename={`${player}-gait-cycles.csv`}
          columns={[
            {
              key: "start_time",
              label: "Match min",
              render: (v) => fmt(v / 60, 2),
            },
            { key: "leg", label: "Leg", render: human },
            { key: "avg_speed", label: "Speed m/s", render: (v) => fmt(v) },
            {
              key: "stride_frequency",
              label: "Cycles/s",
              render: (v) => fmt(v),
            },
            { key: "stride_length", label: "Stride m", render: (v) => fmt(v) },
            {
              key: "knee_angle_range",
              label: "Knee °",
              render: (v) => fmt(v, 1),
            },
            {
              key: "vertical_oscillation",
              label: "Vertical proxy",
              render: (v) => fmt(v, 3),
            },
            {
              key: "body_scale_source_pixels",
              label: "Body px",
              render: (v) => fmt(v, 0),
            },
          ]}
        />
      </Glass>
    </>
  );
}

function PitchPage({ player, period, theme }) {
  const selected = DATA.trajectories.filter(
    (t) => t.player_id === player && allows(halfOf(t.clip_name), period),
  );
  const all = selected
    .flatMap((s) => s.points)
    .filter((p) => p.t != null)
    .sort((a, b) => a.t - b.t);
  const minimum = all[0]?.t || 0,
    maximum = all.at(-1)?.t || 90;
  const [cut, setCut] = useState(maximum),
    [playing, setPlaying] = useState(false),
    [mode, setMode] = useState("path");
  useEffect(() => {
    setCut(maximum);
    setPlaying(false);
  }, [player, period]);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () =>
        setCut((value) => {
          const next = all.find((p) => p.t > value + 0.05);
          if (!next) {
            setPlaying(false);
            return maximum;
          }
          return next.t;
        }),
      120,
    );
    return () => clearInterval(timer);
  }, [playing, player, period, maximum]);
  const points = all.filter((p) => p.t <= cut),
    current = points.at(-1),
    atCurrent = current && Math.abs(current.t - cut) < 0.035;
  const traces =
    mode === "heat"
      ? [
          {
            type: "histogram2d",
            x: points.map((p) => p.x),
            y: points.map((p) => p.y),
            xbins: { start: 0, end: 105, size: 5 },
            ybins: { start: 0, end: 68, size: 4 },
            colorscale: [
              [0, "rgba(70,150,130,0)"],
              [0.25, "rgba(87,201,250,.28)"],
              [0.6, "rgba(114,220,186,.7)"],
              [1, "#d7f6a8"],
            ],
            showscale: false,
            hovertemplate:
              "Field x %{x}m<br>Field y %{y}m<br>Sampled positions %{z}<extra></extra>",
          },
        ]
      : selected.map((s) => {
          const p = s.points.filter((p) => p.t == null || p.t <= cut);
          return {
            type: "scatter",
            mode: "lines",
            x: p.map((p) => p.x),
            y: p.map((p) => p.y),
            line: {
              color:
                halfOf(s.clip_name) === 1
                  ? palette(theme).early
                  : palette(theme).late,
              width: 1.6,
            },
            opacity: 0.8,
            connectgaps: false,
            name: halfOf(s.clip_name) === 1 ? "Early path" : "Late path",
            customdata: p.map((p) => [p.t, p.speed]),
            hovertemplate:
              "x %{x:.1f}m · y %{y:.1f}m<br>Minute %{customdata[0]:.2f}<br>Speed %{customdata[1]:.2f} m/s<extra>%{fullData.name}</extra>",
          };
        });
  if (atCurrent)
    traces.push({
      type: "scatter",
      mode: "markers",
      x: [current.x],
      y: [current.y],
      marker: {
        color: "#ecfcae",
        size: 14,
        line: { color: theme === "light" ? "#316464" : "#163a33", width: 3 },
      },
      hovertemplate: `#${jersey(player)} · ${fmt(current.speed)} m/s<extra></extra>`,
    });
  const line = {
    color: theme === "light" ? "rgba(30,105,76,.45)" : "rgba(147,210,183,.38)",
    width: 1.4,
  };
  const shapes = [
    { type: "rect", x0: 0, y0: 0, x1: 105, y1: 68, line },
    { type: "line", x0: 52.5, y0: 0, x1: 52.5, y1: 68, line },
    {
      type: "circle",
      x0: 52.5 - 9.15,
      x1: 52.5 + 9.15,
      y0: 34 - 9.15,
      y1: 34 + 9.15,
      line,
    },
  ];
  for (const [start, end] of [
    [0, 16.5],
    [88.5, 105],
  ])
    shapes.push({
      type: "rect",
      x0: start,
      x1: end,
      y0: 13.84,
      y1: 54.16,
      line,
    });
  for (const [start, end] of [
    [0, 5.5],
    [99.5, 105],
  ])
    shapes.push({
      type: "rect",
      x0: start,
      x1: end,
      y0: 24.84,
      y1: 43.16,
      line,
    });
  return (
    <>
      <div className="page-intro">
        <div>
          <span className="eyebrow">MEASURED GROUND POSITIONS</span>
          <h2>The match, mapped.</h2>
          <p>Explore observed trajectories on a 105 × 68 meter pitch.</p>
        </div>
        <Segments
          label="Pitch display"
          value={mode}
          onChange={setMode}
          options={[
            ["path", "Trajectory"],
            ["heat", "Occupancy"],
          ]}
        />
      </div>
      <Glass className="pitch-card">
        <div className="pitch-top">
          <div className="legend">
            <span>
              <i style={{ background: palette(theme).early }} />
              Early segment
            </span>
            <span>
              <i style={{ background: palette(theme).late }} />
              Late segment
            </span>
          </div>
          <Pill tone={atCurrent ? "mint" : "amber"}>
            {atCurrent
              ? `${fmt(current.speed)} m/s at cursor`
              : "No observation at cursor"}
          </Pill>
        </div>
        {all.length ? (
          <Plot
            id="pitch-chart"
            traces={traces}
            theme={theme}
            height={490}
            layout={{
              shapes,
              plot_bgcolor:
                theme === "light"
                  ? "rgba(62,146,105,.08)"
                  : "rgba(45,119,81,.085)",
              margin: { l: 40, r: 20, t: 15, b: 40 },
              xaxis: {
                range: [-2, 107],
                title: { text: "Field x (m)" },
                showgrid: false,
                zeroline: false,
              },
              yaxis: {
                range: [-2, 70],
                scaleanchor: "x",
                scaleratio: 1,
                title: { text: "Field y (m)" },
                showgrid: false,
                zeroline: false,
              },
            }}
          />
        ) : (
          <Empty
            title="Trajectories unavailable"
            text="Regenerate this dashboard after frame-level position outputs are available."
          />
        )}
        <div className="playback">
          <button
            className="icon-button"
            aria-label={
              playing ? "Pause trajectory playback" : "Play trajectory playback"
            }
            onClick={() => {
              if (cut >= maximum) setCut(minimum);
              setPlaying(!playing);
            }}
            disabled={!all.length}
          >
            <Icon name={playing ? "pause" : "play"} />
          </button>
          <label className="sr-only" htmlFor="pitch-time">
            Pitch time
          </label>
          <input
            id="pitch-time"
            type="range"
            min={minimum}
            max={maximum}
            step="0.01667"
            value={cut}
            onChange={(e) => {
              setPlaying(false);
              setCut(+e.target.value);
            }}
          />
          <span>
            {fmt(cut, 1)} <small>match min</small>
          </span>
        </div>
        <div className="chart-foot">
          One-second display sampling. Paths break at observation losses and
          identity transitions. Playback advances through observed samples;
          occupancy counts sampled positions.
        </div>
      </Glass>
      <div className="two-column">
        <CoveragePanel {...{ player, period, theme }} />
        <Glass className="map-details">
          <SectionHead
            eyebrow="FIELD GEOMETRY"
            title="Coordinates with provenance"
          />
          <div className="fact-row">
            <span>Ground anchor</span>
            <strong>Detector box bottom</strong>
          </div>
          <div className="fact-row">
            <span>Field convention</span>
            <strong>Corner origin · meters</strong>
          </div>
          <div className="fact-row">
            <span>View coordinates</span>
            <strong>105 m × 68 m</strong>
          </div>
          <div className="fact-row">
            <span>Calibration</span>
            <strong>Validated rational warp</strong>
          </div>
          <div className="fact-row">
            <span>Shown positions</span>
            <strong>{points.length.toLocaleString()}</strong>
          </div>
          <p className="soft-note">
            The world position comes from independently detected ground anchors.
            Reference annotations assist identity and never replace the measured
            coordinates.
          </p>
        </Glass>
      </div>
    </>
  );
}

function EvaluationPage({ theme }) {
  const training = DATA.training,
    test = training?.chosen_test,
    broadcast = DATA.broadcast_clips;
  const frameTotal = DATA.broadcast_frames.reduce(
      (s, r) => s + r.labeled_players,
      0,
    ),
    matched = DATA.broadcast_frames.reduce((s, r) => s + r.matched_players, 0);
  const posePassed = DATA.broadcast_poses.filter(
    (r) => r.six_joint_confident,
  ).length;
  const detection = test
    ? [
        ["Precision", test.precision],
        ["Recall", test.recall],
        ["mAP @ .50", test.map50],
        ["mAP @ .50:.95", test.map50_95],
      ]
    : [];
  return (
    <>
      <div className="page-intro">
        <div>
          <span className="eyebrow">SEPARATE LAYERS OF EVIDENCE</span>
          <h2>Know what each result measures.</h2>
          <p>
            Detection, tracking, calibration and pose are evaluated
            independently.
          </p>
        </div>
        <Pill tone="mint">
          <Icon name="check" size={13} /> Recorded evaluations
        </Pill>
      </div>
      <div className="kpi-grid">
        {detection.map(([label, value]) => (
          <Glass className="kpi" key={label}>
            <div className="kpi-top">
              {label}
              <Icon name="chart" size={16} />
            </div>
            <div className="kpi-number">{pct(value, 2)}</div>
            <span className="quiet">100 held-out detector images</span>
          </Glass>
        ))}
      </div>
      <div className="two-column">
        <Glass className="evaluation-chart">
          <SectionHead
            eyebrow="BROADCAST STRESS AUDIT"
            title="Detection across five clips"
            text="Known-player recall at IoU ≥0.50 on ~2-fps samples."
          />
          {broadcast.length ? (
            <Plot
              id="broadcast-chart"
              traces={[
                {
                  type: "bar",
                  x: broadcast.map((r) => r.clip),
                  y: broadcast.map((r) => r.player_recall_iou50 * 100),
                  marker: {
                    color: [
                      "#55c9fa",
                      "#73d6d2",
                      "#aa9bff",
                      "#d2b3f5",
                      "#76e5b0",
                    ],
                    cornerradius: 5,
                  },
                  hovertemplate: "%{x}<br>Recall %{y:.2f}%<extra></extra>",
                },
              ]}
              theme={theme}
              height={285}
              layout={{
                yaxis: { range: [0, 100], ticksuffix: "%" },
                margin: { l: 43, r: 15, t: 15, b: 40 },
              }}
            />
          ) : (
            <Empty
              title="Broadcast audit not available"
              text="This separate short-clip evaluation is optional for a new study."
            />
          )}
          <div className="fact-row">
            <span>Pooled labeled-player recall</span>
            <strong>{frameTotal ? pct(matched / frameTotal, 2) : "—"}</strong>
          </div>
          <div className="fact-row">
            <span>Labeled / matched boxes</span>
            <strong>
              {frameTotal.toLocaleString()} / {matched.toLocaleString()}
            </strong>
          </div>
          <div className="chart-foot">
            Different from the held-out image evaluation. This is not all-person
            precision or fatigue accuracy.
          </div>
        </Glass>
        <Glass className="evaluation-chart">
          <SectionHead
            eyebrow="GROUND MAPPING"
            title="Field-reference validation"
            text="Withheld pitch reference points, not player-position ground truth."
          />
          <Plot
            id="calibration-chart"
            traces={[
              {
                type: "bar",
                name: "Median error",
                x: Object.keys(DATA.calibration),
                y: Object.values(DATA.calibration).map(
                  (v) => v.validation.median_error_m,
                ),
                marker: { color: palette(theme).early, cornerradius: 4 },
              },
              {
                type: "bar",
                name: "95th percentile",
                x: Object.keys(DATA.calibration),
                y: Object.values(DATA.calibration).map(
                  (v) => v.validation.p95_error_m,
                ),
                marker: { color: palette(theme).late, cornerradius: 4 },
              },
            ]}
            theme={theme}
            height={285}
            layout={{
              barmode: "group",
              yaxis: { title: { text: "Field-point error (m)" } },
              margin: { l: 50, r: 15, t: 15, b: 40 },
            }}
          />
          <div className="legend">
            <span>
              <i style={{ background: palette(theme).early }} />
              Median
            </span>
            <span>
              <i style={{ background: palette(theme).late }} />
              95th percentile
            </span>
          </div>
          <div className="fact-row">
            <span>Selected mapping</span>
            <strong>Quadratic rational</strong>
          </div>
          <div className="fact-row">
            <span>Tested field points / camera</span>
            <strong>13</strong>
          </div>
        </Glass>
      </div>
      <div className="three-column">
        <Glass className="evidence-card">
          <Icon name="runner" />
          <span className="eyebrow">POSE AVAILABILITY</span>
          <h2>
            {DATA.broadcast_poses.length
              ? `${posePassed}/${DATA.broadcast_poses.length}`
              : "—"}
          </h2>
          <p>Crops passing all six hip/knee/ankle confidence gates.</p>
          <small>
            Two tallest matched players every sixth sampled frame. Confidence
            availability is not anatomical accuracy.
          </small>
        </Glass>
        <Glass className="evidence-card">
          <Icon name="activity" />
          <span className="eyebrow">NO-HINT TRACKING AUDIT</span>
          <h2>{pct(DATA.tracking?.known_player_box_recall, 2)}</h2>
          <p>Known-player box recall at IoU ≥0.50 over 60 seconds.</p>
          <small>
            {DATA.tracking?.consecutive_observed_identity_switches ?? "—"}{" "}
            consecutive observed ID switches. No annotation hints. This is not
            IDF1/HOTA.
          </small>
        </Glass>
        <Glass className="evidence-card">
          <Icon name="shield" />
          <span className="eyebrow">SOFTWARE CONTROLS</span>
          <h2>
            {DATA.controls?.controls.filter((c) => c.passed).length ?? 0}/3
          </h2>
          <p>Known-step, stationary-noise and constant-series controls.</p>
          <small>
            Synthetic algorithm controls; they do not measure clinical fatigue
            performance.
          </small>
        </Glass>
      </div>
      <Glass className="table-card">
        <SectionHead
          eyebrow="MODEL EXPERIMENT"
          title="One bounded detector adaptation"
          text="Validation retained the original pretrained YOLOv8n."
        />
        <div className="experiment-summary">
          <div>
            <span>Training / validation / test</span>
            <strong>387 / 100 / 100 images</strong>
          </div>
          <div>
            <span>Completed epochs</span>
            <strong>{training?.epochs_completed ?? "—"}</strong>
          </div>
          <div>
            <span>Experiment + evaluation time</span>
            <strong>{fmt(training?.elapsed_seconds, 0)} s</strong>
          </div>
          <div>
            <span>Validation mAP .50:.95</span>
            <strong>
              {pct(training?.pretrained_validation.map50_95, 2)} base ·{" "}
              {pct(training?.adapted_validation.map50_95, 2)} adapted
            </strong>
          </div>
        </div>
        <p className="soft-note">
          Labels include visually reviewed model-assisted additions. Detection
          scores are an evaluation of this subset; source annotation quality can
          affect AP.
        </p>
      </Glass>
    </>
  );
}

function MethodsPage({ player, theme }) {
  const [tab, setTab] = useState("method");
  const trends = byPlayer(DATA.trends, player),
    sensitivity = byPlayer(DATA.sensitivity, player),
    baselines = byPlayer(DATA.baselines, player);
  return (
    <>
      <div className="page-intro">
        <div>
          <span className="eyebrow">COMPLETE STUDY CONTEXT</span>
          <h2>The details behind the dashboard.</h2>
          <p>Definitions, thresholds, datasets and statistical evidence.</p>
        </div>
        <Segments
          label="Methods detail"
          value={tab}
          onChange={setTab}
          options={[
            ["method", "Methodology"],
            ["evidence", "Analysis tables"],
          ]}
        />
      </div>
      {tab === "method" ? (
        <>
          <div className="two-column">
            <Glass className="method-card">
              <SectionHead
                eyebrow="DATASET & COHORT"
                title="A scoped longitudinal study"
              />
              <div className="fact-row">
                <span>Main footage</span>
                <strong>SoccerTrack v2</strong>
              </div>
              <div className="fact-row">
                <span>Analysis scope</span>
                <strong>
                  {DATA.scope.players} players · {DATA.scope.matches} matches
                </strong>
              </div>
              <div className="fact-row">
                <span>Distinct analyzed footage</span>
                <strong>{DATA.scope.distinct_minutes} min</strong>
              </div>
              <div className="fact-row">
                <span>Native video</span>
                <strong>4096 × 1080 · 25 fps</strong>
              </div>
              <div className="fact-row">
                <span>Identity</span>
                <strong>Annotation-assisted study</strong>
              </div>
              <div className="fact-row">
                <span>Early / late intervals</span>
                <strong>5–30 / 60–85 match min</strong>
              </div>
              <p>
                Source inventory: ~25.29 GB including 184 minutes of full-half
                video, labels and camera assets. Only the selected 100 minutes
                form the longitudinal study. The supplementary broadcast audit
                contains 300 sampled images (~175 MB extracted).
              </p>
              <p className="soft-note">
                Second-half match time uses an approximate 45-minute offset and
                omits first-half stoppage time. Unprocessed intervals are not
                interpolated.
              </p>
            </Glass>
            <Glass className="method-card">
              <SectionHead
                eyebrow="MODELS & EXECUTION"
                title="Pretrained vision, measured features"
              />
              <div className="pipeline-list">
                {[
                  ["YOLOv8n", "Person detection at 12.5 fps"],
                  [
                    "ByteTrack",
                    "Temporary track IDs; reviewed identity association",
                  ],
                  [
                    "YOLOv8m-Pose",
                    "Native-rate joints on suitable running crops",
                  ],
                  [
                    "Rational field warp",
                    "Independent detected ground anchors in meters",
                  ],
                  [
                    "Regression + PELT",
                    "Qualified trends and change-point search",
                  ],
                ].map(([name, detail], i) => (
                  <div key={name}>
                    <span className="step-number">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <strong>{name}</strong>
                      <small>{detail}</small>
                    </div>
                  </div>
                ))}
              </div>
              <p className="soft-note">
                The bounded detector experiment ran on an 8-GB RTX 4060 Laptop
                GPU. No separate physiological-fatigue classifier was trained.
              </p>
            </Glass>
          </div>
          <Glass className="method-card">
            <SectionHead
              eyebrow="MEASUREMENT RULES"
              title="Exposure before interpretation"
            />
            <div className="threshold-grid">
              {[
                [
                  "Movement windows",
                  "120 s; ≥70% valid movement exposure. Display windows may overlap; inference windows do not.",
                ],
                [
                  "Early baseline",
                  "First 600 elapsed seconds of the selected early clip; ≥3 eligible windows and ≥420 movement seconds. Near-zero baselines are unavailable.",
                ],
                [
                  "Gait windows",
                  "≥20 accepted running seconds and ≥10 same-leg cycles, with sufficient finite joint observations. Individual cycles remain descriptive.",
                ],
                [
                  "Running-bout qualification",
                  "≥4 seconds, 2–9 m/s, body height ≥40 source pixels, ≤30° heading spread over 2 seconds.",
                ],
                [
                  "Pose and continuity",
                  "Joint confidence ≥0.5. ≥80% observed gait signal; no pose hole >0.16 s. Ground smoothing never crosses gaps or identity changes.",
                ],
                [
                  "Change-point support",
                  "≥8 contiguous windows, ≥3 pre/post, ≥10 baseline percentage-point drop and ≥2× local noise; penalty and sensitivity agreement required.",
                ],
              ].map(([title, text]) => (
                <div key={title}>
                  <h3>{title}</h3>
                  <p>{text}</p>
                </div>
              ))}
            </div>
          </Glass>
          <Glass className="method-card">
            <SectionHead
              eyebrow="STATISTICS & INTERPRETATION"
              title="Descriptive change, not a diagnosis"
            />
            <p>
              Trend regression needs six contiguous eligible windows, uses lag-1
              HAC uncertainty and Holm correction per player. Sensitivity checks
              0.6/0.8/1.2-second smoothing and 120/180/240-second windows, with
              2/3/4-log(n) penalties. Supported candidates require corroboration
              across at least two smoothing values, two window widths and half
              of assessable settings.
            </p>
            <p>
              Early/late comparisons require four eligible windows and ≥70%
              full-period movement exposure in both periods. The statistical
              unit is the player. Only one player qualifies for a movement pair,
              so the configured population paired test is not run. No
              substitution time is verified for the supplied cohort. Tactics,
              pacing, opportunities, viewpoint and selective missingness can
              affect measurements.
            </p>
            <p>
              Ground calibration cannot convert elevated hip joints into
              physical vertical meters. Cadence and vertical-motion changes do
              not have a universal fatigue direction. This dataset establishes
              movement observations; physiological fatigue and a supported onset
              are not demonstrated.
            </p>
            <a
              className="text-link"
              href="https://github.com/srutan2004/Player-Fatigue-Detection"
              target="_blank"
              rel="noreferrer"
            >
              Project repository <Icon name="external" size={15} />
            </a>
          </Glass>
        </>
      ) : (
        <>
          <Glass className="table-card">
            <SectionHead
              eyebrow="OWN-PLAYER REFERENCE"
              title="Baseline exposure"
            />
            <Table
              rows={baselines}
              filename={`${player}-baselines.csv`}
              columns={[
                {
                  key: "metric",
                  label: "Metric",
                  render: (v) => metricInfo(v).label,
                },
                {
                  key: "baseline",
                  label: "Baseline",
                  render: (v) => fmt(v, 3),
                },
                { key: "baseline_status", label: "Status", render: human },
                { key: "eligible_windows", label: "Windows" },
                {
                  key: "observed_seconds",
                  label: "Observed s",
                  render: (v) => fmt(v, 1),
                },
              ]}
            />
          </Glass>
          <Glass className="table-card">
            <SectionHead
              eyebrow="DESCRIPTIVE HAC REGRESSION"
              title="Trend uncertainty"
            />
            <Table
              rows={trends}
              filename={`${player}-trends.csv`}
              empty="No qualified baseline-normalized continuous sequence for this player."
              columns={[
                {
                  key: "metric",
                  label: "Metric",
                  render: (v) => metricInfo(v).label,
                },
                { key: "windows", label: "Windows" },
                {
                  key: "slope_pct_points_per_minute",
                  label: "pp / min",
                  render: (v) => fmt(v, 3),
                },
                { key: "ci_low", label: "CI lower", render: (v) => fmt(v, 3) },
                { key: "ci_high", label: "CI upper", render: (v) => fmt(v, 3) },
                { key: "p_value", label: "p", render: (v) => fmt(v, 4) },
                {
                  key: "holm_p_value",
                  label: "Holm p",
                  render: (v) => fmt(v, 4),
                },
              ]}
            />
          </Glass>
          <Glass className="table-card">
            <SectionHead
              eyebrow="SENSITIVITY CHECKS"
              title="Alternative analysis settings"
              text="Insufficient exposure remains unavailable under alternative settings."
            />
            <Table
              rows={sensitivity}
              filename={`${player}-sensitivity.csv`}
              columns={[
                { key: "clip_name", label: "Clip" },
                {
                  key: "metric",
                  label: "Metric",
                  render: (v) => metricInfo(v).label,
                },
                { key: "smoothing_seconds", label: "Smoothing s" },
                { key: "window_seconds", label: "Window s" },
                { key: "eligible_windows", label: "Windows" },
                {
                  key: "search_available",
                  label: "Search",
                  render: (v) => (v ? "Available" : "Unavailable"),
                },
                { key: "candidate_times", label: "Candidate seconds" },
                {
                  key: "status",
                  label: "Status",
                  render: (v) => LABELS[v] || human(v),
                },
              ]}
            />
          </Glass>
          <Glass className="table-card">
            <SectionHead
              eyebrow="CANDIDATE EVIDENCE"
              title="Boundaries and robustness"
              text="Only supported candidates appear as accepted plot boundaries."
            />
            <Table
              rows={byPlayer(DATA.analysis.metrics, player).flatMap((m) =>
                m.candidates.map((c) => ({
                  ...c,
                  metric: m.metric,
                  status: m.status,
                })),
              )}
              empty="No candidate boundaries for this player."
              filename={`${player}-candidates.csv`}
              columns={[
                {
                  key: "metric",
                  label: "Metric",
                  render: (v) => metricInfo(v).label,
                },
                {
                  key: "boundary_time",
                  label: "Match min",
                  render: (v) => fmt(v / 60, 1),
                },
                {
                  key: "drop_pct_points",
                  label: "Change pp",
                  render: (v) => fmt(v, 2),
                },
                { key: "penalty_votes", label: "Penalty votes" },
                {
                  key: "sensitivity_supporting_settings",
                  label: "Supporting settings",
                },
                {
                  key: "sensitivity_assessed_settings",
                  label: "Assessed settings",
                },
                {
                  key: "supported",
                  label: "Supported",
                  render: (v) => (v ? "Yes" : "No"),
                },
              ]}
            />
          </Glass>
          <Glass className="table-card">
            <SectionHead
              eyebrow="PLAYER-LEVEL PROXY COMPARISON"
              title="Paired-test eligibility"
            />
            <Table
              rows={DATA.paired_tests?.results || []}
              filename="paired-tests.csv"
              columns={[
                {
                  key: "metric",
                  label: "Metric",
                  render: (v) => metricInfo(v).label,
                },
                { key: "paired_players", label: "Players" },
                { key: "test", label: "Test" },
                { key: "p_value", label: "p", render: (v) => fmt(v, 4) },
                { key: "status", label: "Status", render: human },
              ]}
            />
          </Glass>
        </>
      )}
    </>
  );
}

function App() {
  const params = new URLSearchParams(location.search);
  const [player, setPlayer] = useState(
    PLAYERS.includes(params.get("player")) ? params.get("player") : PLAYERS[0],
  );
  const [view, setView] = useState(
    NAV.some((n) => n[0] === params.get("view"))
      ? params.get("view")
      : "overview",
  );
  const [period, setPeriod] = useState(
    ["both", "early", "late"].includes(params.get("period"))
      ? params.get("period")
      : "both",
  );
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("stride-theme") || "dark";
    } catch {
      return "dark";
    }
  });
  const [metric, setMetric] = useState("avg_speed"),
    [mode, setMode] = useState("physical"),
    [windowSelection, setWindowSelection] = useState(null);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("stride-theme", theme);
    } catch {}
  }, [theme]);
  useEffect(() => {
    const query = new URLSearchParams({ player, view, period });
    try {
      history.replaceState(null, "", `${location.pathname}?${query}`);
    } catch {}
  }, [player, view, period]);
  useEffect(() => {
    if (!windowSelection) return;
    const escape = (e) => {
      if (e.key === "Escape") setWindowSelection(null);
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [windowSelection]);
  const rows = byPlayer(DATA.coverage, player),
    valid = rows.reduce((s, r) => s + r.valid_movement_seconds, 0),
    duration = rows.reduce(
      (s, r) =>
        s +
        (DATA.windows.find((w) => w.clip_name === r.clip_name)?.clip_duration ||
          1500),
      0,
    );
  const selectedName = player.startsWith("A_")
    ? "Match A · blue kit"
    : "Match B · white kit";
  const props = {
    player,
    period,
    theme,
    metric,
    setMetric,
    mode,
    setMode,
    onSelectWindow: setWindowSelection,
  };
  useEffect(() => {
    let frame;
    const reflect = (event) => {
      const card = event.target.closest?.(".glass");
      if (!card) return;
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const rect = card.getBoundingClientRect();
        card.style.setProperty("--glass-x", `${event.clientX - rect.left}px`);
        card.style.setProperty("--glass-y", `${event.clientY - rect.top}px`);
      });
    };
    document.addEventListener("pointermove", reflect, { passive: true });
    return () => {
      document.removeEventListener("pointermove", reflect);
      cancelAnimationFrame(frame);
    };
  }, []);
  const current = NAV.find((n) => n[0] === view);
  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="ambient ambient-three" />
      <a className="skip-link" href="#main">
        Skip to dashboard
      </a>
      <aside className="sidebar glass">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setView("overview");
          }}
        >
          <span className="brand-symbol">
            <Icon name="activity" size={24} />
          </span>
          <span>
            STRIDE<small>PLAYER MOVEMENT LAB</small>
          </span>
        </a>
        <span className="nav-label">WORKSPACE</span>
        <nav aria-label="Dashboard sections">
          {NAV.map(([id, label, icon]) => (
            <button
              key={id}
              className={view === id ? "nav-item active" : "nav-item"}
              onClick={() => {
                setView(id);
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
              aria-current={view === id ? "page" : undefined}
              data-view={id}
            >
              <Icon name={icon} />
              <span>{label}</span>
              {view === id && <i />}
            </button>
          ))}
        </nav>
        <div className="sidebar-study">
          <span className="eyebrow">CURRENT STUDY</span>
          <div className="study-dots">
            <i />
            <i />
            <i />
          </div>
          <strong>{DATA.scope.players} players. One study.</strong>
          <p>
            {DATA.scope.matches} matches · {DATA.scope.distinct_minutes} minutes
            <br />
            Four selected segments
          </p>
          <span className="local-dot">Local · fully offline</span>
        </div>
        <div className="sidebar-footer">
          BCSE407L · Computer Vision
          <br />
          <span>Selected match segments</span>
        </div>
      </aside>
      <main id="main" className="main">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <Icon name="chevron" size={13} />
            <strong>{current[1]}</strong>
          </div>
          <div className="top-actions">
            <span className="snapshot-label">
              <i /> Study snapshot
            </span>
            <button
              className="icon-button"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              aria-label={
                theme === "dark"
                  ? "Switch to light theme"
                  : "Switch to dark theme"
              }
              title="Toggle appearance"
            >
              <Icon name={theme === "dark" ? "sun" : "moon"} />
            </button>
            <ExportMenu {...{ player, period }} />
          </div>
        </header>
        <div className="hero">
          <div>
            <span className="eyebrow">VISION-ONLY SOCCER ANALYSIS</span>
            <h1>
              {view === "overview" ? (
                <>
                  Player movement,
                  <br />
                  <em>in focus.</em>
                </>
              ) : (
                current[1]
              )}
            </h1>
            <p>
              {view === "overview"
                ? "Explore the movement. Understand the evidence."
                : "An interactive view of measured movement and its context."}
            </p>
          </div>
          <div className="hero-insight glass">
            <div className="hero-ring">
              <svg width="70" height="70" viewBox="0 0 70 70">
                <circle
                  cx="35"
                  cy="35"
                  r="29"
                  fill="none"
                  stroke="currentColor"
                  opacity=".12"
                  strokeWidth="3"
                />
                <circle
                  cx="35"
                  cy="35"
                  r="29"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeDasharray={`${duration ? (valid / duration) * 182 : 0} 182`}
                  transform="rotate(-90 35 35)"
                  strokeLinecap="round"
                />
              </svg>
              <span>
                {duration ? fmt((valid / duration) * 100, 0) : "—"}
                <small>%</small>
              </span>
            </div>
            <div>
              <strong>Observed movement</strong>
              <p>
                Valid time across this player's
                <br />
                selected early and late segments
              </p>
            </div>
          </div>
        </div>
        <div className="selection-bar glass">
          <div className="player-control">
            <div
              className={`jersey ${player.startsWith("A_") ? "blue" : "white"}`}
            >
              {jersey(player)}
            </div>
            <label htmlFor="player">
              <span className="eyebrow">SELECTED PLAYER</span>
              <select
                id="player"
                value={player}
                onChange={(e) => {
                  setPlayer(e.target.value);
                  setWindowSelection(null);
                }}
              >
                {PLAYERS.map((p) => (
                  <option key={p} value={p}>
                    {p.startsWith("A_") ? "Blue" : "White"} #{jersey(p)} · Match{" "}
                    {p.startsWith("A_") ? "A" : "B"}
                  </option>
                ))}
              </select>
              <small>{selectedName} · outfield player</small>
            </label>
          </div>
          <div className="period-control">
            <span className="eyebrow">MATCH PERIOD</span>
            <Segments
              label="Match period"
              value={period}
              onChange={setPeriod}
              options={[
                ["both", "Early + late"],
                ["early", "Early"],
                ["late", "Late"],
              ]}
            />
          </div>
          <div className="selection-context">
            <Icon name="clock" size={16} />
            <span>
              25-minute segments<small>5–30 / 60–85 configured min</small>
            </span>
          </div>
        </div>
        <div
          className="page-content"
          data-player={player}
          data-period={period}
          data-section={view}
        >
          {view === "overview" || view === "movement" ? (
            <MovementPage {...props} overview={view === "overview"} />
          ) : view === "gait" ? (
            <GaitPage {...props} />
          ) : view === "pitch" ? (
            <PitchPage {...props} />
          ) : view === "evaluation" ? (
            <EvaluationPage theme={theme} />
          ) : (
            <MethodsPage {...props} />
          )}
        </div>
        <footer className="main-footer">
          <span>
            <Icon name="shield" size={14} /> Movement changes are fatigue
            proxies, not a physiological diagnosis.
          </span>
          <span>
            Study snapshot ·{" "}
            {new Date(DATA.created_utc).toLocaleDateString("en-GB", {
              day: "2-digit",
              month: "short",
              year: "numeric",
            })}
          </span>
        </footer>
      </main>
      {windowSelection && (
        <WindowInspector
          window={windowSelection}
          metric={metric}
          onClose={() => setWindowSelection(null)}
        />
      )}
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
