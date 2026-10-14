"""Exercise offline React dashboard behavior, scientific values and responsive layouts."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path

from playwright.sync_api import sync_playwright


def verify_dashboard():
    """Use real browser interactions to verify missingness, export and evidence presentation."""
    root = Path(__file__).resolve().parents[1]
    output = root / "results/phase4"
    errors, requests, checks = [], [], []

    def check(name, condition=True):
        """Record a reviewable interaction and fail if it misrepresents recorded evidence."""
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    def wait_plot(page, identifier):
        """Wait for actual Plotly initialization rather than a placeholder container."""
        page.wait_for_selector(f'#{identifier}[data-ready="true"]', timeout=20000)

    def nav(page, view):
        """Open a section through its visible navigation, including the mobile layout."""
        page.locator(f'[data-view="{view}"]').click()
        page.wait_for_selector(f'.page-content[data-section="{view}"]')

    with sync_playwright() as driver:
        browser = driver.chromium.launch(
            executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            headless=True,
        )
        context = browser.new_context(
            viewport={"width": 1512, "height": 1180}, accept_downloads=True
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "request",
            lambda request: (
                requests.append(request.url)
                if request.url.startswith(("http://", "https://"))
                else None
            ),
        )
        page.goto((output / "dashboard.html").as_uri(), wait_until="load")
        wait_plot(page, "movement-chart")
        wait_plot(page, "comparison-chart")
        data = page.locator("#study-data").evaluate("e => JSON.parse(e.textContent)")
        check(
            "scope_and_raw_metrics_retained",
            len(data["windows"]) == 72
            and len(data["cycles"]) == 63
            and len(data["analysis"]["metrics"]) == 24,
        )
        pairs = page.locator("#comparison-chart").evaluate(
            "e => ({early: e.data[0].y, late: e.data[1].y, x: e.data[0].x})"
        )
        check(
            "qualified_pair_values_and_missing_periods",
            pairs["late"][1:] == [None, None]
            and math.isclose(pairs["early"][0], 2.2068928331405906, abs_tol=1e-10)
            and math.isclose(pairs["late"][0], 1.930137009462636, abs_tol=1e-10)
            and len(pairs["x"]) == 3,
        )
        page.screenshot(path=str(output / "dashboard_preview.png"))
        for player in ["A_blue_36", "B_white_17", "B_white_9"]:
            page.select_option("#player", player)
            page.wait_for_selector(f'.page-content[data-player="{player}"]')
            wait_plot(page, "movement-chart")
            check(f"player_selection_{player}", page.locator("#player").input_value() == player)

        # Only B_white_9 has a qualified own-player movement baseline.
        page.get_by_role("group", name="Chart value mode").get_by_role(
            "button", name="Baseline %"
        ).click()
        page.wait_for_function("document.querySelector('#movement-chart')?.data?.[0]?.y?.[0] < 10")
        actual = page.locator("#movement-chart").evaluate("e => e.data[0].y[0]")
        expected = next(
            w["avg_speed_pct"] for w in data["windows"] if w["player_id"] == "B_white_9"
        )
        check("baseline_percent_matches_saved_value", math.isclose(actual, expected, abs_tol=1e-10))
        page.select_option("#player", "A_blue_36")
        page.get_by_role("heading", name="Own-player baseline unavailable").wait_for()
        check("unavailable_baseline_is_not_zero_filled")
        page.get_by_role("group", name="Chart value mode").get_by_role(
            "button", name="Measured", exact=True
        ).click()
        wait_plot(page, "movement-chart")
        for metric in ["peak_speed", "p90_running_speed", "high_intensity_distance_per_minute"]:
            page.select_option("#metric", metric)
            check(f"movement_metric_{metric}", page.locator("#metric").input_value() == metric)
        page.select_option("#metric", "avg_speed")
        wait_plot(page, "movement-chart")
        point = page.locator("#movement-chart .scatterlayer .point").first
        point.scroll_into_view_if_needed()
        bounds = point.bounding_box()
        page.mouse.click(bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2)
        page.get_by_role("dialog").wait_for()
        check("chart_point_exposure_inspector", "100.2s" in page.get_by_role("dialog").inner_text())
        page.keyboard.press("Tab")
        check(
            "dialog_keyboard_focus",
            page.get_by_role("button", name="Close window inspector").evaluate(
                "e => e === document.activeElement"
            ),
        )
        page.keyboard.press("Escape")
        page.get_by_role("dialog").wait_for(state="hidden")

        page.get_by_role("group", name="Match period").get_by_role(
            "button", name="Early", exact=True
        ).click()
        page.get_by_role("button", name="Export data").click()
        with page.expect_download() as download_event:
            page.get_by_role("button", name="Player window metrics").click()
        downloaded = download_event.value
        table = list(csv.DictReader(io.StringIO(Path(downloaded.path()).read_text("utf-8"))))
        check(
            "selected_csv_export_exact_scope",
            len(table) == 12
            and all(r["player_id"] == "A_blue_36" and r["half"] == "1" for r in table),
        )
        page.get_by_role("group", name="Match period").get_by_role(
            "button", name="Early + late"
        ).click()

        nav(page, "gait")
        wait_plot(page, "gait-chart")
        page.get_by_text("59 accepted cycles in selection", exact=True).wait_for()
        page.get_by_role("group", name="Gait chart horizontal axis").get_by_role(
            "button", name="Running speed"
        ).click()
        page.select_option('select[aria-label="Gait metric"]', "knee_angle_range")
        page.get_by_role("group", name="Cycle leg filter").get_by_role(
            "button", name="Left", exact=True
        ).click()
        check(
            "gait_speed_axis_and_leg_filter",
            page.locator(".table-card tbody").inner_text().lower().count("right") == 0,
        )
        page.select_option("#player", "B_white_9")
        page.get_by_role("heading", name="No accepted cycles in this selection").wait_for()
        check("gait_empty_state_preserves_missingness")
        page.select_option("#player", "A_blue_36")

        nav(page, "pitch")
        wait_plot(page, "pitch-chart")
        trajectory = page.locator("#pitch-chart").evaluate(
            "e => ({traceCount:e.data.length, "
            "breaks:e.data.slice(0,2).map(t=>t.x.filter(v=>v===null).length)})"
        )
        check("trajectory_breaks_preserved", all(n > 0 for n in trajectory["breaks"]))
        page.locator("#pitch-time").evaluate("""e => {
            Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,'45');
            e.dispatchEvent(new Event('input',{bubbles:true}));
            e.dispatchEvent(new Event('change',{bubbles:true}));
        }""")
        page.get_by_text("No observation at cursor", exact=True).wait_for()
        check("unprocessed_match_time_has_no_current_position")
        page.get_by_role("group", name="Pitch display").get_by_role(
            "button", name="Occupancy"
        ).click()
        page.wait_for_function(
            "document.querySelector('#pitch-chart')?.data?.[0]?.type === 'histogram2d'"
        )
        check("occupancy_mode")
        page.get_by_role("button", name="Play trajectory playback").click()
        page.get_by_role("button", name="Pause trajectory playback").wait_for()
        page.get_by_role("button", name="Pause trajectory playback").click()
        check("trajectory_play_pause")

        nav(page, "evaluation")
        wait_plot(page, "broadcast-chart")
        check(
            "evaluation_has_all_five_clips",
            len(page.locator("#broadcast-chart").evaluate("e => e.data[0].x")) == 5,
        )
        check(
            "detector_and_pose_results_visible",
            "94.68%" in page.locator(".page-content").inner_text()
            and "98/100" in page.locator(".page-content").inner_text(),
        )
        nav(page, "methods")
        page.get_by_role("group", name="Methods detail").get_by_role(
            "button", name="Analysis tables"
        ).click()
        page.get_by_role("heading", name="Baseline exposure").wait_for()
        check(
            "baseline_trend_sensitivity_candidate_and_test_tables",
            page.locator(".table-card").count() == 5,
        )
        nav(page, "movement")
        check("full_window_and_status_tables", page.locator(".table-card").count() == 2)
        page.get_by_role("textbox", name="Search table").last.fill("second")
        check(
            "window_search_filter",
            "first" not in page.locator(".table-card").last.locator("tbody").inner_text().lower(),
        )

        # Check all views at mobile width; table scrolling stays inside the table.
        page.set_viewport_size({"width": 390, "height": 844})
        for view in ["overview", "movement", "gait", "pitch", "evaluation", "methods"]:
            nav(page, view)
            page.wait_for_timeout(100)
            check(
                f"mobile_no_page_overflow_{view}",
                not page.evaluate("document.documentElement.scrollWidth > innerWidth + 1"),
            )
        nav(page, "overview")
        wait_plot(page, "movement-chart")
        page.screenshot(path=str(output / "dashboard_mobile_preview.png"), full_page=True)
        page.set_viewport_size({"width": 1512, "height": 1180})
        page.get_by_role("button", name="Switch to light theme").click()
        page.wait_for_function("document.documentElement.dataset.theme === 'light'")
        wait_plot(page, "movement-chart")
        page.screenshot(path=str(output / "dashboard_light_preview.png"))
        check("light_theme_and_dark_toggle")
        page.get_by_role("button", name="Switch to dark theme").click()

        # URL state survives a reload; no external data, font or script requests occur.
        page.select_option("#player", "B_white_9")
        nav(page, "movement")
        page.reload(wait_until="load")
        page.wait_for_selector('.page-content[data-player="B_white_9"][data-section="movement"]')
        check("deep_link_selection_survives_reload")
        check("zero_external_runtime_requests", not requests)
        check("zero_javascript_errors", not errors)
        browser.close()
    report = {
        "status": "passed",
        "javascript_errors": errors,
        "external_runtime_requests": requests,
        "checks": checks,
        "passed_checks": len(checks),
        "local_standalone_html": True,
        "viewport_checks": ["1512x1180", "390x844"],
        "dashboard_sha256": hashlib.sha256((output / "dashboard.html").read_bytes()).hexdigest(),
    }
    (output / "dashboard_qa.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(verify_dashboard(), indent=2))
