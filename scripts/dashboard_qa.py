"""Check standalone dashboard rendering and player switches using installed headless Edge.

This presentation check prevents missing plots from appearing as movement evidence.
Playwright is an isolated test dependency, not a model or runtime requirement.
"""

import hashlib
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

root = Path.cwd()
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        headless=True,
    )
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto((root / "results/phase4/dashboard.html").as_uri(), wait_until="load")
    page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length === 4")
    page.screenshot(path=str(root / "results/phase4/dashboard_preview.png"))
    selections = []
    for player in ["A_blue_36", "B_white_17", "B_white_9"]:
        page.select_option("#player", player)
        page.wait_for_function(
            '(p) => document.querySelector(".panel.active").id === p', arg=player
        )
        selections.append(
            {
                "player": player,
                "active": page.locator(".panel.active").count(),
                "charts": page.locator(".panel.active .js-plotly-plot").count(),
            }
        )
        assert selections[-1]["active"] == 1 and selections[-1]["charts"] == 1
    browser.close()
report = {
    "status": "passed" if not errors else "failed",
    "javascript_errors": errors,
    "player_selector_checks": selections,
    "local_standalone_html": True,
    "dashboard_sha256": hashlib.sha256(
        (root / "results/phase4/dashboard.html").read_bytes()
    ).hexdigest(),
}
(root / "results/phase4/dashboard_qa.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
if errors:
    raise RuntimeError("Dashboard JavaScript errors")
