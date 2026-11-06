"""Capture fully decoded existing results without rerunning model inference."""

import json
from pathlib import Path
from playwright.sync_api import sync_playwright

reports = []
for path in Path("outputs/app").glob("*/report.json"):
    data = json.loads(path.read_text())
    if "optimized" in data["images"] and "reference" in data["images"]:
        reports.append((path.stat().st_mtime, data))
if not reports:
    raise ValueError("Run the browser integration check first")
data = max(reports, key=lambda item: item[0])[1]
data["download"] = f"/results/{data['id']}/comparison.zip"
out = Path("outputs/browser-check")
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    page.goto("http://127.0.0.1:7860", wait_until="networkidle")
    page.locator('.input-settings details summary').click()
    page.locator("#fp16").check()
    page.evaluate("data => render(data)", data)
    page.evaluate("Promise.all(Array.from(document.images).filter(image => image.src).map(image => image.decode()))")
    page.screenshot(path=str(out / "desktop-results.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.screenshot(path=str(out / "mobile-results.png"), full_page=True)
    browser.close()
print("Captured fully decoded desktop/mobile evidence from a verified existing result")
