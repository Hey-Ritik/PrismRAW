"""Desktop/mobile browser integration check using the installed Edge browser."""

import json
from pathlib import Path
from playwright.sync_api import sync_playwright

output = Path("outputs/browser-check")
output.mkdir(parents=True, exist_ok=True)
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto("http://127.0.0.1:7860", wait_until="networkidle")
    page.get_by_text("Local engine ·", exact=False).first.wait_for(timeout=30000)
    page.screenshot(path=str(output / "desktop-initial.png"), full_page=True)
    page.get_by_role("button", name="Develop this scene").click()
    page.locator("#results").wait_for(state="visible", timeout=120000)
    page.locator("#busy").wait_for(state="hidden", timeout=120000)
    default_duration = page.locator("#duration").inner_text()
    if page.locator("#metrics tr").count() != 3:
        raise ValueError("Default CPU flow should use three FP32 pipelines")
    page.screenshot(path=str(output / "desktop-default.png"), full_page=True)
    page.locator('.input-settings details summary').click()
    page.locator("#fp16").check()
    page.locator("#run-sample").scroll_into_view_if_needed()
    with page.expect_response(lambda response: "/api/sample/" in response.url and response.request.method == "POST", timeout=120000):
        page.get_by_role("button", name="Develop this scene").click()
    page.locator("#results").wait_for(state="visible", timeout=120000)
    page.locator("#busy").wait_for(state="hidden", timeout=120000)
    if page.locator("#error").is_visible():
        raise ValueError(page.locator("#error").inner_text())
    count = page.locator("#output-grid .image-card").count()
    if count < 5:
        raise ValueError(f"Expected four pipelines plus reference; got {count}")
    if page.locator("#metrics tr").count() != 4:
        raise ValueError("Expected four metric rows")
    page.evaluate("Promise.all(Array.from(document.images).filter(image => image.src).map(image => image.decode()))")
    page.screenshot(path=str(output / "desktop-results.png"), full_page=True)
    with page.expect_download() as download:
        page.get_by_text("Download results", exact=False).click()
    download.value.save_as(str(output / "comparison.zip"))
    page.set_viewport_size({"width": 390, "height": 844})
    page.screenshot(path=str(output / "mobile-results.png"), full_page=True)
    if page.evaluate("document.documentElement.scrollWidth > window.innerWidth"):
        raise ValueError("Mobile layout has horizontal overflow")
    if errors:
        raise ValueError(errors)
    page.locator("#show-evaluation").click()
    page.locator("#evaluation table").wait_for(timeout=10000)
    (output / "verification.json").write_text(json.dumps({"desktop": "passed", "mobile": "passed", "pipelines": count-1, "default_total_time": default_duration, "evaluation_summary": "passed", "download": "passed", "javascript_errors": errors}, indent=2))
    browser.close()
print("Desktop, mobile, sample processing, metrics, download and JavaScript checks passed")
