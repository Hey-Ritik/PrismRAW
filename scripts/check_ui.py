"""Verify the studio redesign, comparison controls, upload guidance and responsive views."""

import json
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path("outputs/ui-v2")
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto("http://127.0.0.1:7860/?studio=2", wait_until="networkidle")
    page.locator(".sample-card").last.wait_for(timeout=30000)
    assert page.locator(".sample-card").count() == 4
    page.evaluate("Promise.all(Array.from(document.images).filter(i => i.getAttribute('src')).map(i => i.decode()))")
    page.screenshot(path=str(out / "desktop-start.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
    page.screenshot(path=str(out / "mobile-start.png"), full_page=True)
    page.set_viewport_size({"width": 1440, "height": 1100})
    # A real request verifies the new UI still drives the existing API.
    page.locator('[data-scene="004"]').click()
    with page.expect_response(lambda r: "/api/sample/004" in r.url and r.request.method == "POST", timeout=120000):
        page.get_by_role("button", name="Develop this scene").click()
    page.locator("#busy").wait_for(state="hidden", timeout=120000)
    assert not page.locator("#error").is_visible(), page.locator("#error").inner_text()
    assert page.locator("#output-grid .image-card").count() == 4
    assert page.locator("#metrics tr").count() == 3
    assert page.locator('#viewer-tabs [aria-pressed="true"]').inner_text() == "Hybrid"
    assert "dB" in page.locator("#quick-psnr").inner_text()
    page.locator("#compare-range").focus()
    page.locator("#compare-range").press("Home")
    page.locator("#compare-range").press("ArrowRight")
    assert page.locator("#compare-stage").evaluate("e => e.style.getPropertyValue('--split')") == "1%"
    page.locator("#compare-range").press("End")
    assert page.locator("#compare-range").input_value() == "100"
    page.locator("#compare-range").evaluate("e => {e.value=50;e.dispatchEvent(new Event('input'))}")
    page.get_by_role("button", name="View Classical", exact=True).click()
    assert page.locator("#left-label").inner_text() == "Sensor preview"
    page.get_by_role("button", name="Single view", exact=True).click()
    assert page.locator("#compare-range").is_disabled()
    page.get_by_role("button", name="Compare", exact=True).click()
    assert page.locator("#compare-range").is_enabled()
    page.get_by_role("button", name="View Reference", exact=True).click()
    assert page.locator("#quick-psnr").inner_text() == "—"
    page.get_by_role("button", name="View Hybrid", exact=True).click()
    page.locator("#zoom-current").click()
    assert page.locator("#lightbox").is_visible()
    page.keyboard.press("Escape")
    assert not page.locator("#lightbox").is_visible()
    page.evaluate("Promise.all(Array.from(document.images).filter(i => i.getAttribute('src')).map(i => i.decode()))")
    page.screenshot(path=str(out / "desktop-results.png"), full_page=True)
    page.locator(".measurement-panel > summary").click()
    assert page.locator("#metrics").is_visible()
    page.locator("details.stages > summary").click()
    assert page.locator("#stage-grid").is_visible()
    with page.expect_download() as download:
        page.locator("#download").click()
    download.value.save_as(str(out / "comparison.zip"))
    page.locator("#show-evaluation").click()
    page.locator("#evaluation table").wait_for(timeout=10000)
    page.locator("#show-evaluation").click()
    # Upload guidance: DNG has embedded metadata, arrays require a sidecar.
    page.get_by_role("tab", name="Upload RAW", exact=True).click()
    page.locator("#raw-file").set_input_files({"name": "capture.dng", "mimeType": "image/x-adobe-dng", "buffer": b"UI file-selection fixture; not submitted"})
    assert page.locator("#metadata-field").is_hidden()
    assert page.locator("#dng-note").is_visible()
    page.locator("#raw-file").set_input_files("data/processed/demo/003-noisy.npy")
    assert page.locator("#metadata-field").is_visible()
    assert page.locator("#metadata-file").evaluate("e => e.required")
    page.locator("#raw-file").set_input_files({"name": "photo.png", "mimeType": "image/png", "buffer": b"not raw"})
    assert page.locator("#error").is_visible()
    page.locator("#dismiss-error").click()
    assert page.locator("#error").is_hidden()
    page.get_by_role("tab", name="Sample scenes", exact=True).click()
    # Advanced precision is opt-in and its control is reachable.
    page.locator('.input-settings details summary').click()
    page.locator("#fp16").check()
    assert page.locator("#fp16").is_checked()
    page.locator("#fp16").uncheck()
    page.locator('.input-settings details summary').click()
    page.set_viewport_size({"width": 390, "height": 844})
    assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
    page.locator(".measurement-panel > summary").click()
    page.locator("details.stages > summary").click()
    page.screenshot(path=str(out / "mobile-results.png"), full_page=True)
    if errors:
        raise ValueError(errors)
    (out / "verification.json").write_text(json.dumps({"sample_selection": "passed", "real_processing": "passed", "comparison_slider_keyboard": "passed", "pipeline_selection": "passed", "single_view": "passed", "zoom": "passed", "metrics": "passed", "downloads": "passed", "evaluation": "passed", "upload_guidance": "passed", "error_state": "passed", "mobile_overflow": False, "javascript_errors": errors}, indent=2))
    browser.close()
print("Studio UI checks passed: processing, comparison, keyboard, zoom, metrics, uploads, downloads and responsive layout")
