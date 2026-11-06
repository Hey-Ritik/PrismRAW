"""Integration checks against the running local server; no cloud uploads."""

import json
from pathlib import Path
import httpx
import time

report = {}
with httpx.Client(base_url="http://127.0.0.1:7860", timeout=120) as client:
    deadline = time.monotonic() + 60
    while True:
        try:
            client.get("/api/health", timeout=2).raise_for_status()
            break
        except (httpx.ConnectError, httpx.TimeoutException):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)
    with open("data/processed/demo/003-noisy.npy", "rb") as raw, open("data/processed/demo/003.json", "rb") as metadata:
        response = client.post("/api/upload", files={"raw": ("sensor.npy", raw), "metadata": ("camera.json", metadata)}, data={"adaptive": "true"})
        if not response.is_success:
            raise ValueError(response.text[:500])
        result = response.json()
        if result["metrics"] or "optimized" in result["images"]:
            raise ValueError("Unreferenced default upload must not invent metrics or force FP16")
        if set(result["processing_ms"]) != {"classical", "learned", "hybrid"}:
            raise ValueError("Expected three default pipelines")
        report["array_upload_without_reference"] = "passed"
        report["array_total_ms"] = result["total_ms"]
    with open("data/raw/sidd-reference/colorchart.dng", "rb") as raw:
        response = client.post("/api/upload", files={"raw": ("capture.dng", raw)}, data={"adaptive": "true"})
        response.raise_for_status()
        result = response.json()
        if result["crop_yxhw"][2:] != [512, 512] or result["metrics"]:
            raise ValueError("DNG crop or reference behavior incorrect")
        report["dng_upload_without_sidecar"] = "passed"
        report["dng_total_ms"] = result["total_ms"]
    response = client.post("/api/upload", files={"raw": ("photo.jpg", b"invalid")})
    if response.status_code != 400:
        raise ValueError("Expected actionable invalid-input response")
    report["invalid_input"] = "passed"
Path("outputs/browser-check/upload-verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
