"""Exercise an owned hosted-profile server locally, without creating a cloud service."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import httpx
import psutil

Path("work").mkdir(exist_ok=True)
environment = dict(os.environ, PYTHONPATH="src", PORT="7861", PRISMRAW_HOSTED="1", PRISMRAW_MAX_CROP="256", PRISMRAW_TORCH_THREADS="1")
log = Path("work/render-smoke-server.log").open("w", encoding="utf-8")
process = subprocess.Popen([sys.executable, "scripts/serve.py"], env=environment, stdout=log, stderr=subprocess.STDOUT,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
peak = [0]
stop = threading.Event()
def sample_memory():
    info = psutil.Process(process.pid)
    while not stop.wait(0.01):
        try:
            processes = [info, *info.children(recursive=True)]
            peak[0] = max(peak[0], sum(child.memory_info().rss for child in processes))
        except psutil.Error:
            return
thread = threading.Thread(target=sample_memory, daemon=True)
thread.start()
try:
    with httpx.Client(base_url="http://127.0.0.1:7861", timeout=120) as client:
        deadline = time.monotonic() + 120
        while True:
            if process.poll() is not None:
                raise RuntimeError("Hosted-profile server exited; see work/render-smoke-server.log")
            try:
                health = client.get("/api/health", timeout=2)
                health.raise_for_status()
                assert health.json()["hosted"] and health.json()["max_crop"] == 256
                break
            except (httpx.ConnectError, httpx.TimeoutException):
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        response = client.post("/api/sample/003")
        response.raise_for_status()
        sample = response.json()
        assert sample["crop_yxhw"][2:] == [256, 256]
        assert set(sample["processing_ms"]) == {"classical", "learned", "hybrid"}
        assert len(sample["metrics"]) == 3
        with open("data/processed/demo/003-noisy.npy", "rb") as raw, open("data/processed/demo/003.json", "rb") as meta, open("data/processed/demo/003-clean.npy", "rb") as clean:
            response = client.post("/api/upload", files={"raw": ("sensor.npy", raw), "metadata": ("camera.json", meta), "reference": ("clean.npy", clean)})
            response.raise_for_status()
            assert len(response.json()["metrics"]) == 3
        checks = {"environment_port": "passed", "hosted_health": "passed", "sample_256px": "passed", "array_reference_crop": "passed"}
        dng = Path("data/raw/sidd-reference/colorchart.dng")
        if dng.exists():
            with dng.open("rb") as raw:
                response = client.post("/api/upload", files={"raw": ("capture.dng", raw)})
                response.raise_for_status()
                result = response.json()
                assert result["crop_yxhw"][2:] == [256, 256]
                assert result["original_shape"] == [3000, 5328]
                assert not result["metrics"]
            checks["real_dng_bounded_decode"] = "passed"
        else:
            checks["real_dng_bounded_decode"] = "not run; optional reference DNG absent"
        report = {"checks": checks, "peak_process_tree_rss_mib": peak[0]/2**20, "sample_total_ms": sample["total_ms"],
                  "scope": "Local Windows hosted-profile smoke test; not a Linux Docker build or a Render free-instance memory/speed guarantee"}
        Path("reports/render_profile_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
finally:
    stop.set()
    thread.join()
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log.close()
