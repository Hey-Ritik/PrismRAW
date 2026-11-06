"""FastAPI upload/compare service. Default binding is localhost only."""

import asyncio
import json
import os
from pathlib import Path
import tempfile
from contextlib import asynccontextmanager

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[3] / ".cache/torch"))

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image

from smartphone_isp.app.engine import ROOT, RESULTS, CHECKPOINTS, cached_model, compare, available_models, crop_sensor
from smartphone_isp.app.settings import runtime_settings
from smartphone_isp.data.arrays import load_raw
from smartphone_isp.data.dng import load_dng
from smartphone_isp.app.cors import add_frontend_cors

def warmup():
    import torch
    from smartphone_isp.evaluation.metrics import perceptual_model
    torch.set_num_threads(runtime_settings().threads)
    for method, enabled in available_models().items():
        if enabled and method in {"learned", "hybrid"}:
            cached_model(method, CHECKPOINTS[method].stat().st_mtime_ns)
    perceptual_model()


@asynccontextmanager
async def lifespan(application):
    await run_in_threadpool(warmup)
    yield


app = FastAPI(title="PrismRAW", docs_url="/api/docs", lifespan=lifespan)
add_frontend_cors(app)
FRONTEND = ROOT.parent / "frontend"
API_ONLY = os.environ.get("PRISMRAW_API_ONLY", "").lower() in {"1", "true"}
RESULTS.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=RESULTS), name="results")
if not API_ONLY:
    app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")
gate = asyncio.Semaphore(1)


@app.get("/")
def index():
    if API_ONLY:
        return {"service": "PrismRAW API", "health": "/api/health", "docs": "/api/docs"}
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    samples = [{"id": path.stem, "name": {"001": "Samsung S6 · low light", "003": "Nexus 6 · normal light", "004": "Google Pixel · indoor", "005": "iPhone 7 · low light"}.get(path.stem, path.stem)}
               for path in sorted((ROOT / "data/processed/demo").glob("*.json"))]
    settings = runtime_settings()
    return {"status": "ok", "models": available_models(), "samples": samples, "device": "CPU", "max_crop": settings.max_crop, "hosted": settings.hosted}


@app.post("/api/sample/{sample_id}")
async def sample(sample_id: str, adaptive: bool = True, include_fp16: bool = False):
    if sample_id not in {"001", "003", "004", "005"}:
        raise HTTPException(404, "Unknown sample")
    folder = ROOT / "data/processed/demo"
    try:
        metadata = json.loads((folder / f"{sample_id}.json").read_text())
        noisy = np.load(folder / f"{sample_id}-noisy.npy", allow_pickle=False)
        clean = np.load(folder / f"{sample_id}-clean.npy", allow_pickle=False)
        async with gate:
            methods = ("learned", "hybrid", "optimized") if include_fp16 else ("learned", "hybrid")
            return await run_in_threadpool(compare, noisy, metadata, clean=clean, adaptive=adaptive, methods=methods)
    except FileNotFoundError:
        raise HTTPException(404, "Sample assets are missing; run scripts/prepare_demo_assets.py")


async def store(upload, destination, limit):
    total = 0
    with destination.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                raise ValueError(f"Upload exceeds {limit // (1024*1024)} MB limit")
            stream.write(chunk)
    return destination


@app.post("/api/upload")
async def upload(raw: UploadFile = File(...), metadata: UploadFile | None = File(None),
                 reference: UploadFile | None = File(None), key: str = Form("data"), adaptive: bool = Form(True), include_fp16: bool = Form(False)):
    try:
        async with gate:
            with tempfile.TemporaryDirectory(prefix="prismraw-") as folder:
                folder = Path(folder)
                suffix = Path(raw.filename or "").suffix.lower()
                if suffix not in {".npy", ".npz", ".mat", ".dng"}:
                    raise ValueError("Upload Bayer RAW as DNG, NPY, NPZ, or MAT; JPEG is not RAW sensor data")
                path = await store(raw, folder / f"raw{suffix}", 64*1024*1024)
                if suffix == ".dng":
                    array, meta = await run_in_threadpool(load_dng, path, max_side=runtime_settings().max_crop)
                else:
                    if metadata is None:
                        raise ValueError("Array uploads require camera metadata JSON")
                    meta_path = await store(metadata, folder / "metadata.json", 64*1024)
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if not isinstance(meta, dict):
                        raise ValueError("Camera metadata must be a JSON object")
                    array = await run_in_threadpool(load_raw, path, key)
                _, local_crop = crop_sensor(array)
                def reference_crop(shape):
                    if tuple(shape) == tuple(array.shape):
                        return local_crop
                    if tuple(shape) == tuple(meta.get("original_sensor_shape", ())):
                        outer = meta["sensor_crop_yxhw"]
                        return (outer[0]+local_crop[0], outer[1]+local_crop[1], local_crop[2], local_crop[3])
                    if tuple(shape) == tuple(local_crop[2:]):
                        return (0, 0, *local_crop[2:])
                    raise ValueError("Reference must match sensor dimensions or the processed crop")
                clean, rgb = None, None
                if reference is not None and reference.filename:
                    ext = Path(reference.filename).suffix.lower()
                    ref_path = await store(reference, folder / f"reference{ext}", 64*1024*1024)
                    if ext in {".png", ".jpg", ".jpeg"}:
                        with Image.open(ref_path) as image:
                            if image.width * image.height > 24_000_000:
                                raise ValueError("RGB reference exceeds 24 megapixels")
                            y, x, h, w = reference_crop((image.height, image.width))
                            rgb = np.asarray(image.crop((x, y, x+w, y+h)).convert("RGB"), dtype=np.float32) / 255
                    elif ext in {".npy", ".npz", ".mat"}:
                        clean = await run_in_threadpool(load_raw, ref_path, key)
                        y, x, h, w = reference_crop(clean.shape)
                        clean = clean[y:y+h, x:x+w].copy()
                    else:
                        raise ValueError("Reference must be aligned RGB PNG/JPEG or matching clean RAW array")
                methods = ("learned", "hybrid", "optimized") if include_fp16 else ("learned", "hybrid")
                return await run_in_threadpool(compare, array, meta, clean=clean, reference_rgb=rgb, adaptive=adaptive, methods=methods)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/evaluation")
def evaluation():
    path = ROOT / "reports/final_results.json"
    if not path.exists():
        return {"status": "Evaluation in progress"}
    report = json.loads(path.read_text())
    result = {key: report[key] for key in ("split", "scene_ids", "patch_count", "metrics", "limitations")}
    benchmark = ROOT / "reports/benchmark.json"
    if benchmark.exists():
        result["benchmark"] = json.loads(benchmark.read_text())
    return result
