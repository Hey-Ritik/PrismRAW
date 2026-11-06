"""Bounded local inference and downloadable comparison artifacts."""

from functools import lru_cache
import json
from pathlib import Path
import time
import uuid
import zipfile

import numpy as np
from PIL import Image
import torch

from smartphone_isp.pipelines.classical import process, render_linear
from smartphone_isp.pipelines.hybrid import load_model as load_hybrid, process_hybrid
from smartphone_isp.pipelines.learned import load_model as load_learned, process_learned
from smartphone_isp.app.settings import runtime_settings

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "outputs" / "app"
CHECKPOINTS = {"learned": ROOT / "checkpoints/final-learned.pt",
               "hybrid": ROOT / "checkpoints/final-hybrid.pt",
               "optimized": ROOT / "checkpoints/final-hybrid-fp16.pt"}


@lru_cache(maxsize=2)
def cached_model(method, modified):
    loader = load_learned if method == "learned" else load_hybrid
    return loader(CHECKPOINTS[method])[0]


def available_models():
    return {"classical": True, **{key: path.exists() for key, path in CHECKPOINTS.items()}}


def crop_sensor(raw, limit=None):
    limit = runtime_settings().max_crop if limit is None else limit
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.size > 24_000_000 or min(raw.shape) < 16 or not np.isfinite(raw).all():
        raise ValueError("Expected a finite 2D Bayer sensor array, 16 pixels minimum and 24 megapixels maximum")
    h, w = [min(int(n) // 2 * 2, limit) for n in raw.shape]
    y, x = [(int(n) - size) // 4 * 2 for n, size in zip(raw.shape, (h, w))]
    return raw[y:y+h, x:x+w], (y, x, h, w)


def cleanup():
    import shutil
    RESULTS.mkdir(parents=True, exist_ok=True)
    for folder in RESULTS.iterdir():
        if folder.is_dir() and not folder.is_symlink() and folder.resolve().is_relative_to(RESULTS.resolve()) and time.time() - folder.stat().st_mtime > 3600:
            shutil.rmtree(folder)


def compare(raw, metadata, clean=None, reference_rgb=None, adaptive=True, methods=None):
    torch.set_num_threads(runtime_settings().threads)
    started = time.perf_counter()
    original_shape = raw.shape
    raw, crop = crop_sensor(raw)
    y, x, h, w = crop
    if clean is not None:
        if clean.shape == original_shape:
            clean = clean[y:y+h, x:x+w]
        elif clean.shape != raw.shape:
            raise ValueError("Clean RAW must match the input sensor array exactly")
    if reference_rgb is not None:
        if reference_rgb.shape[:2] == original_shape:
            reference_rgb = reference_rgb[y:y+h, x:x+w]
        elif reference_rgb.shape[:2] != raw.shape:
            raise ValueError("RGB reference must match full sensor dimensions or the processed crop; no automatic registration/resizing is applied")
    cleanup()
    identifier = uuid.uuid4().hex
    folder = RESULTS / identifier
    folder.mkdir()
    urls = lambda name: f"/results/{identifier}/{name}"
    def save(name, image):
        Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).save(folder / f"{name}.png")
        return urls(f"{name}.png")

    classical, stages, stats = process(raw, metadata, adaptive=adaptive)
    timings = {"classical": stats["processing_ms"]}
    del stages["denoised_linear"], stages["tone_mapped_linear"]
    # Save previews before neural inference so stage arrays do not overlap with
    # network activations or the perceptual metric's buffers.
    preview = np.repeat(stages.pop("normalized_mosaic")[..., None], 3, axis=-1)
    preview = np.clip(preview * stats["exposure_gain"], 0, 1) ** (1/2.2)
    images = {"sensor_preview": save("sensor-preview", preview)}
    del preview
    intermediate_urls = {"White-balanced mosaic": save("stage-balanced", np.clip(stages.pop("balanced_mosaic"), 0, 1) ** (1/2.2)),
                         "Demosaiced camera RGB": save("stage-demosaic", np.clip(stages.pop("camera_rgb"), 0, 1) ** (1/2.2)),
                         "Colour corrected": save("stage-colour", render_linear(stages.pop("linear_srgb"), stats["exposure_gain"])[0]),
                         "Denoised + tone mapped": save("stage-denoised", stages.pop("encoded_srgb"))}
    del stages
    target = reference_rgb
    target_kind = "Uploaded RGB reference (alignment supplied by user)" if reference_rgb is not None else None
    if clean is not None:
        clean_display, clean_stages, clean_stats = process(clean, metadata, adaptive=False, denoise=False)
        clean_linear = clean_stages["linear_srgb"]
        del clean_display, clean_stages, clean_stats
        target = render_linear(clean_linear, stats["exposure_gain"])[0]
        del clean_linear
        target_kind = "Fixed-rendered matching clean RAW; approximate colour calibration"
    metrics = {}
    if target is not None:
        from smartphone_isp.evaluation.metrics import quality

    def record(method, image):
        if target is not None:
            metrics[method] = quality(image, target, perceptual=min(raw.shape) >= 64)
        images[method] = save(method, image)

    record("classical", classical)
    del classical
    enabled = available_models()
    for method in methods or ("learned", "hybrid", "optimized"):
        if method == "classical" or not enabled.get(method):
            continue
        # The optional FP16 model is neither preloaded nor kept in the cache.
        model = (load_hybrid(CHECKPOINTS[method])[0] if method == "optimized"
                 else cached_model(method, CHECKPOINTS[method].stat().st_mtime_ns))
        before = time.perf_counter()
        if method == "learned":
            unused_display, intermediate, method_stats = process_learned(raw, metadata, model, stats["exposure_gain"])
            predicted = intermediate["predicted_linear"]
            del unused_display, intermediate, method_stats
            image = render_linear(predicted, stats["exposure_gain"], stats["sharpen_strength"])[0]
            del predicted
        else:
            image, intermediate, method_stats = process_hybrid(raw, metadata, model, adaptive=adaptive)
            del intermediate, method_stats
        timings[method] = (time.perf_counter() - before) * 1000
        del model
        record(method, image)
        del image
    if target is not None:
        images["reference"] = save("reference", target)
    public_metadata = {key: metadata[key] for key in ("normalized", "bayer_pattern", "white_balance", "camera_to_linear_srgb", "colour_transform", "dataset", "scene_id") if key in metadata}
    report = {"id": identifier, "images": images, "intermediates": intermediate_urls,
              "processing_ms": timings, "metrics": metrics, "reference_kind": target_kind,
              "crop_yxhw": [y+metadata.get("sensor_crop_yxhw", [0, 0])[0], x+metadata.get("sensor_crop_yxhw", [0, 0])[1], h, w],
              "original_shape": metadata.get("original_sensor_shape", list(original_shape)),
              "adaptive": adaptive, "metadata": public_metadata,
              "settings": {key: stats[key] for key in ("noise_estimate", "exposure_gain", "bilateral_sigma", "sharpen_strength")},
              "limitations": ["Model quality varies; classical is the current reference baseline", "Camera colour calibration is approximate", "CPU timings exclude model loading and file I/O; these are single runs, not benchmark medians", f"RAW processing uses a central crop up to {runtime_settings().max_crop}x{runtime_settings().max_crop}, preserving Bayer phase"],
              "total_ms": (time.perf_counter() - started) * 1000}
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with zipfile.ZipFile(folder / "comparison.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in folder.iterdir():
            if path.suffix in {".png", ".json"}:
                archive.write(path, path.name)
    report["download"] = urls("comparison.zip")
    report["report"] = urls("report.json")
    return report
