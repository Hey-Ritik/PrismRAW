"""Learned-only RAW reconstruction; common display rendering follows the network."""

import time
import numpy as np
import torch

from smartphone_isp.classical.stages import normalize_raw
from smartphone_isp.data.bayer import pack_bayer
from smartphone_isp.models.raw_to_rgb import RawToRGBNet
from smartphone_isp.pipelines.classical import render_linear


def calibration_vector(metadata):
    gains = np.asarray(metadata["white_balance"], dtype=np.float32)
    matrix = np.asarray(metadata["camera_to_linear_srgb"], dtype=np.float32)
    pattern = metadata["bayer_pattern"].upper()
    patterns = ("RGGB", "BGGR", "GRBG", "GBRG")
    if gains.shape != (3,) or matrix.shape != (3, 3) or pattern not in patterns:
        raise ValueError("Invalid camera metadata shapes or Bayer pattern")
    if not np.isfinite(gains).all() or np.any(gains <= 0) or not np.isfinite(matrix).all():
        raise ValueError("Camera metadata must be finite with positive gains")
    return np.concatenate([gains, matrix.ravel(), np.array([pattern == p for p in patterns], dtype=np.float32)])


def load_model(checkpoint, device="cpu"):
    document = torch.load(checkpoint, map_location=device, weights_only=True)
    if document.get("format") != "prismraw-bayer-v1" or document.get("epoch", 0) < 1:
        raise ValueError("Expected a trained PrismRAW Bayer-to-RGB checkpoint")
    model = RawToRGBNet(document["base_channels"]).to(device)
    model.load_state_dict(document["state_dict"])
    model.eval()
    return model, document


def reconstruct(model, packed, calibration, device="cpu"):
    source = torch.from_numpy(np.ascontiguousarray(packed.transpose(2, 0, 1))).unsqueeze(0).to(device)
    maps = torch.as_tensor(calibration, dtype=source.dtype, device=device).view(1, 16, 1, 1)
    maps = maps.expand(1, 16, *packed.shape[:2])
    with torch.inference_mode():
        result = model(source, maps)
    return result[0].permute(1, 2, 0).cpu().numpy()


def process_learned(raw, metadata, model, exposure=1.0):
    started = time.perf_counter()
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or min(raw.shape) < 8 or not np.isfinite(raw).all():
        raise ValueError("Expected finite 2D Bayer array, at least 8x8")
    if type(metadata.get("normalized")) is not bool:
        raise ValueError("Explicit normalized boolean metadata required")
    if not np.isfinite(exposure) or exposure <= 0:
        raise ValueError("Exposure must be finite and positive")
    if metadata["normalized"]:
        if raw.min() < -0.1 or raw.max() > 1.1:
            raise ValueError("Normalized RAW outside supported [-0.1,1.1] range")
        normalized = np.clip(raw, 0, 1)
    else:
        if not all(np.isscalar(metadata.get(k)) for k in ("black_level", "white_level")):
            raise ValueError("Scalar black_level and white_level required")
        normalized = normalize_raw(raw, metadata["black_level"], metadata["white_level"])
    packed = pack_bayer(normalized, metadata["bayer_pattern"])
    linear = reconstruct(model, packed, calibration_vector(metadata))
    output, toned, encoded = render_linear(linear, exposure)
    return output, {"normalized_mosaic": normalized, "predicted_linear": linear,
                    "tone_mapped_linear": toned, "encoded_srgb": encoded}, {
        "method": "learned", "shape": list(output.shape), "exposure_gain": exposure,
        "processing_ms": (time.perf_counter() - started) * 1000,
        "timing_scope": "CPU normalization, packing, network and display render; excludes model load and I/O",
        "note": "No classical demosaicing, white-balance multiplication, colour matrix application, or denoising before network; common fixed display rendering follows"}
