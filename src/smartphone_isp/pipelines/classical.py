"""Transparent NumPy classical ISP for explicitly described Bayer arrays."""

import time
import numpy as np

from smartphone_isp.classical.stages import normalize_raw, apply_ccm, linear_to_srgb
from smartphone_isp.data.bayer import PATTERNS


def _blur(image):
    pad = np.pad(image, ((1, 1), (1, 1)) + ((0, 0),) * (image.ndim - 2), mode="reflect")
    h, w = image.shape[:2]
    return sum(pad[y:y+h, x:x+w] * weight for y, x, weight in
               ((0, 0, 1), (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 1, 4),
                (1, 2, 2), (2, 0, 1), (2, 1, 2), (2, 2, 1))) / 16


def demosaic(raw, pattern):
    """Normalized bilinear interpolation; pattern refers to array's top-left pixel."""
    h, w = raw.shape
    planes = []
    for color in "RGB":
        mask = np.zeros((h, w), dtype=np.float32)
        for i, value in enumerate(pattern):
            if value == color:
                mask[i // 2::2, i % 2::2] = 1
        plane = _blur(raw * mask) / np.maximum(_blur(mask), 1e-8)
        planes.append(np.where(mask > 0, raw, plane))
    return np.stack(planes, axis=-1)


def bilateral(image, sigma):
    """Small 5x5 edge-preserving baseline; not optimized for mobile inference."""
    if sigma <= 0:
        return image.copy()
    h, w = image.shape[:2]
    pad = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode="reflect")
    numerator = np.zeros_like(image)
    denominator = np.zeros((h, w, 1), dtype=np.float32)
    for y in range(5):
        for x in range(5):
            neighbor = pad[y:y+h, x:x+w]
            distance = np.mean((neighbor - image) ** 2, axis=-1, keepdims=True)
            weight = np.exp(-distance / (2 * sigma**2) - ((y-2)**2 + (x-2)**2) / 4)
            numerator += weight * neighbor
            denominator += weight
    return numerator / denominator


def process(raw, metadata, adaptive=True):
    """Return display RGB, named intermediates, and measured processing statistics.

    Metadata requires normalized(bool), bayer_pattern, white_balance RGB gains,
    and camera_to_linear_srgb (3x3 column-vector convention). For unnormalized
    input, scalar black_level and white_level are also required.
    """
    started = time.perf_counter()
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or min(raw.shape) < 4 or any(n % 2 for n in raw.shape):
        raise ValueError("RAW must be a 2D even-sized Bayer array, at least 4x4")
    if not np.all(np.isfinite(raw)):
        raise ValueError("RAW contains nonfinite values")
    pattern = metadata.get("bayer_pattern", "").upper()
    if pattern not in PATTERNS:
        raise ValueError("An explicit supported bayer_pattern is required")
    if type(metadata.get("normalized")) is not bool:
        raise ValueError("Metadata must explicitly declare normalized: true or false")
    gains = np.asarray(metadata.get("white_balance"), dtype=np.float32)
    if gains.shape != (3,) or not np.all(np.isfinite(gains)) or np.any(gains <= 0):
        raise ValueError("white_balance must contain three finite positive RGB gains")
    if "camera_to_linear_srgb" not in metadata:
        raise ValueError("An explicit camera_to_linear_srgb matrix is required")
    if metadata["normalized"]:
        # Small negative/above-white values may occur in floating sensor arrays.
        if raw.min() < -0.1 or raw.max() > 1.1:
            raise ValueError("Normalized RAW values are outside the supported [-0.1,1.1] range")
        normalized = np.clip(raw, 0, 1)
    else:
        if not all(np.isscalar(metadata.get(k)) for k in ("black_level", "white_level")):
            raise ValueError("This adapter requires scalar black_level and white_level")
        normalized = normalize_raw(raw, metadata["black_level"], metadata["white_level"])
    balanced = normalized.copy()
    for i, color in enumerate(pattern):
        balanced[i // 2::2, i % 2::2] *= gains["RGB".index(color)]
    camera_rgb = demosaic(balanced, pattern)
    linear = apply_ccm(camera_rgb, metadata["camera_to_linear_srgb"])
    # Difference of same-colour neighbors; texture can inflate this heuristic.
    difference = normalized[:, 2:] - normalized[:, :-2]
    noise = float(np.median(np.abs(difference - np.median(difference))) / (0.6745 * np.sqrt(2)))
    luminance = np.maximum(linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), 0)
    median_luma = float(np.median(luminance))
    sigma = float(np.clip(noise * 2, 0.005, 0.08)) if adaptive else 0.03
    denoised = bilateral(linear, sigma)
    exposure = float(np.clip(0.18 / max(median_luma, 1e-4), 0.5, 3)) if adaptive else 1.0
    exposed = np.maximum(denoised * exposure, 0)
    # One luminance-dependent scale for all channels to limit hue shifts.
    lum = exposed @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    toned = exposed / (1 + lum[..., None])
    encoded = linear_to_srgb(toned)
    sharpen = float(0.15 / (1 + 20 * noise)) if adaptive else 0.15
    output = np.clip(encoded + sharpen * (encoded - _blur(encoded)), 0, 1)
    stages = {"normalized_mosaic": normalized, "balanced_mosaic": balanced,
              "camera_rgb": camera_rgb, "linear_srgb": linear,
              "denoised_linear": denoised, "tone_mapped_linear": toned,
              "encoded_srgb": encoded}
    stats = {"method": "classical", "adaptive": adaptive, "shape": list(output.shape),
             "noise_estimate": noise, "median_linear_luminance": median_luma,
             "bilateral_sigma": sigma, "exposure_gain": exposure,
             "sharpen_strength": sharpen,
             "processing_ms": (time.perf_counter() - started) * 1000,
             "timing_scope": "array processing only; excludes load/save",
             "note": "Noise estimate is a heuristic and can respond to scene texture."}
    return output, stages, stats
