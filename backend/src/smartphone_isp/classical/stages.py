"""Small reusable stages; callers must track sensor and colour-space metadata."""

import numpy as np


def normalize_raw(raw, black_level, white_level):
    """Normalize unprocessed sensor values; do not apply to normalized SIDD RAW.

    Levels may be scalar or arrays broadcastable to the RAW shape. Per-CFA
    channel metadata must first be expanded into the appropriate mosaic layout.
    """
    black = np.asarray(black_level, dtype=np.float32)
    white = np.asarray(white_level, dtype=np.float32)
    if not np.all(np.isfinite(white - black)) or np.any(white <= black):
        raise ValueError("White levels must be finite and greater than black levels")
    return np.clip((np.asarray(raw, dtype=np.float32) - black) / (white - black), 0, 1)


def apply_white_balance(rgb, gains):
    """Apply three RGB gains in linear camera RGB, not gamma-encoded sRGB."""
    image = np.asarray(rgb, dtype=np.float32)
    gains = np.asarray(gains, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected H x W x 3 RGB")
    if gains.shape != (3,) or not np.all(np.isfinite(gains)) or np.any(gains <= 0):
        raise ValueError("Expected three finite positive RGB gains")
    return image * gains


def apply_ccm(rgb, camera_to_linear_srgb):
    """Use an explicitly camera-to-linear-sRGB matrix, with column-vector convention.

    A DNG ColorMatrix cannot generally be passed directly as this matrix.
    Preserve out-of-gamut values here; handle them in the rendering stage.
    """
    image = np.asarray(rgb, dtype=np.float32)
    matrix = np.asarray(camera_to_linear_srgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected H x W x 3 RGB")
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Expected a finite 3 x 3 colour matrix")
    return image @ matrix.T


def linear_to_srgb(rgb):
    """Encode linear sRGB for display using the piecewise sRGB transfer curve."""
    image = np.clip(np.asarray(rgb, dtype=np.float32), 0, 1)
    return np.where(image <= 0.0031308, 12.92 * image,
                    1.055 * np.power(image, 1 / 2.4) - 0.055)
