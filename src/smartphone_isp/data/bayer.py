"""Bayer packing without demosaicing or hidden colour transformations."""

import numpy as np

PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}


def pack_bayer(raw, pattern="RGGB"):
    """Pack into semantic R, G-on-R-row, G-on-B-row, B channels.

    Output shape is H/2 x W/2 x 4. Reject odd sizes rather than silently crop.
    """
    image = np.asarray(raw)
    pattern = pattern.upper()
    if pattern not in PATTERNS:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    if image.ndim != 2 or any(size == 0 or size % 2 for size in image.shape):
        raise ValueError("Expected a nonempty 2D Bayer mosaic with even dimensions")
    positions = {color: divmod(pattern.index(color), 2) for color in ("R", "B")}
    r, b = positions["R"], positions["B"]
    indices = (r, (r[0], 1 - r[1]), (b[0], 1 - b[1]), b)
    return np.stack([image[row::2, col::2] for row, col in indices], axis=-1)
