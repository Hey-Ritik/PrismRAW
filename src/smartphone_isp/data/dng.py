"""Visible Bayer sensor extraction; never call a hidden ISP postprocess routine."""

import numpy as np


def load_dng(path, max_side=None):
    import rawpy
    try:
        camera = rawpy.imread(str(path))
    except rawpy.LibRawError as exc:
        raise ValueError("DNG could not be decoded; use a supported Bayer RAW file") from exc
    with camera:
        if camera.raw_pattern is None or camera.raw_pattern.shape != (2, 2):
            raise ValueError("Only 2x2 Bayer RAW sensors are supported")
        original_shape = camera.raw_image_visible.shape
        if len(original_shape) != 2 or np.prod(original_shape) > 24_000_000:
            raise ValueError("DNG sensor must be 2D and no larger than 24 megapixels")
        height, width = [int(n)//2*2 for n in original_shape]
        y, x = 0, 0
        if max_side is not None:
            if max_side < 16 or max_side % 2:
                raise ValueError("DNG crop must be even and at least 16 pixels")
            height, width = min(height, max_side), min(width, max_side)
            y, x = [(int(n)-size)//4*2 for n, size in zip(original_shape, (height, width))]
        region = (slice(y, y+height), slice(x, x+width))
        # Crop before float conversion and per-pixel calibration allocations.
        colors = camera.raw_colors_visible[region].copy()
        desc = camera.color_desc.decode("ascii").rstrip("\x00")
        if desc[:3] != "RGB":
            raise ValueError("Unsupported camera colour layout")
        pattern = "".join(desc[int(i)] for i in colors[:2, :2].ravel())
        if pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
            raise ValueError("Unsupported Bayer pattern")
        black = np.asarray(camera.black_level_per_channel, dtype=np.float32)
        levels = camera.camera_white_level_per_channel
        white = np.asarray(levels if levels is not None else [camera.white_level] * 4, dtype=np.float32)
        if np.any(white[:3] <= black[:3]):
            raise ValueError("Invalid RAW black/white levels")
        # Some profiles leave the second green's white level unspecified.
        if len(white) == 4 and white[3] <= black[3]:
            white[3] = white[1]
        raw = camera.raw_image_visible[region].astype(np.float32)
        raw = np.clip((raw - black[colors]) / (white[colors] - black[colors]), 0, 1)
        gains = np.asarray(camera.camera_whitebalance[:3], dtype=np.float32)
        if not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("Camera white-balance metadata unavailable; supply an explicit calibrated array instead")
        gains /= gains[1]
        camera_to_xyz = np.asarray(camera.rgb_xyz_matrix[:3], dtype=np.float64).T
        sums = camera_to_xyz.sum(axis=1)
        if np.any(np.abs(sums) < 1e-8) or not np.isfinite(camera_to_xyz).all():
            # Generic DNG files can have a valid profile but no LibRaw model matrix.
            import tifffile
            with tifffile.TiffFile(path) as tiff:
                tags = tiff.pages[0].tags
                tag = tags.get("ColorMatrix2") or tags.get("ColorMatrix1")
                if tag is None:
                    raise ValueError("Camera colour transform unavailable")
                values = np.asarray(tag.value, dtype=np.float64)
                if values.size == 18:
                    if np.any(values[1::2] == 0):
                        raise ValueError("Invalid DNG rational colour matrix")
                    values = values[::2] / values[1::2]
                if values.size != 9 or not np.isfinite(values).all():
                    raise ValueError("Unsupported DNG colour profile")
                xyz_to_camera = values.reshape(3, 3)
                if np.any(np.abs(xyz_to_camera.sum(axis=1)) < 1e-8):
                    raise ValueError("Invalid DNG colour profile")
                xyz_to_camera /= xyz_to_camera.sum(axis=1, keepdims=True)
                camera_to_xyz = np.linalg.inv(xyz_to_camera)
                sums = camera_to_xyz.sum(axis=1)
        camera_to_xyz *= (np.array([0.95047, 1, 1.08883]) / sums)[:, None]
        xyz_to_rgb = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.969266, 1.8760108, 0.041556], [0.0556434, -0.2040259, 1.0572252]])
        metadata = {"normalized": True, "bayer_pattern": pattern, "white_balance": gains.tolist(),
                    "camera_to_linear_srgb": (xyz_to_rgb @ camera_to_xyz).tolist(),
                    "colour_transform": "Approximate LibRaw/profile single-illuminant camera-to-XYZ, neutral-normalized to D65; not full dual-illuminant DNG calibration",
                    "orientation": "sensor order; orientation metadata not applied", "sensor_flip": int(camera.sizes.flip),
                    "original_sensor_shape": list(original_shape), "sensor_crop_yxhw": [y, x, height, width]}
    h, w = raw.shape
    metadata["even_crop"] = [h - h % 2, w - w % 2]
    return raw[:h-h%2, :w-w%2], metadata
