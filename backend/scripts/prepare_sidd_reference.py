"""Prepare an even crop from the authors' clean reference RAW for ISP validation.

This is not a noisy/clean training pair. The colour transform is an approximate
single-illuminant rendering convention inspired by the authors' baseline,
not a calibrated dual-illuminant DNG processor or an exact SIDD RGB reproduction.
"""

import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from smartphone_isp.data.arrays import load_raw

root = Path("data/raw/sidd-reference")
m = loadmat(root / "0001_METADATA_RAW_010.MAT", simplify_cells=True)["metadata"]
tags = {int(tag["ID"]): tag["Value"] for tag in m["UnknownTags"]}
pattern = "".join("RGB"[int(i)] for i in np.asarray(tags[33422]).ravel())
xyz_to_cam = np.asarray(m["ColorMatrix2"]).reshape(3, 3)
xyz_to_cam /= xyz_to_cam.sum(axis=1, keepdims=True)
xyz_to_srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.969266, 1.8760108, 0.041556],
                        [0.0556434, -0.2040259, 1.0572252]])
matrix = xyz_to_srgb @ np.linalg.inv(xyz_to_cam)
gains = 1 / np.asarray(m["AsShotNeutral"])
gains /= gains[1]
raw = load_raw(root / "0001_GT_RAW_010.MAT", "x")
# Fixed even-offset crop preserves the CFA phase. Keep orientation in sensor order.
crop = raw[1000:1512, 2000:2512]
out = Path("data/processed/sidd-reference")
out.mkdir(parents=True, exist_ok=True)
np.save(out / "clean-crop.npy", crop)
metadata = {"normalized": True, "bayer_pattern": pattern,
            "white_balance": gains.tolist(), "camera_to_linear_srgb": matrix.tolist(),
            "source": "AbdoKamel/simple-camera-pipeline/data/0001_GT_RAW_010.MAT",
            "source_shape": list(raw.shape), "crop_yxhw": [1000, 2000, 512, 512],
            "orientation": "sensor order; EXIF orientation not applied",
            "sample_kind": "real clean sensor sample, not a paired benchmark",
            "colour_transform": "Approximate row-normalized ColorMatrix2 inverse to D65 linear sRGB; no dual-illuminant interpolation"}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Prepared {crop.shape}, {pattern}, normalized range {crop.min():.5f}..{crop.max():.5f}")
