"""Validate the downloaded SIDD-CC pair and prepare aligned development crops.

No train/test split or benchmark score is claimed for a single scene.
Colour rendering uses an approximate single-illuminant matrix convention.
"""

import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

scene = "0001_001_S6_00100_00060_3200_L"
root = Path("data/raw/sidd-cc-subset") / scene
noisy = loadmat(root / "0001_NOISY_RAW_034.MAT")["data"]
clean = loadmat(root / "0001_GT_RAW_034.MAT")["data"]
meta = loadmat(root / "0001_METADATA_RAW_034.MAT", simplify_cells=True)
if noisy.shape != clean.shape or noisy.ndim != 2:
    raise ValueError("Expected matching 2D Bayer arrays")
if not np.isfinite(noisy).all() or not np.isfinite(clean).all():
    raise ValueError("Nonfinite sensor data")
if min(noisy.min(), clean.min()) < -0.1 or max(noisy.max(), clean.max()) > 1.1:
    raise ValueError("Confirm dataset normalization before processing")
matrix = np.asarray(meta["ColorMatrix2"]).reshape(3, 3)
matrix /= matrix.sum(axis=1, keepdims=True)
xyz_to_srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                       [-0.969266, 1.8760108, 0.041556],
                       [0.0556434, -0.2040259, 1.0572252]])
gains = 1 / np.asarray(meta["AsShotNeutral"])
gains /= gains[1]
out = Path("data/processed/sidd-cc-subset")
out.mkdir(parents=True, exist_ok=True)
np.save(out / "noisy.npy", noisy[1000:1512, 2000:2512])
np.save(out / "clean.npy", clean[1000:1512, 2000:2512])
metadata = {"normalized": True, "bayer_pattern": str(meta["cfa_str"]).upper(),
            "white_balance": gains.tolist(),
            "camera_to_linear_srgb": (xyz_to_srgb @ np.linalg.inv(matrix)).tolist(),
            "dataset": "SIDD-CC research derivative", "scene_id": "001",
            "crop_yxhw": [1000, 2000, 512, 512], "orientation": "sensor order",
            "colour_transform": "Approximate single-illuminant rendering; not calibrated SIDD RGB ground truth"}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
report = {"dataset": "SIDD-CC", "full_shape": list(noisy.shape),
          "noisy_range": [float(noisy.min()), float(noisy.max())],
          "clean_range": [float(clean.min()), float(clean.max())],
          "paired_array_shape_check": "passed", "finite_values_check": "passed",
          "registration": "Correspondence supplied by dataset; independent spatial verification pending",
          "usage": "development only; one scene cannot establish generalization"}
(out / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
manifest = Path("data/splits/development.csv")
manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(["sample_id", "scene_id", "camera_id", "split", "raw_path", "clean_raw_path", "metadata_path"])
    writer.writerow(["sidd-cc-0001-034", "001", "S6", "development",
                     (out / "noisy.npy").as_posix(), (out / "clean.npy").as_posix(),
                     (out / "metadata.json").as_posix()])
print(json.dumps(report, indent=2))
