"""Create an artificial Bayer array for plumbing checks, NOT quality evaluation."""

from pathlib import Path
import numpy as np

target = Path("data/processed/synthetic-smoke.npy")
target.parent.mkdir(parents=True, exist_ok=True)
y, x = np.mgrid[:256, :256].astype(np.float32) / 255
rgb = np.stack([0.05 + 0.65*x, 0.05 + 0.65*y, 0.15 + 0.3*(x > y)], axis=-1)
mosaic = np.empty((256, 256), dtype=np.float32)
for i, color in enumerate("RGGB"):
    mosaic[i//2::2, i%2::2] = rgb[i//2::2, i%2::2, "RGB".index(color)]
mosaic += np.random.default_rng(42).normal(0, 0.01, mosaic.shape).astype(np.float32)
np.save(target, np.clip(mosaic, 0, 1))
print(f"Created {target}: synthetic input, not a real sensor capture")
