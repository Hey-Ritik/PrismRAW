"""Create small real-scene UI thumbnails from the attributed clean demo RAW."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from smartphone_isp.pipelines.classical import process

out = Path("app/static/samples")
out.mkdir(parents=True, exist_ok=True)
for metadata_path in Path("data/processed/demo").glob("*.json"):
    raw = np.load(metadata_path.with_name(metadata_path.stem + "-clean.npy"), allow_pickle=False)
    metadata = json.loads(metadata_path.read_text())
    rgb, _, _ = process(raw, metadata)
    image = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
    image.thumbnail((240, 180))
    image.save(out / f"{metadata_path.stem}.jpg", quality=85)
print("Prepared thumbnails from real SIDD-CC demo captures; attribution in THIRD_PARTY_NOTICES.md")
