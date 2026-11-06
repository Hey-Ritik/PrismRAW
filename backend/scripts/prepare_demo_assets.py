"""Bundle small paired demonstration crops and metadata, with dataset attribution."""

import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from smartphone_isp.data.pilot import camera_metadata

index = json.loads(Path("data/raw/sidd-cc-subset/pilot-index.json").read_text())
out = Path("data/processed/demo")
out.mkdir(parents=True, exist_ok=True)
for record in index["records"]:
    if record["scene_id"] not in {"001", "003", "004", "005"}:
        continue
    noisy = loadmat(record["files"]["noisy"]["path"])["data"]
    clean = loadmat(record["files"]["clean"]["path"])["data"]
    metadata = camera_metadata(loadmat(record["files"]["metadata"]["path"], simplify_cells=True))
    h, w = noisy.shape
    y, x = (h // 2 - 192) // 2 * 2, (w // 2 - 192) // 2 * 2
    metadata.update(scene_id=record["scene_id"], dataset="SIDD-CC", revision=index["revision"],
                    crop_yxhw=[y, x, 384, 384], target="Matching clean RAW; rendered reference, not captured RGB")
    name = record["scene_id"]
    np.save(out / f"{name}-noisy.npy", noisy[y:y+384, x:x+384])
    np.save(out / f"{name}-clean.npy", clean[y:y+384, x:x+384])
    (out / f"{name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Demo {name}: {record['instance']}")
