"""Coarse translational correspondence check, not a colour-calibration claim."""

import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from skimage.registration import phase_cross_correlation

index = json.loads(Path("data/raw/sidd-cc-subset/pilot-index.json").read_text())
rows = []
for record in index["records"]:
    arrays = [loadmat(record["files"][kind]["path"])["data"][::8, ::8].astype(np.float32) for kind in ("clean", "noisy")]
    shift, _, _ = phase_cross_correlation(arrays[0], arrays[1], upsample_factor=10, normalization=None)
    rows.append({"scene_id": record["scene_id"], "instance": record["instance"], "estimated_raw_shift_yx": (shift*8).tolist(),
                 "within_one_raw_pixel": bool(np.max(np.abs(shift*8)) <= 1)})
    print(rows[-1], flush=True)
Path("reports/registration.json").write_text(json.dumps({"method": "Same-CFA samples at stride 8, phase cross-correlation with 10x shift refinement. Checks translation only; no assertion about local warp, motion, absolute colour or subpixel registration.", "results": rows}, indent=2), encoding="utf-8")
