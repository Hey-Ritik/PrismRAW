"""Download one SIDD-CC pair, with provenance and hashes, without executing remote code.

SIDD-CC is a research derivative, not the original SIDD Small release.
See https://huggingface.co/datasets/MhdMohammadi/SIDD-CC for terms and citations.
"""

import hashlib
import json
from pathlib import Path
import urllib.request

scene = "0001_001_S6_00100_00060_3200_L"
base = "https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/resolve/main/"
folder = Path("data/raw/sidd-cc-subset") / scene
folder.mkdir(parents=True, exist_ok=True)
records = []
for kind in ("NOISY", "GT", "METADATA"):
    name = f"0001_{kind}_RAW_034.MAT"
    url = base + f"Raw/train/Data/{scene}/{name}"
    target = folder / name
    if not target.exists():
        temporary = target.with_suffix(".part")
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(target)
    records.append({"file": name, "url": url, "bytes": target.stat().st_size,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    print(f"Ready: {name} ({target.stat().st_size:,} bytes)", flush=True)
(folder.parent / "provenance.json").write_text(json.dumps({
    "dataset": "MhdMohammadi/SIDD-CC", "scene": scene,
    "upstream": "https://abdokamel.github.io/sidd/",
    "terms": "https://huggingface.co/datasets/MhdMohammadi/SIDD-CC/blob/main/README.md",
    "purpose": "One-pair loader development; insufficient for training or generalization claims",
    "files": records}, indent=2), encoding="utf-8")
