"""Validate tracked deployment inputs and the Render Blueprint without a cloud login."""

import hashlib
import json
from pathlib import Path
import yaml

root = Path.cwd()
blueprint = yaml.safe_load((root / "render.yaml").read_text())
service = blueprint["services"][0]
assert service["type"] == "web" and service["runtime"] == "docker"
assert service["plan"] == "free" and service["healthCheckPath"] == "/api/health"
environment = {entry["key"]: str(entry["value"]) for entry in service["envVars"]}
assert environment["PORT"] == "7860" and environment["PRISMRAW_HOSTED"] == "1"
assert environment["PRISMRAW_MAX_CROP"] == "256"
required = [Path("checkpoints") / name for name in ("final-hybrid.pt", "final-learned.pt", "final-hybrid-fp16.pt")]
for scene in ("001", "003", "004", "005"):
    required.extend(Path("data/processed/demo") / f"{scene}{suffix}" for suffix in (".json", "-noisy.npy", "-clean.npy"))
records = []
for path in required:
    if not path.is_file():
        raise ValueError(f"Missing deployment asset: {path}")
    records.append({"path": path.as_posix(), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
report = {"blueprint": "local structural checks passed; remote validation/build still required", "asset_count": len(records),
          "asset_bytes": sum(row["bytes"] for row in records), "assets": records,
          "environment": environment, "docker_build": "not run: Docker unavailable in current environment",
          "render_deployment": "not created; account authorization required"}
Path("reports/render_readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Render config and {len(records)} deployment assets verified ({report['asset_bytes']/1e6:.2f} MB)")
