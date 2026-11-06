"""Create a local release bundle with source, reports, weights, and small demo assets."""

import hashlib
import json
from pathlib import Path
import zipfile

root = Path.cwd()
files = []
for folder in ("src", "app", "configs", "docs", "reports", "scripts", "tests", "data/processed/demo"):
    files.extend(path for path in (root / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
for name in ("README.md", "pyproject.toml", "Dockerfile", ".dockerignore", ".gitignore", ".env.example", "run_local.ps1", "THIRD_PARTY_NOTICES.md", "render.yaml"):
    if (root / name).exists():
        files.append(root / name)
for name in ("final-hybrid.pt", "final-learned.pt", "final-hybrid-fp16.pt"):
    path = root / "checkpoints" / name
    if not path.exists():
        raise ValueError(f"Missing final checkpoint: {name}")
    files.append(path)
output = root / "outputs/prismraw-release.zip"
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(root))
    manifest = {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    archive.writestr("RELEASE_SHA256.json", json.dumps(manifest, indent=2))
print(f"Packaged {len(files)} files into {output} ({output.stat().st_size/1e6:.2f} MB)")
