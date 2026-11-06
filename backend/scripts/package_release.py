"""Bundle the separated app, final models, demo crops and measured reports."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[2]
files = []
for folder in ("backend/src", "backend/configs", "backend/docs", "backend/reports", "backend/scripts", "backend/tests", "backend/data/processed/demo", "frontend", "docs"):
    files.extend(path for path in (root / folder).rglob("*") if path.is_file() and not {"__pycache__", "dist", "node_modules"}.intersection(path.parts))
for name in ("README.md", ".gitignore", "run_local.ps1", "THIRD_PARTY_NOTICES.md", "render.yaml", "backend/README.md", "backend/pyproject.toml", "backend/Dockerfile", "backend/.dockerignore", "backend/.env.example", "backend/THIRD_PARTY_NOTICES.md"):
    if (root / name).exists(): files.append(root / name)
for name in ("final-hybrid.pt", "final-learned.pt", "final-hybrid-fp16.pt"):
    path = root / "backend/checkpoints" / name
    if not path.exists(): raise ValueError(f"Missing final checkpoint: {name}")
    files.append(path)
output = root / "outputs/prismraw-release.zip"
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files: archive.write(path, path.relative_to(root))
    manifest = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    archive.writestr("RELEASE_SHA256.json", json.dumps(manifest, indent=2))
print(f"Packaged {len(files)} files into {output} ({output.stat().st_size/1e6:.2f} MB)")
