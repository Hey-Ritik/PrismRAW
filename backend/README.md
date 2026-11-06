# PrismRAW backend

FastAPI, classical ISP, trained models, evaluation and RAW assets. Run Python commands from this folder with PYTHONPATH=src. Install using pip install -e ".[raw,eval,app]" after installing the CPU torch/torchvision versions in Dockerfile.

For local combined serving, run ../run_local.ps1 from PowerShell; it uses the root virtual environment and sibling frontend. Docker defaults to API-only mode and serves /api, /results and an API status at /.

Render Docker build context: backend. Dockerfile: backend/Dockerfile relative to repository root. See ../docs/split-deployment.md.
