# Efficient Smartphone ISP — P2

Hybrid smartphone RAW-to-RGB reconstruction combining classical image processing and a lightweight learned model.

## Current status

Initial project scaffold. Implemented: Bayer packing, normalization, camera-stage functions, a compact RGB residual network, and an environment-check CLI. Dataset integration, demosaicing, model training, inference orchestration, evaluation, and hosted UI are still to be implemented. No trained weights or measured results are included.

## Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
isp check
```

Install optional components as needed:

```powershell
python -m pip install -e ".[raw,train,eval,app]"
```

Select a PyTorch build appropriate for your hardware. Python 3.11 or 3.12 is a conservative environment choice if optional packages do not support your Python version.

## Layout

```text
app/                       Future hosted demo
configs/                   Experiment settings
data/raw/                  Local dataset files (ignored)
data/processed/            Cached patches (ignored)
data/splits/               Scene-level manifests
src/smartphone_isp/
  classical/               Sensor normalization and colour processing
  data/                    Bayer utilities and dataset contract
  models/                  Lightweight residual network
  pipelines/               Classical / learned / hybrid integration
  training/                Training implementation location
  evaluation/              Metrics and benchmark implementation location
  optimization/            Export and compression implementation location
checkpoints/               Local model weights (ignored)
outputs/                   Generated images and reports (ignored)
docs/                      Architecture, design, implementation roadmap
reports/                   Submission evaluation template
scripts/                   Dataset and experiment entry points
```

## First implementation milestone

Validate one paired RAW/RGB sample and metadata, then implement an end-to-end classical ISP. Follow `docs/implementation-plan.md`. Reference-based metrics must only be computed against aligned ground truth. Preview renders of RAW data are not the original sensor measurements.

## Reproducibility

Split by scene before cropping patches. Record dataset versions, split manifests, configuration, random seed, source revision, checkpoint, output colour encoding, and benchmark hardware with every experiment. Never report placeholders as evaluation results.
