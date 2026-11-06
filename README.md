# Efficient Smartphone ISP — P2

Hybrid smartphone RAW-to-RGB reconstruction combining classical image processing and a lightweight learned model.

## Current status

Classical array-based RAW-to-RGB inference is implemented: explicit normalization, Bayer white balance, bilinear demosaicing, colour correction, bilateral denoising, adaptive exposure/tone mapping, sRGB encoding, and sharpening. NPY/NPZ and MATLAB RAW arrays are supported with explicit metadata; direct DNG decoding is pending. A compact RGB residual network is defined but untrained. Training, reference-quality evaluation, optimized export, and hosted UI are pending.

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

The classical pipeline has been smoke-tested on synthetic input and a crop of the authors' real clean SIDD reference RAW. This does not establish reconstruction quality. Reference-based metrics must only be computed against aligned ground truth. Preview renders of RAW data are not the original sensor measurements.

## Run the classical pipeline

Without an editable install, set the source path in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
python scripts/create_smoke_input.py
python -m smartphone_isp.cli classical --input data/processed/synthetic-smoke.npy --metadata configs/synthetic-metadata.json --output outputs/classical-smoke
python -m unittest discover -s tests -v
```

Outputs: `output.png`, floating-point `intermediates.npz`, and `processing.json` with selected settings and array-processing latency. Synthetic input is a plumbing check, not a benchmark. Use `--fixed` for non-adaptive parameters. `--key x` selects an array from a MAT/NPZ container.

Metadata JSON must explicitly provide `normalized`, `bayer_pattern`, `white_balance`, and `camera_to_linear_srgb`. Unnormalized arrays additionally require scalar `black_level` and `white_level`. Identity matrices in the synthetic example are not real camera calibration.

## Dataset acquisition

`python scripts/download_sidd_subset.py` downloads one noisy/clean RAW pair and metadata from the [SIDD-CC research derivative](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC), with source URLs and SHA-256 hashes in the ignored dataset folder. This is not the original SIDD Small release and is insufficient for training or evaluation claims. The original download endpoint timed out during setup.

The clean sample used for initial ISP validation comes from the [SIDD authors' reference repository](https://github.com/AbdoKamel/simple-camera-pipeline/tree/master/data). `scripts/prepare_sidd_reference.py` prepares that sample if the two reference MAT files are present. Its single-illuminant colour transform is approximate, preserves sensor orientation, and is not an exact reproduction of SIDD's sRGB rendering.

After installing the `raw` optional dependencies, prepare and process the paired development subset:

```powershell
python scripts/download_sidd_subset.py
python scripts/prepare_sidd_subset.py
python -m smartphone_isp.cli classical --input data/processed/sidd-cc-subset/noisy.npy --metadata data/processed/sidd-cc-subset/metadata.json --output outputs/sidd-cc-classical
```

The prepared subset contains noisy/clean **RAW** crops, not an independently calibrated RGB reference. Its manifest is explicitly marked `development`. Scene expansion, registration checks, and target-rendering validation are required before evaluation or training.

## Reproducibility

Split by scene before cropping patches. Record dataset versions, split manifests, configuration, random seed, source revision, checkpoint, output colour encoding, and benchmark hardware with every experiment. Never report placeholders as evaluation results.
