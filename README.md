# Efficient Smartphone ISP — P2

Hybrid smartphone RAW-to-RGB reconstruction combining classical image processing and a lightweight learned model.

## Current status

Classical array-based RAW-to-RGB inference is implemented: explicit normalization, Bayer white balance, bilinear demosaicing, colour correction, bilateral denoising, adaptive exposure/tone mapping, sRGB encoding, and sharpening. NPY/NPZ and MATLAB RAW arrays are supported with explicit metadata; direct DNG decoding is pending. A compact conditioned residual network has completed a three-epoch CPU pilot with scene-separated data. Checkpoint-based hybrid inference and a limited PSNR check work. The pilot hybrid underperforms the classical baseline. Full training, learned-only baseline, complete evaluation, optimized export, and hosted UI are pending.

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

`python scripts/download_sidd_subset.py --scenes 3` downloads one noisy/clean RAW pair per physical scene from a pinned revision of the [SIDD-CC research derivative](https://huggingface.co/datasets/MhdMohammadi/SIDD-CC), with source URLs, size checks, LFS checksums where provided, and SHA-256 provenance in the ignored dataset folder. It reuses existing verified files. Use `--scenes 5` through `--scenes 10` to expand later. This is not the original SIDD Small release. The original download endpoint timed out during setup.

The clean sample used for initial ISP validation comes from the [SIDD authors' reference repository](https://github.com/AbdoKamel/simple-camera-pipeline/tree/master/data). `scripts/prepare_sidd_reference.py` prepares that sample if the two reference MAT files are present. Its single-illuminant colour transform is approximate, preserves sensor orientation, and is not an exact reproduction of SIDD's sRGB rendering.

After installing the `raw` optional dependencies, prepare and process the paired development subset:

```powershell
python scripts/download_sidd_subset.py
python scripts/prepare_sidd_subset.py
python -m smartphone_isp.cli classical --input data/processed/sidd-cc-subset/noisy.npy --metadata data/processed/sidd-cc-subset/metadata.json --output outputs/sidd-cc-classical
```

The legacy single-pair preparation command above remains a development-only example. For the scene-separated pilot, follow the commands below. No independently captured RGB reference is supplied here; targets are fixed renders of clean RAW. Independent registration and colour calibration remain pending.

## Prepare, train, and inspect the CPU pilot

Install the `raw` and `train` optional dependencies first. Run from the repository root:

```powershell
$env:PYTHONPATH = 'src'
python scripts/download_sidd_subset.py --scenes 3
python scripts/prepare_training_data.py
python -m smartphone_isp.training.train --config configs/pilot.yaml
python -m smartphone_isp.evaluation.pilot
python -m unittest discover -s tests -v
```

The default prepares 24 deterministic 128x128 patches: eight from each of three physical scenes. Assignments are saved before extraction; patches from one scene cannot enter multiple splits. The pilot has one training scene, one validation scene, and one held-out test scene. Crops have even Bayer offsets and use a four-pixel processing border, removed before training. No augmentation is currently applied.

Inputs and targets use the same approximate camera transform in linear RGB. Display exposure is selected using the noisy input and reused for the target. Training uses L1 plus gradient consistency; the checkpoint is selected by validation linear-RGB L1 among trained epochs. The test split is not loaded during training. A manifest checksum guards evaluation against mismatched training data.

Artifacts: `checkpoints/pilot-best.pt`, `runs/pilot/training.json`, `outputs/pilot-evaluation/evaluation.json`, and comparison PNGs ordered **classical | hybrid | clean-RAW render**. The comparison targets and tiny scene count cannot establish real-world colour accuracy. See `reports/pilot_summary.md` for the observed underperformance.

Run hybrid inference using the locally trained checkpoint:

```powershell
python -m smartphone_isp.cli hybrid --input data/processed/sidd-cc-subset/noisy.npy --metadata data/processed/sidd-cc-subset/metadata.json --checkpoint checkpoints/pilot-best.pt --output outputs/hybrid-pilot
```

Hybrid inference still computes a redundant classical bilateral result during preparation; its current latency is not an optimized model benchmark. Model weights, data, and generated results are kept out of Git.

## Reproducibility

Split by scene before cropping patches. Record dataset versions, split manifests, configuration, random seed, source revision, checkpoint, output colour encoding, and benchmark hardware with every experiment. Never report placeholders as evaluation results.
