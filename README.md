# PrismRAW

The studio includes backend wake-up/retry handling, per-capture quality/latency highlights, and measured failure examples. Three learned-model improvement trials were evaluated; none passed the quality promotion gate, so deployed weights remain unchanged. See [the measured experiment notes](docs/model-improvement.md).

Current structure: `frontend/` contains the static website and Vercel configuration; `backend/` contains Python source, scripts, tests, models, configs, reports and datasets. Set **Vercel Root Directory to frontend**. Render uses the repository-root Blueprint with Docker context `./backend` and Dockerfile `./backend/Dockerfile`. Run Python/training/evaluation commands below from **backend/**. The root `run_local.ps1` still launches both locally. Generated release ZIPs remain under root `outputs/`.


For **Vercel frontend + Render backend**, follow [the split deployment guide](docs/split-deployment.md). The static Vercel build uses `PUBLIC_API_BASE_URL`; Render uses `PRISMRAW_API_ONLY=1` and the exact `PRISMRAW_FRONTEND_ORIGINS` allowlist. Local serving still works without these overrides.

An efficient hybrid smartphone ISP for Computer Vision Hackathon P2. Converts Bayer sensor data into RGB with classical, learned-only, hybrid, and reduced-precision hybrid paths.

## Run locally

The prepared workspace contains the environment, final models and demo crops:

```powershell
.\run_local.ps1
```

Open **http://127.0.0.1:7860** after startup finishes warming the models. Stop with Ctrl+C. `./run_local.ps1 --port 7861` selects another port.

The responsive local app includes real sample scenes, DNG/array uploads, four-way comparison, reference metrics, stage previews, zoom, adaptive controls, evaluation results and downloadable images/reports. Processing preserves Bayer phase using an even-offset central crop up to 512x512. Classical remains the reference baseline; neural quality varies by scene.

The studio UI adds real-scene thumbnail selection, a large classical-versus-selected split viewer, keyboard-accessible comparison slider, single-image mode, compact pipeline switching, and selected-image quality/time summaries. Full measurements and stage details expand on demand. DNG selection hides the unnecessary metadata sidecar field; array selection requires it. RAW files can be selected or dropped into the upload area. Desktop/mobile interaction checks are available in `scripts/check_ui.py` and `reports/ui_v2_verification.json`.

## Measured final result

The hybrid scores **39.2312 dB PSNR** against **38.4589 dB** for classical on 64 patches from fresh scenes 009/010. SSIM improves from 0.8867 to 0.9027, Delta E 00 decreases from 2.7438 to 2.5231, and LPIPS-Squeeze decreases from 0.1042 to 0.0936. These are limited local research results against approximate clean-RAW renders, not official benchmark rankings or absolute colour calibration.

FP16 reduces the hybrid checkpoint from 483,245 to 245,497 bytes and preserves measured quality closely. It can be slower on this CPU. See `reports/eval_P2_T47.md`, `reports/final_results.json`, `reports/benchmark.json`, and `reports/optimization.json` for complete evidence and runtime measurements. Learned-only still oversmooths and makes colour errors despite sometimes high SSIM.

## Fresh setup

Use Python 3.11+; the executed Windows environment uses Python 3.13, PyTorch 2.7.1 CPU and torchvision 0.22.1. Select the correct PyTorch build for your device, then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[raw,train,eval,app]"
$env:PYTHONPATH = 'src'
$env:TORCH_HOME = Join-Path (Get-Location) '.cache/torch'
isp check
```

The three final model checkpoints and small attributed demo crops are included in Git so Docker/Render can build directly from a fresh checkout. Full training data, intermediate checkpoints and generated outputs remain Git-ignored. The release ZIP also includes these deployment assets. LPIPS downloads a roughly 5 MB pretrained SqueezeNet backbone on first setup if uncached. The local serving process needs no cloud inference.

## Reproduce data and training

```powershell
$env:PYTHONPATH = 'src'
python scripts/download_sidd_subset.py --scenes 10 --diverse-cameras
python scripts/check_registration.py
python scripts/prepare_training_data.py --output data/processed/final --existing-splits configs/final-splits.json --input-context 4 --patches-per-scene 32
python scripts/prepare_demo_assets.py
python -m smartphone_isp.training.train --config configs/final-hybrid.yaml
python -m smartphone_isp.training.train --config configs/final-learned.yaml
python -m smartphone_isp.evaluation.suite --select-blend
python -m smartphone_isp.evaluation.suite --split validation
python -m smartphone_isp.evaluation.suite --split test
python -m smartphone_isp.evaluation.benchmark
python scripts/write_submission_report.py
python scripts/package_release.py
```

The downloader pins SIDD-CC revision `5eb2ab6375d7f99f36b59bb5aacb46af18409412`, verifies sizes/checksums, and records provenance. Six training scenes (003–008), two validation scenes (001/002), and two fresh final scenes (009/010) yield 192/64/64 deterministic 128x128 patches across five phone camera types. Old inspected pilot scene 002 is development validation only. Never tune using the fresh final results.

Every method receives the same four-pixel RAW border. Hybrid input is bilateral-denoised linear RGB; learned-only input is packed Bayer plus camera gains, colour matrix and CFA phase. Models use L1 plus gradient consistency and 30 CPU epochs. Validation chooses checkpoints and hybrid residual blend. Optional CFA-safe augmentation is available with `augment: true`; it was disabled in these recorded final runs. Colour profiles are approximate, and coarse alignment checks do not prove local/subpixel registration.

## Metrics and ablations

PSNR, SSIM, CIEDE2000 (Lab D65), LPIPS v0.1 with pretrained SqueezeNet; parameter count, checkpoint size, warmed median/p95 CPU array-processing latency, and sampled process RSS. Reports include per-scene metrics, learned-refinement removal (classical), adaptation removal, FP16 comparison, and prespecified dark/noisy, heavy-noise and colour-cast stress tests. Metrics can disagree; no single score proves colour or texture quality.

## Tests and application verification

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
```

Tests cover sensor normalization, CFA patterns and augmentation, MAT/HDF5 axes, scene leakage, context cropping, gradient updates, versioned checkpoints, FP16 inference, metric conventions and bad API inputs. `scripts/check_browser.py` optionally uses Playwright with installed Edge to check desktop/mobile rendering, sample processing, metric rows, JavaScript errors and ZIP downloads. Browser screenshots are under `outputs/browser-check`.

## Input contract

DNG uses visible Bayer measurements, per-channel black/white normalization and explicit camera/profile metadata. No hidden `rawpy.postprocess()` ISP is used. Arrays accept NPY, NPZ, or MAT (including MATLAB v7.3) plus JSON with `normalized`, `bayer_pattern`, `white_balance` RGB gains, and `camera_to_linear_srgb` (3x3 column-vector convention). Unnormalized arrays also need scalar `black_level` and `white_level`. Use `--key` for NPZ/MAT variables. SIDD MAT RAW is already normalized; do not subtract black twice.

References are optional matching clean RAW arrays or aligned RGB PNG/JPEG. No quality scores are invented for unreferenced uploads. Unsupported files, missing metadata and incompatible dimensions produce errors. Bayer-only support and approximate colour calibration limit generalization to arbitrary cameras.

## Layout

```text
frontend/
  index.html           camera studio
  static/              CSS, JavaScript, thumbnails and local API configuration
  build.mjs            static build with the Render API origin
  vercel.json          Vercel deployment settings
backend/
  src/smartphone_isp/   ISP pipelines, models, training, evaluation and FastAPI
  scripts/             serving, data preparation, checks and release packaging
  tests/               pipeline, numerical, API and CORS regression tests
  configs/             training recipes and scene policies
  checkpoints/         final deployment weights; intermediate weights ignored
  data/                demo crops tracked; training data ignored
  reports/             measured results and submission report
  Dockerfile           Render backend image
  pyproject.toml       Python dependencies
  outputs/             new API downloads (ignored)
docs/                  architecture, deployment and demo guides
render.yaml            Render Blueprint using backend Docker context
run_local.ps1          local combined launcher
outputs/               release ZIP and earlier local artifacts (ignored)
```

## Delivery and remaining submission actions

Docker packaging is supplied; see `docs/deployment.md`. The current verification is local, not a claimed public deployment or smartphone benchmark. Docker build has not been verified in this environment. A public host needs adequate resources; the hosted profile displays server-processing privacy wording. See `THIRD_PARTY_NOTICES.md` for data and model attribution.

Team **T47**: **1RUA24CSE0371, 1RUA24CSE0365**. Identity is saved in `configs/team.json` and used by the report generator. The submission report is `eval_P2_T47.md`, with comma-separated USNs on its first line. A public hosted URL and actual team demo/viva remain submission actions. Historical pilot reports retain earlier results; the final report supersedes their current-quality conclusions.


## Deploy on Render Free

[Deploy PrismRAW on Render](https://render.com/deploy?repo=https://github.com/Hey-Ritik/PrismRAW)

The committed `render.yaml` selects a Docker web service on the Free plan with health check `/api/health`. It sets `PORT=7860`, `PRISMRAW_HOSTED=1`, `PRISMRAW_MAX_CROP=256` and `PRISMRAW_TORCH_THREADS=1`. No API keys or database credentials are required. Sign in to Render, connect the repository if requested, review the Free service, and deploy. Future updates use Manual Deploy because auto-deploy is deliberately off for this quick-deploy Blueprint.

The hosted profile processes 256x256 central crops rather than the local default 512x512. DNG calibration crops before allocating float/per-pixel calibration buffers. References are cropped to the same sensor region; no geometric registration or image resampling is invented. The UI distinguishes local from hosted processing. This reduces memory pressure but does not guarantee that every upload fits the free instance or that local timings match Render's limited CPU.

FP16 loads only when selected and is released after inference. The app saves intermediate previews early and scores/saves one reconstruction at a time to shorten array lifetimes. Live LPIPS scoring remains enabled, with unchanged crops and model weights. A four-method 64px regression check reproduced all images and metrics exactly. The measured 256px sample/array/DNG peak changed only slightly (494.1 to 492.2 MiB on Windows); no substantial RAM saving or Render memory guarantee is claimed. Details are in `reports/memory_optimization.json`.

Run `python scripts/check_deployment.py` to check required files/configuration, and `python scripts/check_render_profile.py` for a localhost-only hosted-profile integration check. The latter creates and stops its own temporary server. Docker is unavailable in the development environment, so the actual Linux image build and hosted resource fit must be verified from Render logs. No public Render service has been created merely by committing this configuration.

See `docs/deployment.md` for the complete settings and troubleshooting guide. Free services sleep after inactivity, and generated files are ephemeral. A keep-alive request does not fix memory limits or guarantee uptime.
