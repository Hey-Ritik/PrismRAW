# Local run and deployment packaging

## Local run (verified workflow)

From the project root, run `./run_local.ps1`. The server binds to **127.0.0.1:7860**, preloads local trained weights and pretrained LPIPS-Squeeze, then serves the camera lab. `--port 7861` selects a different port. Use Ctrl+C in the serving terminal to stop it.

Required model files: `checkpoints/final-hybrid.pt`, `final-learned.pt`, and `final-hybrid-fp16.pt`. Demo scenes live in `data/processed/demo/`. These final model files and small demo crops are now tracked in Git and also included in the local release ZIP; full training data and non-final weights remain excluded. The pretrained SqueezeNet backbone is cached under `.cache/torch`; the first setup needs a roughly 5 MB download if not cached. Runtime uses no cloud inference.

For a fresh environment: create `.venv`, activate it, install the appropriate PyTorch/torchvision CPU or GPU build, then `pip install -e ".[raw,train,eval,app]"`. The executed Windows runtime used Python 3.13 and PyTorch 2.7.1 CPU with torchvision 0.22.1. Full training uses optional `train`; the demo needs RAW, evaluation, app, and PyTorch dependencies.

## Container packaging

The Dockerfile installs a matching CPU PyTorch/torchvision build, bundles only final models and small demo assets, warms the perceptual backbone at build time, runs as an unprivileged user, and supplies a health check.

```text
docker build -t prismraw .
docker run --rm -p 127.0.0.1:7860:7860 prismraw
```

Docker build/runtime is a separate environment check; local browser verification does not establish Docker compatibility. A public host needs adequate CPU/RAM, HTTPS, request limits, and persistent access to the bundled weights. No public URL is published by running localhost.

## Data and service behavior

- RAW upload formats: 2x2 Bayer DNG, NPY, NPZ, MAT. Array uploads require camera JSON; DNG uses explicit raw sensor extraction and profile metadata rather than a hidden ISP postprocess call.
- Upload size: 64 MB per sensor/reference file and 64 KB metadata. Maximum decoded sensor shape: 24 megapixels. NPZ total uncompressed size is capped at 192 MB. Native parsers still consume resources; use a hardened isolated upload service for public untrusted workloads.
- Processing uses an even-offset central crop up to 512x512 locally, or 256x256 with the Render Blueprint, not a resized sensor mosaic. The app serializes inference requests to bound CPU load.
- Metrics appear only with a matching clean RAW or aligned RGB reference. No automatic registration is claimed.
- Temporary uploads are removed after processing. Output folders are removed after one hour on the next inference request. Do not expose the local results directory as permanent public storage.
- The health endpoint exposes hosted/local mode and crop size. The UI says uploads are processed on the hosting server in hosted mode, and reports the actual configured crop limit. No claim that hosted uploads stay on the visitor device is made.

## Release and submission

`python scripts/package_release.py` creates `outputs/prismraw-release.zip` with source, configs, docs, measured reports, small demo assets, final checkpoints, and SHA-256 hashes. It excludes full training data, credentials, caches and Git internals. Model weights can be distributed separately via a release artifact when publishing the repository.

Team T47 and USNs 1RUA24CSE0371, 1RUA24CSE0365 are recorded in `configs/team.json` and `reports/eval_P2_T47.md`. Public hosting and the actual demo/viva remain submission actions. Do not replace these with fabricated values or treat localhost as a live hosted submission URL.


## Render Free: exact deployment steps

1. Open https://render.com/deploy?repo=https://github.com/Hey-Ritik/PrismRAW and sign in to your Render account.
2. Connect the GitHub repository if Render requests access. The service configuration comes from the root `render.yaml`.
3. Confirm service `prismraw-t47`, runtime Docker, branch `main`, plan **Free**, Dockerfile `./Dockerfile`, and health path `/api/health`.
4. Confirm the four public environment values below. No secret keys, database URL, custom build command or Docker command is required.
5. Create/deploy the service. Inspect the build and runtime logs. When it is Live, open its generated URL and verify `/api/health`, a sample scene, an unreferenced array upload and a supported DNG upload.
6. Keep FP16 comparison off for routine CPU use. After later repository updates, use Manual Deploy → Deploy latest commit.

| Variable | Render value | Purpose |
|---|---|---|
| PORT | 7860 | Startup and Docker health check use the same port |
| PRISMRAW_HOSTED | 1 | Accurate server-processing wording in the browser |
| PRISMRAW_MAX_CROP | 256 | Bounded sensor crops and smaller feature buffers |
| PRISMRAW_TORCH_THREADS | 1 | Conservative CPU thread use |

`PYTHONPATH`, `TORCH_HOME`, and `PYTHONUNBUFFERED` are set by the Dockerfile. `.env.example` documents process variables; it is not automatically loaded. Local defaults remain 512px and two threads without hosted overrides.

The final three checkpoints and all twelve sample array/metadata files are deliberately tracked. `reports/render_readiness.json` records their sizes and SHA-256 hashes. Source/metric dependencies and the pretrained SqueezeNet backbone are installed at image build time. Full datasets are not needed to serve the app.

## Limits and troubleshooting

- Render Free currently has 512 MB RAM and a limited CPU share. Smaller crops reduce inference allocation, and DNG is cropped before float normalization; native decoding still consumes memory. Local Windows process-tree RSS is not a Linux cgroup memory guarantee. If logs show an out-of-memory exit, inspect the failing input and reduce workload or use a larger instance; health pings cannot correct resource exhaustion.
- The actual Docker build is not verified locally because Docker is unavailable. A successful local hosted-profile smoke test does not prove a successful Render build. Read Render's build logs for dependency/network/platform errors.
- If Render reports no listening port, check that `PORT` matches the startup logs. The Docker CMD binds `0.0.0.0`; do not override it with a localhost-only command.
- If model files are missing at build time, verify deployment uses the new commit containing `checkpoints/final-*.pt` and `data/processed/demo/`.
- Free services sleep after 15 idle minutes and can restart. Results are temporary and are lost on restart/redeploy. Download results promptly. Automatic monitoring does not establish an uptime guarantee.
- A live public URL still requires the user's Render account and successful service creation. The repo/configuration alone is not a hosted deployment.

Official references: https://render.com/docs/blueprint-spec, https://render.com/docs/docker, https://render.com/docs/web-services, https://render.com/docs/free.
