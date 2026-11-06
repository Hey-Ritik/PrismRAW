# Local run and deployment packaging

## Local run (verified workflow)

From the project root, run `./run_local.ps1`. The server binds to **127.0.0.1:7860**, preloads local trained weights and pretrained LPIPS-Squeeze, then serves the camera lab. `--port 7861` selects a different port. Use Ctrl+C in the serving terminal to stop it.

Required model files: `checkpoints/final-hybrid.pt`, `final-learned.pt`, and `final-hybrid-fp16.pt`. Demo scenes live in `data/processed/demo/`. These binary assets are excluded from Git and included in the local release ZIP. The pretrained SqueezeNet backbone is cached under `.cache/torch`; the first setup needs a roughly 5 MB download if not cached. Runtime uses no cloud inference.

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
- Processing uses an even-offset central crop up to 512x512, not a resized sensor mosaic. The app serializes inference requests to bound CPU load.
- Metrics appear only with a matching clean RAW or aligned RGB reference. No automatic registration is claimed.
- Temporary uploads are removed after processing. Output folders are removed after one hour on the next inference request. Do not expose the local results directory as permanent public storage.
- Local-only UI says data stays on this machine. Update that wording before public server deployment because uploads would then be sent to that server.

## Release and submission

`python scripts/package_release.py` creates `outputs/prismraw-release.zip` with source, configs, docs, measured reports, small demo assets, final checkpoints, and SHA-256 hashes. It excludes full training data, credentials, caches and Git internals. Model weights can be distributed separately via a release artifact when publishing the repository.

Team T47 and USNs 1RUA24CSE0371, 1RUA24CSE0365 are recorded in `configs/team.json` and `reports/eval_P2_T47.md`. Public hosting and the actual demo/viva remain submission actions. Do not replace these with fabricated values or treat localhost as a live hosted submission URL.
