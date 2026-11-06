# Three-scene CPU pilot — 6 November 2026

This is an engineering check, not the P2 submission evaluation or an official SIDD benchmark. The learned refinement has not improved quality.

## Data and rendering

- Dataset: MhdMohammadi/SIDD-CC, revision `5eb2ab6375d7f99f36b59bb5aacb46af18409412`.
- One noisy/clean RAW capture from each physical scene: 003 train (Nexus 6), 001 validation (Samsung S6), 002 test (Google Pixel).
- Eight deterministic 128x128 patches per scene, 24 total; seed 42. Scene assignments precede extraction.
- Targets: approximate single-illuminant rendering of clean RAW, not captured RGB ground truth. Exposure is selected from noisy input only and shared with the target. Independent registration validation is pending.
- Different cameras across splits confound scene and sensor changes. Eight patches from one image are not eight independent scenes.

## Training

PyTorch 2.7.1 CPU, two threads, residual U-Net (118,595 parameters), Adam, learning rate 0.0002, batch size 2, three epochs. Objective: linear-RGB L1 plus 0.05 times horizontal/vertical gradient L1. Best trained epoch selected by validation L1; test not accessed by training.

| Check | Linear-RGB validation L1 |
|---|---:|
| Initial identity refinement | 0.00546048 |
| Epoch 1 | 0.00552754 |
| Epoch 2 | 0.00551622 |
| Epoch 3, selected trained checkpoint | 0.00548725 |

Even the best trained checkpoint remains worse than the initial identity on validation. It is retained as a pilot artifact, not promoted as the preferred pipeline.

## Held-out pilot check

| Method | Mean per-patch sRGB PSNR, dB |
|---|---:|
| Classical bilateral ISP | 42.7959 |
| Trained hybrid pilot | 39.2200 |

PSNR uses float sRGB in [0,1], the complete stored patch, and an arithmetic mean over eight patches from one held-out scene. Classical denoising is the stronger baseline in this check. SSIM, Delta E, LPIPS, repeated latency measurements, and peak-memory measurement remain pending.

## Interpretation and next experiments

This run confirms that pairing, scene separation, optimization steps, checkpoint reload, and hybrid inference execute. One training image and three epochs do not establish a useful restoration model. The approximate colour transform and unverified alignment also limit interpretation. Preserve this test result; select future hyperparameters using training/validation only, and establish a fresh final holdout before making comparative claims.

Next: expand training scenes/cameras, inspect spatial correspondence, implement the learned-only RAW baseline, and train for meaningful convergence. Keep classical as the current quality baseline. Do not repeatedly tune against this pilot test scene.

## Reproduce

Follow the pilot commands in the README with `configs/pilot.yaml`. Generated manifests, checkpoints, logs, and images are local Git-ignored artifacts. Re-running preparation requires the same pinned dataset revision, seed, and crop settings.
