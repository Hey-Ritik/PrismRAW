# Expanded three-way pilot — 6 November 2026

Engineering results only; not official SIDD benchmark scores or the final P2 evaluation.

## Data and protocol

SIDD-CC revision `5eb2ab6375d7f99f36b59bb5aacb46af18409412`. One capture per physical scene; deterministic 128x128 crops with seed 42.

| Split | Scenes | Cameras | Patches |
|---|---|---|---:|
| Train | 003, 004, 005 | Nexus 6 | 24 |
| Validation | 001 | Samsung S6 | 8 |
| Exploratory test | 002 | Google Pixel | 8 |

Original holdout scene assignments are frozen. All 16 old validation/test source, target, classical-output, display-target, and conditioning arrays were verified identical after expansion. Independent RAW-pair registration checks remain pending. This test scene was already inspected in the earlier pilot; it is not a fresh final holdout.

Targets are approximate fixed renders of clean RAW, not captured RGB ground truth. For the comparison, all methods share noisy-input display exposure and sharpening; target rendering uses the same exposure without sharpening. Metrics use full stored patches. Classical interpolation has four pixels of RAW context around each patch; the learned-only model currently receives the central RAW patch, a boundary-context limitation to address before final benchmarking.

## Models and training

Both runs: PyTorch 2.7.1 CPU, two threads, seed 42, 20 epochs, Adam (learning rate 0.0002), batch size 2, linear-RGB L1 plus gradient consistency. Checkpoints selected by validation L1, without loading test samples during training.

| Model | Parameters | Selected epoch | Initial validation L1 | Selected validation L1 |
|---|---:|---:|---:|---:|
| Hybrid residual refinement | 118,595 | 20 | 0.00546048 | 0.00537623 |
| Learned-only Bayer-to-RGB | 120,908 | 17 | 0.12229781 | 0.03287937 |

Learned-only input: normalized Bayer packed into semantic R/G/G/B channels plus 16 constant camera features (RGB white-balance gains, nine colour-matrix entries, four CFA-phase indicators). A compact U-Net and pixel-shuffle head reconstruct full-resolution linear RGB. No classical white-balance multiplication, demosaicing, matrix multiplication, or denoising precedes that network. Both learned models use common display rendering after reconstruction for this comparison.

## Exploratory comparison

| Method | Mean per-patch sRGB PSNR, dB |
|---|---:|
| Classical ISP | 42.7959 |
| Learned-only ISP | 11.9375 |
| Hybrid ISP | 39.4078 |

Arithmetic mean of eight per-patch PSNR values against the approximate clean-RAW render, float sRGB [0,1]. Classical remains strongest. These values must not be presented as official benchmark results or evidence of learned superiority.

## Observed failures and limitations

- Learned-only comparison images show pronounced checkerboard artifacts and colour errors. Uneven subpixel reconstruction is a plausible architectural contributor; the causal contribution has not been isolated by ablation.
- Twenty epochs on only three training captures do not establish convergence or robust reconstruction. Training-camera coverage is limited to Nexus 6; camera changes and scene changes are confounded in the holdouts.
- Hybrid validation improves slightly over identity refinement but remains behind classical denoising on the exploratory test. No broad quality-improvement claim is supported.
- Approximate camera colour transforms, unverified registration, limited crop coverage, and boundary-context differences constrain interpretation.
- SSIM, Delta E, LPIPS, device benchmarks, model compression, and a fresh final holdout remain pending.

## Next work

Use validation data to investigate reconstruction artifacts and improve the architecture. Expand camera/scene coverage and patch diversity, validate alignment and target rendering, equalize boundary context, and establish a fresh final evaluation protocol. Keep classical as the default quality baseline. Avoid tuning against the already-inspected exploratory test.

## Reproduction

Use the expanded comparison commands in the README and the committed split policy/configurations. Model weights, dataset files, and generated comparison images remain local and Git-ignored.
