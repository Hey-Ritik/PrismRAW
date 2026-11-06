# Shared-context decoder comparison — 6 November 2026

Validation-only engineering experiment. No new exploratory-test scores were computed for this change, and no final benchmark claim is made.

## Problem and changes

The original learned-only output showed pronounced 2x2 periodic artifacts. The original comparison also gave classical interpolation four RAW pixels of boundary context that the neural inputs lacked.

Schema 3 now gives every method the same 136x136 RAW crop for a central 128x128 target. Hybrid receives 136x136 transformed linear RGB; learned-only receives 68x68 packed Bayer. Both reconstruct 136x136 RGB, with four output pixels removed per edge for loss/evaluation. All 40 central sources, Bayer phases, targets, conditioning values, and classical outputs were checked against the previous prepared data and remained identical.

The v2 decoder replaces the learned-only pixel-shuffle output with bilinear resizing of learned features followed by two shared spatial convolutions. It does not perform classical demosaicing. V1 remains available for comparison and checkpoint compatibility.

## Controlled settings

Pinned five-scene SIDD-CC pilot, seed 42. Scenes 003/004/005 are training; 001 is validation; 002 remains the already-inspected exploratory test. Eight patches per scene. Both decoder runs use the same backbone structure, training inputs, objective, Adam learning rate 0.0002, batch size 2, CPU threads 2, and 20 epochs. Head parameter counts and initialization differ, so this is an output-head comparison, not an isolated proof of every cause of artifact reduction.

Targets are approximate fixed renders of clean RAW. Independent registration and colour calibration are still pending. All training scenes are Nexus 6; validation is Samsung S6, so camera coverage remains limited. Validation results are used for development and cannot establish final generalization.

## Results on eight validation patches

| Method | Selected epoch | Parameters | Selected linear-RGB validation L1 | Mean display PSNR, dB |
|---|---:|---:|---:|---:|
| Classical ISP | N/A | N/A | Not measured | 31.7973 |
| Hybrid residual | 20 | 118,595 | 0.00542766 | 27.1641 |
| Learned-only subpixel v1 | 17 | 120,908 | 0.03300908 | 15.6979 |
| Learned-only resize v2 | 19 | 123,459 | 0.02385094 | 19.3759 |

Checkpoints are selected by validation linear-RGB L1 among trained epochs. PSNR is an arithmetic mean of per-patch float-sRGB [0,1] scores with noisy-input display exposure shared across methods. These are validation scores, not the earlier exploratory-test scores.

## Periodic-bias diagnostic and visual inspection

The mean-channel standard deviation of the four 2x2-phase mean sRGB residuals was 0.07261084 for v1 and 0.00051951 for v2 (approximately 99.3% lower). This custom diagnostic measures systematic periodic bias; it is not a standard perceptual-quality metric and does not capture every artifact.

Validation previews show substantially less checkerboarding with v2. Fine text and edges remain too smooth, and colour/detail reconstruction is still poor relative to classical processing. V2 improves this narrow development check but does not solve the reconstruction problem.

## Verification

19 tests passed, including explicit context cropping/gradient placement, refusal of shape broadcasting, versioned v2 checkpoint reload, flat-feature periodic-bias regression, and all previous sensor/split/inference tests. Preparation consistency checks covered all 40 patches. Training and evaluation used validation for architecture comparison; the exploratory test scene was not rescored.

## Reproduction and next work

Run the shared-context commands in the README with the committed `context-*.yaml` configurations and frozen split policy. Data, checkpoints, logs, and previews remain Git-ignored.

Next: improve training-camera and texture coverage, inspect alignment, add full quality metrics, and investigate detail-preserving losses/architectures using validation. Keep classical as the current quality baseline and reserve a fresh final evaluation set before making improvement claims.
