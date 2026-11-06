# Design document

## Problem and scope
P2: reconstruct high-quality RGB from smartphone RAW while balancing quality, model size, memory, and latency.

## Input contract
Validated sensor data plus explicit normalization state and camera metadata. See `data/README.md`.

## Proposed approach
Classical camera stages with a compact conditioned RGB residual network. Maintain a separate packed-Bayer learned baseline.

## Dataset and splits
Proposed: SIDD Small initially. Verify paired target rendering and scene grouping before training. Dataset has not been downloaded.

## Training
Patch-based learning with reconstruction, structure, and colour objectives. Hyperparameters in `configs/default.yaml` are starting values, not tuned results. Training loop and learned-only network are pending.

## Evaluation
PSNR, SSIM, Delta E, LPIPS; parameters, exported size, latency, peak memory. Record target encoding, metric ranges, hardware, image resolution, warm-up and timing protocol.

## Optimization
Select runtime first; compare FP32 with supported FP16 or INT8. Report quality change and complete pipeline timings. Hosted inference does not establish on-device performance.

## Failure cases and limitations
Investigate clipping, mixed illuminants, metadata errors, unseen cameras, oversmoothing, and demosaicing artifacts. No cross-camera robustness claims without evidence.

## Deployment
Planned hosted comparison UI with loading/error states and bounded upload sizes. No deployment exists yet.

## Privacy, licensing, and ethics
Record dataset/code licenses. Avoid retaining user uploads by default; document temporary-file retention. Strip private camera/location metadata from downloadable previews unless intentionally requested.
