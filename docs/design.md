# Design document

## Problem and scope
P2: reconstruct high-quality RGB from smartphone RAW while balancing quality, model size, memory, and latency.

## Input contract
Validated sensor data plus explicit normalization state and camera metadata. See `data/README.md`.

## Proposed approach
Classical camera stages with a compact conditioned RGB residual network. Maintain a separate packed-Bayer learned baseline.

## Dataset and splits
Initial real-data validation uses a clean RAW and metadata from the SIDD authors' reference repository. A pinned SIDD-CC pilot now includes one capture from each of three physical scenes. Seeded scene-level assignments precede patch extraction: scene 003 train, 001 validation, 002 test. Eight patches per scene are used. This small split also changes camera between scenes, so it does not isolate scene versus sensor generalization. Independent alignment and colour-rendering validation remain pending.

## Training
Implemented pilot: conditioned RGB residual U-Net, linear-RGB L1 plus gradient consistency, Adam, seeded CPU training, and checkpoint selection by validation L1 among trained epochs. `configs/pilot.yaml` is the executed recipe; `configs/default.yaml` is an unexecuted starting proposal. Targets are clean RAW rendered through the same approximate fixed transform as noisy input. No perceptual or explicit colour loss or augmentation is included yet. The three-epoch pilot underperforms classical denoising; its validation L1 also remains worse than the initial identity refinement. Learned-only network and larger training remain pending.

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
