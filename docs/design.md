# Design document

## Problem and scope
P2: reconstruct high-quality RGB from smartphone RAW while balancing quality, model size, memory, and latency.

## Input contract
Validated sensor data plus explicit normalization state and camera metadata. See `data/README.md`.

## Proposed approach
Classical camera stages with a compact conditioned RGB residual network. Maintain a separate packed-Bayer learned baseline.

## Dataset and splits
Initial real-data validation uses a clean RAW and metadata from the SIDD authors' reference repository. The expanded pinned SIDD-CC pilot includes five physical scenes: 003/004/005 train, 001 validation, 002 test. Eight patches per scene are used. `configs/pilot-splits.json` freezes original holdouts; additional scenes enter training. Existing validation/test array values were checked to remain identical. All three training captures are Nexus 6, while validation is Samsung S6 and test Google Pixel. Scene and camera changes are confounded. Independent alignment and colour-rendering validation remain pending.

## Training
Implemented: conditioned RGB residual U-Net and learned-only packed-Bayer U-Net with subpixel RGB reconstruction. Training uses linear-RGB L1 plus gradient consistency, Adam, seeded CPU execution, and validation-only checkpoint selection. Learned-only camera inputs include white balance, colour matrix, and CFA phase; these are network features, not classical preprocessing operations. `configs/expanded-hybrid.yaml` and `configs/expanded-learned.yaml` execute 20 epochs on the same expanded data. Targets use an approximate fixed transform of clean RAW. No perceptual/explicit colour loss or augmentation is included. Larger training remains necessary; both models underperform classical on the exploratory pilot test.

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
