# PrismRAW design

## Problem and scope
P2 requires efficient hybrid smartphone RAW-to-RGB reconstruction, comparisons against classical and learned-only processing, adaptation, model optimization and measured quality/resource trade-offs. This implementation provides a local app and reproducible research evaluation, for team T47 (USNs 1RUA24CSE0371, 1RUA24CSE0365), with public hosting pending.

## Sensor and metadata contract
2x2 Bayer arrays with explicit normalization state, CFA phase, RGB white-balance gains, and camera-to-linear-sRGB matrix. DNG extraction uses rawpy sensor buffers plus profile metadata; array loading supports NPY/NPZ and both MATLAB formats. Colour profiles are approximate single-illuminant transforms, not full calibrated DNG processing. Input limits, privacy and deployment behavior are in deployment.md.

## Processing architecture
Classical: normalization, Bayer white balance, bilinear demosaicing, CCM, bilateral filtering, adaptive exposure and luminance tone mapping, sRGB encoding and mild sharpening. Hybrid: the same front end and denoising, followed by a conditioned residual U-Net before common display rendering. Learned-only: packed Bayer and 16 camera features through a compact U-Net with resize/convolution RGB reconstruction, then the common display renderer. FP16 hybrid really executes half-precision weights/tensors and converts output to float32 for rendering.

## Dataset and leakage control
Pinned SIDD-CC revision 5eb2ab6375d7f99f36b59bb5aacb46af18409412, ten captures from ten physical scenes across Nexus 6, Google Pixel, iPhone 7, LG G4 and Samsung S6. Scenes 003–008 train, 001/002 validation, 009/010 fresh final holdout. Thirty-two deterministic patches per scene. Frozen scene assignments precede extraction. Old pilot test 002 became development-only because it was already inspected; fresh final data was excluded from model selection. Same-CFA translational checks found zero shift for these pairs at the tested resolution, without establishing exact local registration.

## Training and selection
Hybrid learns a residual after classical denoising. Learned-only reconstructs colour and structure from RAW. Both use 30 CPU epochs, Adam, L1 plus gradient consistency, fixed seeds and matching four-pixel context. Outputs are cropped to central 128x128 targets for loss. Optional spatial augmentation updates Bayer phase and semantic packing, and is restricted to training; it was disabled in recorded final runs. Checkpoints and residual blend are selected by validation linear-RGB L1. Test scores do not choose models. Recipe/config, manifest hash, epoch, dtype and input stage are recorded with checkpoints.

## Evaluation and efficiency
Target RGB is the fixed approximate rendering of clean RAW. All compared methods share noisy-input exposure settings. Report PSNR, SSIM, Delta E 00 in Lab D65, LPIPS-Squeeze v0.1, model parameters/file sizes, warmed CPU median/p95 latency and sampled process RAM. RSS includes the runtime and is not smartphone memory. Ablations remove learned correction/adaptation and compare precision. Synthetic stress tests retain the original target and are reported separately from real captures. Full methodology and measured results are in the ordered report and JSON.

## Current evidence and limitations
On 64 fresh holdout patches, hybrid improves mean PSNR from 38.4589 to 39.2312 dB, SSIM from 0.8867 to 0.9027, colour error from 2.7438 to 2.5231, and LPIPS from 0.1042 to 0.0936. This limited two-scene result does not establish broad superiority. Learned-only still oversmooths and shifts colours. Highlight clipping, extreme noise, unknown profiles and mixed illumination remain failure modes. FP16 reduces storage but may worsen CPU latency. No on-device speed or absolute colour-fidelity claim is made.

## Application and deployment
FastAPI serves a responsive local UI with serialized CPU inference, samples, uploads, loading/error states, intermediate visualizations, optional reference metrics and downloadable results. Preloading avoids expensive model/metric initialization on the first click. Files are cropped rather than resizing the mosaic; user references must already correspond spatially. Docker packaging and a small binary release bundle are supplied. Local use is verified; public hosting, actual Docker build and team viva are not falsely marked complete.

## Privacy and attribution
No external inference or image API is used. Hosted uploads are processed on the hosting server; the UI reflects that setting. Uploads use temporary storage; generated results expire after an hour on a later request. Full training data and intermediate weights are not committed to Git; the three final deployment weights and explicitly attributed small demo crops are tracked for reproducible hosting. Dataset/software notices are in THIRD_PARTY_NOTICES.md.

## Hosted profile
The Render Blueprint uses 256px crops and one CPU thread, respects PORT, and labels server processing accurately. DNG cropping precedes float calibration allocation, while optional references are cropped to the same region. This profile changes workload size, not model weights or historical evaluation results. Deployment assets are tracked; public service creation, Linux image build, and real Render resource fit remain external checks.
