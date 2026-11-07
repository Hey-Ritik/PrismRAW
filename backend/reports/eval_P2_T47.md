1RUA24CSE0371,1RUA24CSE0365

# P2 — PrismRAW — T47

## Submission status

Team NOCTICS (T47), Section D; members Ritik Singh (1RUA24CSE0371) and Rishab Patil (1RUA24CSE0365). USNs are recorded in the required first-line format. Engineering evaluation is measured.

Public access: [Vercel frontend](https://prismraw-frontend.vercel.app/) and [Render backend](https://prismraw.onrender.com). The frontend is hosted on Vercel and the backend is hosted on Render. The student reports personally testing the deployed application; no specific test cases or outcomes beyond that statement are asserted. This documentation update does not claim independent deployed inference testing or verified Docker build/deployment evidence.

## Comparative evaluation

| Metric (P2 order) | Classical | Learned-only | Hybrid FP32 | Hybrid FP16 |
|---|---:|---:|---:|---:|
| PSNR (dB) ↑ | 38.4589 | 25.1178 | 39.2312 | 39.2310 |
| SSIM ↑ | 0.8867 | 0.9280 | 0.9027 | 0.9027 |
| Delta E 00 ↓ | 2.7438 | 10.4193 | 2.5231 | 2.5231 |
| LPIPS-Squeeze ↓ | 0.1042 | 0.2893 | 0.0936 | 0.0936 |
| Model parameters | N/A | 123,459 | 118,595 | 118,595 |
| Model file size (MB) | N/A | 0.5032 | 0.4832 | 0.2455 |
| CPU inference median / p95 (ms) | 99.47 / 129.83 | 41.48 / 65.88 | 153.79 / 168.43 | 21742.88 / 28262.64 |
| Peak process RAM (MiB) | 201.16 | 242.62 | 280.24 | 290.33 |

## Dataset and protocol

SIDD-CC pinned revision 5eb2ab6375d7f99f36b59bb5aacb46af18409412. Six physical training scenes (003–008), two development validation scenes (001/002), and two fresh final scenes (009/010). 32 deterministic 128x128 patches per scene; 192 train, 64 validation, 64 final test. Each method has the same four-pixel RAW context. Model and residual-blend selection used validation only. No model changes were selected using final holdout scores.

Targets are fixed approximate renders of matching clean RAW. Metrics use float sRGB [0,1]; Delta E is mean CIEDE2000 in Lab D65; LPIPS is v0.1 with pretrained SqueezeNet. Scores are arithmetic means over patches, with per-scene results in final_results.json. Coarse same-CFA translational checks found zero shift at the tested resolution for all ten pairs; this does not establish exact local/subpixel registration or absolute camera colour accuracy.

Both models trained for 30 CPU epochs. The hybrid refines bilateral-denoised linear RGB, learning residual detail/noise corrections. The learned-only model reconstructs directly from normalized packed Bayer plus camera metadata. A shared display renderer uses noisy-input exposure for all methods and references.

## Hardware and timing

```json
{
  "platform": "Windows-11-10.0.26300-SP0",
  "processor": "Intel64 Family 6 Model 141 Stepping 1, GenuineIntel",
  "logical_cpus": 12,
  "torch": "2.7.1+cpu",
  "device": "CPU",
  "torch_threads": 2
}
```

Isolated process per method; complete array processing, excludes model loading, file I/O, network and metrics. Sampled RSS includes Python/PyTorch runtime, not model-only memory. Not a smartphone measurement.
Three warm-up runs and ten timed runs at 256x256. Process RAM is sampled every 2 ms and includes the Python/PyTorch runtime; it is not model-only or smartphone memory. These are local CPU results.

## Optimization

FP16 checkpoint export: 483,245 to 245,497 bytes. Inference really uses half-precision weights and tensors; output is converted back to float32 for rendering and metrics. CPU FP16 may be slower; the measured times above are authoritative. FP32 remains available.

## Ablations

Removing learned residual refinement gives the classical row. Reduced precision is the FP16 row. Disabling hybrid adaptation gives:

```json
{
  "psnr": 22.329340485970704,
  "ssim": 0.9042694047093391,
  "delta_e_00": 8.917588559910655,
  "lpips": 0.06627127925457899
}
```

The no-adaptation output uses fixed exposure while the reference preserves the desired noisy-input-derived adaptive exposure. Its colour/brightness fidelity is therefore part of this ablation. Metrics can disagree: LPIPS/SSIM alone must not be interpreted as complete colour accuracy.

## Synthetic stress tests

These alter the held-out noisy input while retaining the original clean target rendering. Four patches total, not four independent scenes. Results demonstrate degradation and any partial recovery; no guaranteed recovery is claimed.

| Condition | Adaptive | Mean PSNR (dB) | Mean SSIM | Mean Delta E 00 |
|---|---|---:|---:|---:|
| colour_cast | False | 17.995 | 0.8816 | 16.879 |
| colour_cast | True | 21.810 | 0.8730 | 13.496 |
| dark_and_noisy | False | 12.943 | 0.4569 | 23.088 |
| dark_and_noisy | True | 19.828 | 0.3770 | 16.395 |
| heavy_noise | False | 17.214 | 0.1874 | 18.883 |
| heavy_noise | True | 22.497 | 0.3065 | 13.479 |

## Failure analysis

1. Learned-only reconstruction still oversmooths detail and produces colour errors. Limited paired captures and camera/style diversity prevent a strong end-to-end model despite improved resizing architecture. High SSIM can coexist with poor PSNR, Delta E and LPIPS.
2. Severe synthetic underexposure plus noise loses information. Adaptive tone mapping can recover brightness only partly and may amplify noise; see stress results rather than assuming recovery.
3. Heavy noise and colour casts degrade reconstruction. The metadata/profile correction is approximate and cannot recover clipped colours or infer an arbitrary missing calibration.
4. FP16 reduces storage but can increase CPU latency; a compatible mobile accelerator must be benchmarked before making mobile-speed claims.

## Application and reproducibility

Public application: [Vercel frontend](https://prismraw-frontend.vercel.app/), with API processing hosted by the [Render backend](https://prismraw.onrender.com). For optional local reproduction, run `.\run_local.ps1` and open http://127.0.0.1:7860. The app supports real sample scenes, DNG and explicit-metadata array uploads, reference metrics, stage previews, bounded central crops, error states, and downloadable PNG/JSON/ZIP results. Source uploads are temporary; generated results are cleaned after one hour on subsequent requests. Models and LPIPS are warmed at startup.

See README.md for exact download, preparation, training, selection, evaluation, benchmark, and packaging commands. Dockerfile is supplied; Docker image build is not claimed as verified unless separately tested. The release bundle contains small demo crops and trained weights; full datasets stay outside Git.

## Submission documentation

Team identity and public deployment URLs are supplied: [Vercel frontend](https://prismraw-frontend.vercel.app/) and [Render backend](https://prismraw.onrender.com). Complete the six required submission fields and any remaining student-authored observations. The separate [technical solution report](../../docs/submission/Technical_Solution_Report_P2_T47.md) retains its AI-assistance disclosure and unconfirmed contribution placeholders.
