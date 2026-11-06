# Implementation and verification status

- [x] Pinned paired dataset downloads, provenance and checksum checks.
- [x] Ten physical scenes across five camera types; fixed development/final splits.
- [x] CFA/normalization inspection and coarse translational correspondence checks.
- [x] NPY/NPZ/MAT/HDF5 input and real Bayer DNG extraction.
- [x] Classical ISP with adaptive parameters and intermediate outputs.
- [x] Learned-only RAW-to-RGB model, hybrid residual model, matched RAW context.
- [x] CFA-safe optional training augmentation and scene-leakage checks.
- [x] Thirty-epoch training with validation-only checkpoint/blend selection.
- [x] Full PSNR/SSIM/Delta E/LPIPS comparison on fresh final scenes.
- [x] Refinement/adaptation/precision ablations and synthetic stress/failure analysis.
- [x] Actual FP16 export/inference, model size, CPU latency and sampled process memory.
- [x] Responsive local app, samples/uploads, loading/errors, references, stage views, downloads.
- [x] Docker packaging recipe, reproducibility docs and release-bundle script.
- [x] Local desktop/mobile browser and download checks; pipeline/API tests.
- [x] Ordered measured evaluation report and architecture/design/demo documents.
- [ ] Absolute colour calibration, broader generalization and true mobile-device benchmarking: research limitations, not claimed complete.
- [x] Team T47 and both USNs recorded in configuration and the submission report.
- [ ] Public hosted URL and actual team demo/viva: submission actions pending.
- [ ] Docker image build: recipe supplied; not verified on this machine.

Historical pilot reports are retained as experimental history. Current results and limits are recorded in the final report; local operation is not presented as a public deployment.
