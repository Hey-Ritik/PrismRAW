# Implementation plan

## 1. Data validation
- [x] Download one SIDD-CC noisy/clean RAW development pair with metadata and provenance.
- [x] Inspect RAW pair dimensions, values, and metadata; RGB target validation remains pending.
- [ ] Verify CFA, normalization state, colour matrices and registration.
- [x] Save a deterministic pilot scene split plan before patch extraction, then save patch manifest.

## 2. Classical ISP
- [ ] Implement MAT/DNG adapters without hidden full-ISP preprocessing.
- [x] Integrate Bayer white balance and bilinear demosaicing for array inputs.
- [ ] Validate camera-to-output colour transformation.
- [x] Add bilateral denoising, adaptive tone mapping, controlled sharpening.
- [x] Export intermediate arrays and end-to-end RGB output.

## 3. Learning
- [ ] Implement paired patch dataset and CFA-safe augmentation.
- [x] Implement and run compact packed-Bayer-to-RGB baseline with explicit camera metadata.
- [x] Run three-epoch CPU pilot training with explicit linear-RGB input/target pairs; larger training pending.
- [x] Add noise/illumination conditioning and checkpoint/config tracking.
- [x] Expand to five physical scenes and run both models for 20 CPU epochs, preserving pilot holdouts.

## 4. Evidence
- [x] Run exploratory classical/learned-only/hybrid PSNR comparison on identical pilot patches; full metrics/fresh final holdout pending.
- [ ] Add metric definitions, colour-space conversions, and alignment checks.
- [ ] Ablate learned refinement, adaptation, and reduced precision separately.
- [ ] Record real and synthetic stress tests separately; explain failures.

## 5. Optimization and delivery
- [ ] Select device/runtime, export and benchmark optimized model.
- [ ] Measure model and complete-pipeline latency and memory.
- [ ] Implement and host the comparison app.
- [ ] Complete evaluation report, architecture, design, reproducibility instructions.
- [ ] Rehearse a three-minute demonstration with unseen inputs.
