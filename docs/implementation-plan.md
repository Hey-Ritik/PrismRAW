# Implementation plan

## 1. Data validation
- [ ] Download a manageable paired dataset subset.
- [ ] Inspect one RAW, clean target, and metadata tuple.
- [ ] Verify CFA, normalization state, colour matrices and registration.
- [ ] Save scene-level train/validation/test manifests before extracting patches.

## 2. Classical ISP
- [ ] Implement MAT/DNG adapters without hidden full-ISP preprocessing.
- [ ] Integrate Bayer white balance and demosaicing.
- [ ] Validate camera-to-output colour transformation.
- [ ] Add denoising, adaptive tone mapping, controlled sharpening.
- [ ] Export intermediate stages and end-to-end RGB output.

## 3. Learning
- [ ] Implement paired patch dataset and CFA-safe augmentation.
- [ ] Implement compact packed-Bayer-to-RGB baseline.
- [ ] Train residual hybrid model using explicit linear-RGB inputs.
- [ ] Add noise/illumination conditioning and checkpoint/config tracking.

## 4. Evidence
- [ ] Evaluate classical, learned-only, hybrid on identical held-out scenes.
- [ ] Add metric definitions, colour-space conversions, and alignment checks.
- [ ] Ablate learned refinement, adaptation, and reduced precision separately.
- [ ] Record real and synthetic stress tests separately; explain failures.

## 5. Optimization and delivery
- [ ] Select device/runtime, export and benchmark optimized model.
- [ ] Measure model and complete-pipeline latency and memory.
- [ ] Implement and host the comparison app.
- [ ] Complete evaluation report, architecture, design, reproducibility instructions.
- [ ] Rehearse a three-minute demonstration with unseen inputs.
