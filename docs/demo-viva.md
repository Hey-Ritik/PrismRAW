# Three-minute local demo and viva guide

1. **0:00–0:30 — Problem.** Explain Bayer measurements, sensor noise, white balance, camera colour correction, and the quality/resource trade-off. Open the local URL on a fresh browser tab.
2. **0:30–1:20 — Process a real capture.** Choose a SIDD-CC sample; run adaptive processing. Compare classical, learned-only, hybrid and FP16 images. Zoom into an edge or texture and show the clean-RAW render reference. Explain that captured RGB ground truth is not being claimed.
3. **1:20–2:00 — Evidence.** Show quality metrics, intermediate stages, the final held-out report, and the CPU/storage trade-off. Explain that FP16 halves weights but may be slower on this CPU. Learned-only can score high SSIM while having poor colour and PSNR.
4. **2:00–2:30 — Unseen input and errors.** Upload supported Bayer DNG or an array with metadata. Show that a reference is required for scores. Demonstrate clear rejection of JPEG-as-RAW or missing metadata. Do not promise every camera profile is supported.
5. **2:30–3:00 — Limitations.** Discuss approximate colour calibration, only two final holdout scenes, stress-test degradation, and the need for genuine device benchmarks. Explain each member's implementation contribution truthfully.

Every member should be able to describe scene-level leakage prevention, residual learning after denoising, learned-only camera metadata, CFA-safe augmentation, context cropping, validation-only checkpoint/blend selection, CIEDE2000/LPIPS conventions, and at least one observed failure. This guide is not a substitute for an actual team rehearsal.
