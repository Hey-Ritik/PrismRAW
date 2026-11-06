# CPU latency optimization

The RGB bilateral denoiser now reuses scratch/distance buffers and explicitly sums the three colour channels. It preserves the original arithmetic order and avoids repeated temporary allocations and generic channel reductions. Hybrid processing skips the classical display render that it previously discarded before rendering the learned refinement. Default classical processing still returns all rendered stages.

No checkpoint, crop limit, denoising window, noise adaptation, sharpening, precision or quality metric was changed. LPIPS remains enabled in the application. Tests verify exact denoiser equality for float32/float64 and four sigma values, and exact hybrid image equality while skipping the unused render. A separate before/after check confirmed exact classical outputs for all four 256px demo scenes and exact hybrid output for the benchmark capture.

## Fresh before/after measurements

Same machine, separate process per method, 256x256 Bayer capture, two PyTorch threads, three warmups and 20 measured iterations. Timings exclude loading, file I/O, network and metric computation.

| Method | Before median ms | After median ms | Reduction |
|---|---:|---:|---:|
| Classical | 62.5184 | 44.3938 | 28.99% |
| Learned | 22.8481 | 22.4206 | 1.87% |
| Hybrid | 103.6925 | 77.8347 | 24.94% |

The learned pipeline was unchanged; its small timing difference is run variation, not a claimed optimization. The hybrid remains slower than classical but has lower latency than its previous implementation. These are warmed local CPU timings, not Render, smartphone, full-resolution frame or complete web-request measurements. Runtime speed varies with hardware and load.

The historical submission benchmark (153.79 ms hybrid) is retained as a separate earlier measurement. Do not describe 153.79 to 77.83 as a controlled speedup: the fresh controlled comparison starts at 103.69 ms. Full median/p95, memory and protocol records are in backend/reports/latency_optimization.json.

To measure current isolated processing from backend/:

```text
python -m smartphone_isp.evaluation.benchmark --worker classical --iterations 20
python -m smartphone_isp.evaluation.benchmark --worker hybrid --iterations 20
```

Use PYTHONPATH=src and the same CPU/thread/input settings for comparisons. The normal application uses the optimization automatically after restarting or redeploying the backend.
