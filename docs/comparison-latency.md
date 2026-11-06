# Reuse ISP work in the comparison

The comparison retains the denoised classical RGB buffer and reuses it for a denoised-input hybrid model, including optional FP16. The original standalone hybrid path remains available for independent calls and linear-input models. Reuse is internal to the same request and requires matching dimensions and adaptive settings; it is not a cache of previous images.

Learned processing now accepts the final sharpening strength so the app renders it once instead of discarding its first render. Default standalone learned processing is unchanged.

## Measured complete comparison time

Same 256x256 Nexus sample, two CPU threads, three warmups and ten measured calls, classical/learned/hybrid enabled and FP16 off. Wall time includes LPIPS, PNG previews/output images, JSON and ZIP creation, but excludes HTTP/network and cold model loading.

| Before median | After median | Reduction |
|---:|---:|---:|
| 670.27 ms | 641.19 ms | 4.34% |

The modest improvement is for the complete comparison, not a new standalone hybrid benchmark. The earlier 77.83 ms standalone hybrid figure is a different measurement scope and must not be replaced with the comparison wall time. Full observations are in backend/reports/comparison_latency_optimization.json.

The before/after comparison at 256px produced identical PNG hashes and quality scores including LPIPS. Additional exact comparisons covered all four demo scenes at 64px with adaptation on/off and included FP16 for scene 003. Test-only copies of the earlier implementation remained in the ignored work directory.

## Timing and memory accounting

The report's processing_ms attributes the measured classical cost to hybrid as well as its additional refinement/render cost. This avoids making hybrid appear faster merely by excluding shared work. incremental_processing_ms separately records the additional computation. Because classical work is reused rather than repeated, method totals should not be summed to estimate request time. Standalone pipeline benchmarks remain separate.

Keeping one float32 RGB buffer costs approximately 0.75 MiB at 256px or 3 MiB at 512px until the final method finishes. Other stage arrays are still released early. This is a time/memory trade-off, not a claim of reduced peak RAM or guaranteed Render Free compatibility. No crops, weights, precision, restoration settings or quality metrics were changed.
