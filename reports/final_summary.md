# PrismRAW final local results — T47

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

See the ordered evaluation report and JSON for methodology, ablations, stress tests and limitations. Local application: http://127.0.0.1:7860. Public hosting remains pending.
