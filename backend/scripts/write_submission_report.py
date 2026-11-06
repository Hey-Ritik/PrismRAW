"""Generate ordered submission metrics from actual evaluation and benchmark JSON."""

import argparse
from datetime import date
import json
from pathlib import Path
import re
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
team_path = Path("configs/team.json")
team = json.loads(team_path.read_text(encoding="utf-8")) if team_path.exists() else {}
parser.add_argument("--team-id", default=team.get("team_id", "TEAMID"))
parser.add_argument("--usns", default=",".join(team.get("usns", [])) or "USNS_PENDING_USER_INPUT")
args = parser.parse_args()
identity_pending = args.team_id == "TEAMID" or args.usns == "USNS_PENDING_USER_INPUT"
if not re.fullmatch(r"[A-Za-z0-9_-]+", args.team_id):
    raise ValueError("Use letters, digits, underscore or hyphen in team ID")
quality = json.loads(Path("reports/final_results.json").read_text())
benchmark = json.loads(Path("reports/benchmark.json").read_text())
optimization = json.loads(Path("reports/optimization.json").read_text())
methods = ("classical", "learned", "hybrid", "optimized")
labels = ("Classical", "Learned-only", "Hybrid FP32", "Hybrid FP16")
table = ["| Metric (P2 order) | " + " | ".join(labels) + " |", "|---|---:|---:|---:|---:|"]
for name, key, digits in (("PSNR (dB) ↑", "psnr", 4), ("SSIM ↑", "ssim", 4), ("Delta E 00 ↓", "delta_e_00", 4), ("LPIPS-Squeeze ↓", "lpips", 4)):
    values = ["Infinity" if quality["metrics"][m][key] is None else f"{quality['metrics'][m][key]:.{digits}f}" for m in methods]
    table.append("| " + name + " | " + " | ".join(values) + " |")
for name, key, formatter in (("Model parameters", "parameters", lambda v: f"{v:,}" if v else "N/A"),
                              ("Model file size (MB)", "model_bytes", lambda v: f"{v/1e6:.4f}" if v else "N/A"),
                              ("CPU inference median / p95 (ms)", "median_ms", lambda v: f"{v:.2f}"),
                              ("Peak process RAM (MiB)", "peak_process_rss_mib", lambda v: f"{v:.2f}")):
    values = [formatter(benchmark["methods"][m][key]) + (f" / {benchmark['methods'][m]['p95_ms']:.2f}" if key == "median_ms" else "") for m in methods]
    table.append("| " + name + " | " + " | ".join(values) + " |")
text = [args.usns, "", f"# P2 — PrismRAW — {args.team_id}", "", "## Submission status", "",
        ("Team identity is pending. " if identity_pending else f"Team {args.team_id}; USNs are recorded on the first line. ") + "Engineering evaluation is measured. The app runs locally; a public hosted URL has not been published.", "", "## Comparative evaluation", "", *table, "",
        "## Dataset and protocol", "",
        "SIDD-CC pinned revision 5eb2ab6375d7f99f36b59bb5aacb46af18409412. Six physical training scenes (003–008), two development validation scenes (001/002), and two fresh final scenes (009/010). 32 deterministic 128x128 patches per scene; 192 train, 64 validation, 64 final test. Each method has the same four-pixel RAW context. Model and residual-blend selection used validation only. No model changes were selected using final holdout scores.", "",
        "Targets are fixed approximate renders of matching clean RAW. Metrics use float sRGB [0,1]; Delta E is mean CIEDE2000 in Lab D65; LPIPS is v0.1 with pretrained SqueezeNet. Scores are arithmetic means over patches, with per-scene results in final_results.json. Coarse same-CFA translational checks found zero shift at the tested resolution for all ten pairs; this does not establish exact local/subpixel registration or absolute camera colour accuracy.", "",
        "Both models trained for 30 CPU epochs. The hybrid refines bilateral-denoised linear RGB, learning residual detail/noise corrections. The learned-only model reconstructs directly from normalized packed Bayer plus camera metadata. A shared display renderer uses noisy-input exposure for all methods and references.", "",
        "## Hardware and timing", "", "```json", json.dumps(benchmark["hardware"], indent=2), "```", "",
        benchmark["scope"], "Three warm-up runs and ten timed runs at 256x256. Process RAM is sampled every 2 ms and includes the Python/PyTorch runtime; it is not model-only or smartphone memory. These are local CPU results.", "",
        "## Optimization", "", f"FP16 checkpoint export: {optimization['fp16_export']['original_bytes']:,} to {optimization['fp16_export']['optimized_bytes']:,} bytes. Inference really uses half-precision weights and tensors; output is converted back to float32 for rendering and metrics. CPU FP16 may be slower; the measured times above are authoritative. FP32 remains available.", "",
        "## Ablations", "", "Removing learned residual refinement gives the classical row. Reduced precision is the FP16 row. Disabling hybrid adaptation gives:", "", "```json", json.dumps(quality["metrics"]["hybrid_no_adaptation"], indent=2), "```", "",
        "The no-adaptation output uses fixed exposure while the reference preserves the desired noisy-input-derived adaptive exposure. Its colour/brightness fidelity is therefore part of this ablation. Metrics can disagree: LPIPS/SSIM alone must not be interpreted as complete colour accuracy.", "",
        "## Synthetic stress tests", "", "These alter the held-out noisy input while retaining the original clean target rendering. Four patches total, not four independent scenes. Results demonstrate degradation and any partial recovery; no guaranteed recovery is claimed.", "", "| Condition | Adaptive | Mean PSNR (dB) | Mean SSIM | Mean Delta E 00 |", "|---|---|---:|---:|---:|"]
for condition in sorted({row["condition"] for row in quality["stress_tests"]}):
    for adaptive in (False, True):
        rows = [row["metrics"] for row in quality["stress_tests"] if row["condition"] == condition and row["adaptive"] == adaptive]
        text.append(f"| {condition} | {adaptive} | {np.mean([r['psnr'] for r in rows]):.3f} | {np.mean([r['ssim'] for r in rows]):.4f} | {np.mean([r['delta_e_00'] for r in rows]):.3f} |")
text += ["", "## Failure analysis", "",
         "1. Learned-only reconstruction still oversmooths detail and produces colour errors. Limited paired captures and camera/style diversity prevent a strong end-to-end model despite improved resizing architecture. High SSIM can coexist with poor PSNR, Delta E and LPIPS.",
         "2. Severe synthetic underexposure plus noise loses information. Adaptive tone mapping can recover brightness only partly and may amplify noise; see stress results rather than assuming recovery.",
         "3. Heavy noise and colour casts degrade reconstruction. The metadata/profile correction is approximate and cannot recover clipped colours or infer an arbitrary missing calibration.",
         "4. FP16 reduces storage but can increase CPU latency; a compatible mobile accelerator must be benchmarked before making mobile-speed claims.", "",
         "## Application and reproducibility", "",
         "Run `.\\run_local.ps1` and open http://127.0.0.1:7860. The app supports real sample scenes, DNG and explicit-metadata array uploads, reference metrics, stage previews, bounded central crops, error states, and downloadable PNG/JSON/ZIP results. Source uploads are temporary; generated results are cleaned after one hour on subsequent requests. Models and LPIPS are warmed at startup.", "",
         "See README.md for exact download, preparation, training, selection, evaluation, benchmark, and packaging commands. Dockerfile is supplied; Docker image build is not claimed as verified unless separately tested. The release bundle contains small demo crops and trained weights; full datasets stay outside Git.", "",
         "## Remaining submission-only actions", "", ("Provide team ID and comma-separated USNs, " if identity_pending else "Team identity is complete. Next, ") + "publish/verify a public hosted URL and record/rehearse the team demo/viva. A localhost link is not a public hosted submission URL."]
destination = Path("reports") / f"eval_P2_{args.team_id}.md"
destination.write_text("\n".join(text) + "\n", encoding="utf-8")
Path("reports/final_summary.md").write_text(f"# PrismRAW final local results — {args.team_id}\n\n" + "\n".join(table) + "\n\nSee the ordered evaluation report and JSON for methodology, ablations, stress tests and limitations. Local application: http://127.0.0.1:7860. Public hosting remains pending.\n", encoding="utf-8")
print(f"Wrote {destination}")
