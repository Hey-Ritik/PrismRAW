"""One-time held-out pilot check after validation-only checkpoint selection."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from smartphone_isp.data.splits import validate_disjoint
from smartphone_isp.pipelines.classical import render_linear
from smartphone_isp.pipelines.hybrid import load_model, refine


def psnr(prediction, target):
    mse = float(np.mean((prediction.astype(np.float64) - target) ** 2))
    return None if mse == 0 else float(-10 * np.log10(mse))


def evaluate(manifest_path, checkpoint, output):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_disjoint(manifest["samples"])
    samples = [row for row in manifest["samples"] if row["split"] == "test"]
    torch.set_num_threads(2)
    model, document = load_model(checkpoint)
    if document.get("manifest_sha256") != hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest():
        raise ValueError("Checkpoint/manifest mismatch; use the manifest recorded by this training run")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample in enumerate(samples):
        with np.load(Path(manifest_path).parent / sample["path"], allow_pickle=False) as pair:
            predicted = refine(model, pair["source"], *pair["conditions"])
            hybrid = render_linear(predicted, float(pair["exposure"]), float(pair["sharpen"]))[0]
            target = pair["target_srgb"]
            classical = pair["classical"]
            row = {"sample_id": sample["sample_id"], "scene_id": sample["scene_id"],
                   "classical_psnr": psnr(classical, target), "hybrid_psnr": psnr(hybrid, target)}
            rows.append(row)
            if index < 4:
                panel = np.concatenate([classical, hybrid, target], axis=1)
                Image.fromarray(np.rint(np.clip(panel, 0, 1) * 255).astype(np.uint8)).save(output / f"comparison-{index}.png")
    means = {key: (None if any(row[key] is None for row in rows) else float(np.mean([row[key] for row in rows])))
             for key in ("classical_psnr", "hybrid_psnr")}
    report = {"scope": "Pilot only; not official benchmark results", "split": "test",
              "target": manifest["target"], "registration": manifest["registration"],
              "checkpoint_epoch": document["epoch"], "samples": rows, "mean_patch_psnr": means,
              "metric": "sRGB float [0,1], full stored patch, arithmetic mean of per-patch PSNR in dB; null denotes infinite PSNR from exact matching",
              "comparison_image_order": "Classical | Hybrid | Fixed-rendered clean RAW target",
              "limitations": ["Only one held-out scene in default pilot", "Approximate colour rendering",
                              "No SSIM/Delta E/LPIPS or device benchmark in this pilot check"]}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(means, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/pilot/manifest.json")
    parser.add_argument("--checkpoint", default="checkpoints/pilot-best.pt")
    parser.add_argument("--output", default="outputs/pilot-evaluation")
    args = parser.parse_args()
    evaluate(args.manifest, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
