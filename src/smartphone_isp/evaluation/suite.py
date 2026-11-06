"""Complete quality comparison, component ablations, and synthetic stress checks."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from smartphone_isp.data.bayer import unpack_bayer
from smartphone_isp.data.splits import validate_disjoint
from smartphone_isp.evaluation.metrics import quality
from smartphone_isp.pipelines.classical import process, render_linear
from smartphone_isp.pipelines.hybrid import load_model as load_hybrid, refine, process_hybrid
from smartphone_isp.pipelines.learned import load_model as load_learned, reconstruct
from smartphone_isp.optimization.export import export_fp16


def paths():
    return {"hybrid": Path("checkpoints/final-hybrid.pt"), "learned": Path("checkpoints/final-learned.pt"),
            "optimized": Path("checkpoints/final-hybrid-fp16.pt")}


def select_blend(manifest_path):
    """Select residual blend on validation linear-RGB L1, never test samples."""
    torch.set_num_threads(2)
    manifest = json.loads(Path(manifest_path).read_text())
    model, document = load_hybrid(paths()["hybrid"])
    model.blend = 1.0
    scores = {alpha: [] for alpha in (0.0, 0.25, 0.5, 1.0)}
    context = manifest["input_context"]
    for row in manifest["samples"]:
        if row["split"] != "validation":
            continue
        with np.load(Path(manifest_path).parent / row["path"], allow_pickle=False) as pair:
            source = pair["denoised"]
            prediction = refine(model, source, *pair["conditions"])
            for alpha in scores:
                blended = source + alpha * (prediction - source)
                if context:
                    blended = blended[context:-context, context:-context]
                scores[alpha].append(float(np.mean(np.abs(blended-pair["target"]))))
    means = {alpha: float(np.mean(values)) for alpha, values in scores.items()}
    selected = min(means, key=means.get)
    document.update(blend=selected, blend_selection={"split": "validation", "metric": "linear-RGB L1", "scores": means})
    torch.save(document, paths()["hybrid"])
    result = {"selected": selected, "validation_l1": means, "fp16_export": export_fp16(paths()["hybrid"], paths()["optimized"])}
    print(json.dumps(result, indent=2), flush=True)
    return result


def mean_metrics(rows):
    return {key: (None if any(row[key] is None for row in rows) else float(np.mean([row[key] for row in rows]))) for key in rows[0]}


def evaluate(manifest_path, split="test", ablations=True):
    torch.set_num_threads(2)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validate_disjoint(manifest["samples"])
    context = manifest["input_context"]
    models = {}
    for method, path in paths().items():
        loader = load_learned if method == "learned" else load_hybrid
        model, doc = loader(path)
        if doc["manifest_sha256"] != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
            raise ValueError("Model/manifest mismatch")
        models[method] = model
    all_rows = []
    stress = []
    for i, sample in enumerate(row for row in manifest["samples"] if row["split"] == split):
        with np.load(manifest_path.parent / sample["path"], allow_pickle=False) as pair:
            exposure, sharpen = float(pair["exposure"]), float(pair["sharpen"])
            def display(linear):
                rendered = render_linear(linear, exposure, sharpen)[0]
                return rendered[context:-context, context:-context] if context else rendered
            outputs = {"classical": pair["classical"],
                       "learned": display(reconstruct(models["learned"], pair["packed_raw"], pair["calibration"]))}
            for method in ("hybrid", "optimized"):
                outputs[method] = display(refine(models[method], pair["denoised"], *pair["conditions"]))
            vector = pair["calibration"]
            pattern = ("RGGB", "BGGR", "GRBG", "GBRG")[int(np.argmax(vector[12:]))]
            metadata = {"normalized": True, "bayer_pattern": pattern, "white_balance": vector[:3].tolist(), "camera_to_linear_srgb": vector[3:12].reshape(3, 3).tolist()}
            raw = unpack_bayer(pair["packed_raw"], pattern)
            if ablations:
                fixed = process_hybrid(raw, metadata, models["hybrid"], adaptive=False)[0]
                outputs["hybrid_no_adaptation"] = fixed[context:-context, context:-context] if context else fixed
            measurements = {method: quality(image, pair["target_srgb"]) for method, image in outputs.items()}
            all_rows.append({"sample_id": sample["sample_id"], "scene_id": sample["scene_id"], "metrics": measurements})
            # Prespecified stress protocol on first two patches of each evaluation scene.
            if ablations and int(sample["sample_id"].split("-")[-1]) < 2:
                rng = np.random.default_rng(42+i)
                variants = {"dark_and_noisy": np.clip(raw * 0.25 + rng.normal(0, 0.01, raw.shape), 0, 1).astype(np.float32),
                            "heavy_noise": np.clip(raw + rng.normal(0, 0.04, raw.shape), 0, 1).astype(np.float32),
                            "colour_cast": np.clip(raw * np.tile(np.array([[1.3, 0.8], [0.8, 0.6]], dtype=np.float32), (raw.shape[0]//2, raw.shape[1]//2)), 0, 1)}
                for condition, degraded in variants.items():
                    for adaptive in (False, True):
                        image = process_hybrid(degraded, metadata, models["hybrid"], adaptive=adaptive)[0]
                        if context:
                            image = image[context:-context, context:-context]
                        stress.append({"sample_id": sample["sample_id"], "condition": condition, "adaptive": adaptive,
                                       "metrics": quality(image, pair["target_srgb"], perceptual=False)})
        print(f"{split}: {i+1} patches evaluated", flush=True) if (i+1) % 16 == 0 else None
    by_method = {method: mean_metrics([row["metrics"][method] for row in all_rows]) for method in all_rows[0]["metrics"]}
    report = {"split": split, "scene_ids": sorted({row["scene_id"] for row in all_rows}), "patch_count": len(all_rows),
              "dataset_revision": manifest["revision"], "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "metrics": by_method, "per_scene": {scene: {method: mean_metrics([row["metrics"][method] for row in all_rows if row["scene_id"] == scene]) for method in by_method} for scene in sorted({row["scene_id"] for row in all_rows})},
              "samples": all_rows, "stress_tests": stress,
              "protocol": {"colour": "Float sRGB [0,1], Lab D65, CIEDE2000 mean pixels", "lpips": "LPIPS v0.1 SqueezeNet, pretrained ImageNet backbone, input [-1,1]",
                           "aggregation": "Arithmetic mean per patch; also report scene means", "target": manifest["target"],
                           "context": manifest["context_protocol"], "stress": "Synthetic degradations; target remains original clean render; not a separate real-capture benchmark"},
              "limitations": ["Only two fresh final scenes; not official benchmark ranking", "Approximate colour calibration; Delta E measures the supplied target rendering", "Neural superiority is not assumed", "Team identity and public hosting are separate submission steps"]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/final/manifest.json")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--select-blend", action="store_true")
    args = parser.parse_args()
    Path("reports").mkdir(exist_ok=True)
    if args.select_blend:
        report = select_blend(args.manifest)
        destination = Path("reports/optimization.json")
    else:
        report = evaluate(args.manifest, args.split)
        destination = Path("reports/final_results.json" if args.split == "test" else "reports/validation_results.json")
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
