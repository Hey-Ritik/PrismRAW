"""Reproducible pilot training with validation-only checkpoint selection."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

from smartphone_isp.data.paired import PairedDataset
from smartphone_isp.models.residual_unet import ResidualUNet
from smartphone_isp.models.raw_to_rgb import RawToRGBNet, ResizeRawToRGBNet
from smartphone_isp.training.alignment import crop_prediction


def objective(prediction, target, gradient_weight=0.05):
    reconstruction = F.l1_loss(prediction, target)
    dx = F.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1])
    dy = F.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    return reconstruction + gradient_weight * (dx + dy)


def resume_manifest_matches(document, manifest, manifest_hash, previous_path=None):
    if document.get("manifest_sha256") == manifest_hash:
        return True
    if not previous_path:
        return False
    previous_path = Path(previous_path)
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    return (hashlib.sha256(previous_path.read_bytes()).hexdigest() == document.get("manifest_sha256")
            and previous["scene_assignments"] == manifest["scene_assignments"]
            and previous.get("input_context", 0) == manifest.get("input_context", 0))


def validation_loss(model, loader, device, context=0):
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for source, conditions, target in loader:
            source, conditions, target = [value.to(device) for value in (source, conditions, target)]
            prediction = crop_prediction(model(source, conditions), target, context)
            total += F.l1_loss(prediction, target).item() * len(source)
            count += len(source)
    return total / count


def train(config):
    for key in ("epochs", "batch_size", "base_channels", "threads"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 < float(config["learning_rate"]) < 1:
        raise ValueError("Invalid learning rate")
    if not 0 <= float(config.get("gradient_weight", 0.05)) <= 1:
        raise ValueError("gradient_weight must be finite and between 0 and 1")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config["threads"]))
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    kind = config.get("model_kind", "hybrid")
    architecture = config.get("architecture", "subpixel-v1")
    if kind == "learned" and architecture not in {"subpixel-v1", "resize-v2"}:
        raise ValueError("Unknown learned architecture")
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    context = int(manifest.get("input_context", 0))
    train_data = PairedDataset(config["manifest"], "train", kind, config.get("hybrid_input", "linear"), bool(config.get("augment", False)))
    val_data = PairedDataset(config["manifest"], "validation", kind, config.get("hybrid_input", "linear"))
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], num_workers=0)
    model_type = (ResizeRawToRGBNet if architecture == "resize-v2" else RawToRGBNet) if kind == "learned" else ResidualUNet
    model = model_type(config["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest()
    if config.get("resume"):
        document = torch.load(config["resume"], map_location=device, weights_only=True)
        expected_format = "prismraw-bayer-resize-v2" if kind == "learned" and architecture == "resize-v2" else "prismraw-bayer-v1" if kind == "learned" else "prismraw-residual-v1"
        matching_manifest = resume_manifest_matches(document, manifest, manifest_hash, config.get("resume_manifest"))
        if document.get("format") != expected_format or not matching_manifest:
            raise ValueError("Resume checkpoint must match architecture and scene-separated manifest")
        model.load_state_dict(document["state_dict"])
    initial_val = validation_loss(model, val_loader, device, context)
    report = {"scope": "Small CPU pilot; no benchmark/generalization claim", "config": config,
              "manifest_sha256": manifest_hash, "torch_version": str(torch.__version__),
              "initial_validation_l1": initial_val, "epochs": [],
              "selection": "Lowest validation linear-RGB L1 among trained epochs; test not read during training",
              "input_context": context, "train_samples": len(train_data), "validation_samples": len(val_data)}
    best = float("inf")
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for source, conditions, target in train_loader:
            source, conditions, target = [value.to(device) for value in (source, conditions, target)]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(crop_prediction(model(source, conditions), target, context), target, float(config.get("gradient_weight", 0.05)))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(source)
        val = validation_loss(model, val_loader, device, context)
        if not np.isfinite(val):
            raise ValueError("Nonfinite validation loss")
        row = {"epoch": epoch, "train_objective": total / len(train_data), "validation_l1": val}
        report["epochs"].append(row)
        if val < best:
            best = val
            report["best_epoch"] = epoch
            temporary = checkpoint_path.with_suffix(".tmp")
            checkpoint_format = "prismraw-bayer-v1" if kind == "learned" else "prismraw-residual-v1"
            if kind == "learned" and architecture == "resize-v2":
                checkpoint_format = "prismraw-bayer-resize-v2"
            torch.save({"format": checkpoint_format, "base_channels": config["base_channels"],
                        "state_dict": model.state_dict(), "epoch": epoch, "validation_l1": val,
                        "manifest_sha256": manifest_hash, "config": config,
                        "input_context": context,
                        "target": "Approximate fixed rendering of clean SIDD-CC RAW"}, temporary)
            temporary.replace(checkpoint_path)
        report["elapsed_seconds"] = time.perf_counter() - started
        (output / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
    print(f"Best trained epoch: {report['best_epoch']}; checkpoint: {checkpoint_path}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    train(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
