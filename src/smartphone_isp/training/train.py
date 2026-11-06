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


def objective(prediction, target):
    reconstruction = F.l1_loss(prediction, target)
    dx = F.l1_loss(prediction[..., 1:] - prediction[..., :-1], target[..., 1:] - target[..., :-1])
    dy = F.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
    return reconstruction + 0.05 * (dx + dy)


def validation_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for source, conditions, target in loader:
            source, conditions, target = [value.to(device) for value in (source, conditions, target)]
            total += F.l1_loss(model(source, conditions), target).item() * len(source)
            count += len(source)
    return total / count


def train(config):
    for key in ("epochs", "batch_size", "base_channels", "threads"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 < float(config["learning_rate"]) < 1:
        raise ValueError("Invalid learning rate")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config["threads"]))
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    train_data = PairedDataset(config["manifest"], "train")
    val_data = PairedDataset(config["manifest"], "validation")
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], num_workers=0)
    model = ResidualUNet(config["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(Path(config["manifest"]).read_bytes()).hexdigest()
    initial_val = validation_loss(model, val_loader, device)
    report = {"scope": "Small CPU pilot; no benchmark/generalization claim", "config": config,
              "manifest_sha256": manifest_hash, "torch_version": str(torch.__version__),
              "initial_validation_l1": initial_val, "epochs": [],
              "selection": "Lowest validation linear-RGB L1 among trained epochs; test not read during training",
              "train_samples": len(train_data), "validation_samples": len(val_data)}
    best = float("inf")
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for source, conditions, target in train_loader:
            source, conditions, target = [value.to(device) for value in (source, conditions, target)]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(source, conditions), target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(source)
        val = validation_loss(model, val_loader, device)
        if not np.isfinite(val):
            raise ValueError("Nonfinite validation loss")
        row = {"epoch": epoch, "train_objective": total / len(train_data), "validation_l1": val}
        report["epochs"].append(row)
        if val < best:
            best = val
            report["best_epoch"] = epoch
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({"format": "prismraw-residual-v1", "base_channels": config["base_channels"],
                        "state_dict": model.state_dict(), "epoch": epoch, "validation_l1": val,
                        "manifest_sha256": manifest_hash, "config": config,
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
