"""Command-line environment checks and explicit-metadata ISP inference."""

import argparse
import importlib.util
import platform
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Smartphone ISP project utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    for mode in ("classical", "hybrid", "learned"):
        command = commands.add_parser(mode, help=f"Process a Bayer array using the {mode} ISP")
        command.add_argument("--input", required=True, type=Path)
        command.add_argument("--metadata", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--key", help="Array variable in MAT/NPZ")
        if mode != "learned":
            command.add_argument("--fixed", action="store_true", help="Disable adaptive parameter selection")
        if mode in ("hybrid", "learned"):
            command.add_argument("--checkpoint", required=True, type=Path)
        if mode == "learned":
            command.add_argument("--exposure", type=float, default=1.0, help="Fixed display exposure after learned RAW reconstruction")
    args = parser.parse_args()
    if args.command in ("classical", "hybrid", "learned"):
        try:
            run_inference(args)
        except (ValueError, KeyError, OSError, ImportError) as exc:
            parser.error(str(exc))
        return
    print(f"Python: {platform.python_version()}")
    for name in ("numpy", "cv2", "rawpy", "scipy", "torch", "gradio", "skimage", "lpips"):
        status = "available" if importlib.util.find_spec(name) else "not installed"
        print(f"{name}: {status}")
    print("Status: classical, learned, and hybrid array ISP available; full evaluation pending.")


def run_inference(args):
    import numpy as np
    from PIL import Image
    from smartphone_isp.data.arrays import load_raw
    from smartphone_isp.pipelines.classical import process

    raw = load_raw(args.input, args.key)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if args.command == "learned":
        import torch
        from smartphone_isp.pipelines.learned import load_model, process_learned
        torch.set_num_threads(2)
        model, checkpoint = load_model(args.checkpoint)
        output, stages, stats = process_learned(raw, metadata, model, args.exposure)
        stats.update(checkpoint=str(args.checkpoint), trained_epoch=checkpoint["epoch"])
    elif args.command == "hybrid":
        import torch
        from smartphone_isp.pipelines.hybrid import load_model, process_hybrid
        torch.set_num_threads(2)
        model, checkpoint = load_model(args.checkpoint)
        output, stages, stats = process_hybrid(raw, metadata, model, adaptive=not args.fixed)
        stats["checkpoint"] = str(args.checkpoint)
        stats["trained_epoch"] = checkpoint["epoch"]
    else:
        output, stages, stats = process(raw, metadata, adaptive=not args.fixed)
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(output * 255).astype(np.uint8)).save(args.output / "output.png")
    np.savez_compressed(args.output / "intermediates.npz", **stages)
    stats["input"] = str(args.input)
    stats["metadata"] = metadata
    stats["input_description"] = "Unverified sensor array unless dataset provenance is documented"
    (args.output / "processing.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved {args.command} output and intermediates to {args.output}")
    print(f"Processing: {stats['processing_ms']:.1f} ms (excludes file I/O)")


if __name__ == "__main__":
    main()
