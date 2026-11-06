"""Command-line environment checks and explicit-metadata classical inference."""

import argparse
import importlib.util
import platform
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Smartphone ISP project utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    classical = commands.add_parser("classical", help="Process a Bayer array with JSON metadata")
    classical.add_argument("--input", required=True, type=Path)
    classical.add_argument("--metadata", required=True, type=Path)
    classical.add_argument("--output", required=True, type=Path)
    classical.add_argument("--key", help="Array variable in MAT/NPZ")
    classical.add_argument("--fixed", action="store_true", help="Disable adaptive parameter selection")
    args = parser.parse_args()
    if args.command == "classical":
        try:
            run_classical(args)
        except (ValueError, KeyError, OSError, ImportError) as exc:
            parser.error(str(exc))
        return
    print(f"Python: {platform.python_version()}")
    for name in ("numpy", "cv2", "rawpy", "scipy", "torch", "gradio", "skimage", "lpips"):
        status = "available" if importlib.util.find_spec(name) else "not installed"
        print(f"{name}: {status}")
    print("Status: classical array ISP available; real-data validation and learned inference pending.")


def run_classical(args):
    import numpy as np
    from PIL import Image
    from smartphone_isp.data.arrays import load_raw
    from smartphone_isp.pipelines.classical import process

    raw = load_raw(args.input, args.key)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    output, stages, stats = process(raw, metadata, adaptive=not args.fixed)
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(output * 255).astype(np.uint8)).save(args.output / "output.png")
    np.savez_compressed(args.output / "intermediates.npz", **stages)
    stats["input"] = str(args.input)
    stats["metadata"] = metadata
    stats["input_description"] = "Unverified sensor array unless dataset provenance is documented"
    (args.output / "processing.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved classical output and intermediates to {args.output}")
    print(f"Processing: {stats['processing_ms']:.1f} ms (excludes file I/O)")


if __name__ == "__main__":
    main()
