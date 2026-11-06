"""Export reduced-precision weights without claiming a latency improvement."""

import argparse
from pathlib import Path
import torch


def export_fp16(source, destination):
    document = torch.load(source, map_location="cpu", weights_only=True)
    document["state_dict"] = {key: value.half() if value.is_floating_point() else value for key, value in document["state_dict"].items()}
    document["precision"] = "float16"
    document["optimization"] = "Reduced-precision weights and inference; validate on target runtime"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(document, destination)
    return {"original_bytes": Path(source).stat().st_size, "optimized_bytes": destination.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(export_fp16(args.source, args.destination))


if __name__ == "__main__":
    main()
