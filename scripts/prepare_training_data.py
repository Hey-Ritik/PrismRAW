"""Prepare a scene-separated paired linear-RGB training pilot."""

import argparse
from smartphone_isp.data.pilot import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/raw/sidd-cc-subset/pilot-index.json")
    parser.add_argument("--output", default="data/processed/pilot")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-scene", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.index, args.output, args.patch_size, args.patches_per_scene, args.seed)
    print(f"Prepared {len(manifest['samples'])} patches; splits: {manifest['scene_assignments']}")


if __name__ == "__main__":
    main()
