"""Deterministic physical-scene separation, performed before patch extraction."""

import random


def assign_scenes(records, seed=42):
    scenes = sorted({record["scene_id"] for record in records})
    if len(scenes) < 3:
        raise ValueError("At least three physical scenes are needed for train/val/test")
    random.Random(seed).shuffle(scenes)
    holdout = max(1, len(scenes) // 5)
    return {scene: ("test" if i < holdout else "validation" if i < holdout * 2 else "train")
            for i, scene in enumerate(scenes)}


def validate_disjoint(rows):
    seen = {}
    for row in rows:
        scene, split = row["scene_id"], row["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if scene in seen and seen[scene] != split:
            raise ValueError(f"Scene leakage: {scene}")
        seen[scene] = split
    if set(seen.values()) != {"train", "validation", "test"}:
        raise ValueError("Manifest must include train, validation, and test scenes")
