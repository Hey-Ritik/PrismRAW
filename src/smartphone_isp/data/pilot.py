"""Prepare explicit linear-RGB pairs from a small SIDD-CC scene pilot."""

import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from smartphone_isp.data.splits import assign_scenes, validate_disjoint
from smartphone_isp.data.bayer import pack_bayer
from smartphone_isp.pipelines.classical import process, render_linear


def camera_metadata(meta):
    """Approximate row-normalized single-illuminant baseline, not a DNG profile."""
    matrix = np.asarray(meta["ColorMatrix2"], dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(matrix)) or np.any(np.abs(matrix.sum(axis=1)) < 1e-8):
        raise ValueError("Invalid camera colour matrix")
    matrix /= matrix.sum(axis=1, keepdims=True)
    xyz_to_srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                           [-0.969266, 1.8760108, 0.041556],
                           [0.0556434, -0.2040259, 1.0572252]])
    neutral = np.asarray(meta["AsShotNeutral"], dtype=np.float64)
    if neutral.shape != (3,) or not np.all(np.isfinite(neutral)) or np.any(neutral <= 0):
        raise ValueError("Invalid camera white balance")
    gains = 1 / neutral
    gains /= gains[1]
    return {"normalized": True, "bayer_pattern": str(meta["cfa_str"]).upper(),
            "white_balance": gains.tolist(),
            "camera_to_linear_srgb": (xyz_to_srgb @ np.linalg.inv(matrix)).tolist()}


def prepare(index_path, destination, patch_size=128, patches_per_scene=8, seed=42, existing_splits=None):
    if patch_size < 8 or patch_size % 2 or patches_per_scene < 1:
        raise ValueError("Use an even patch size >=8 and a positive patch count")
    index_path, destination = Path(index_path), Path(destination)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assignments = assign_scenes(index["records"], seed)
    if existing_splits:
        previous = json.loads(Path(existing_splits).read_text(encoding="utf-8"))
        assignments = {scene: previous["scenes"].get(scene, "train") for scene in assignments}
        if not set(previous["scenes"]).issubset(assignments):
            raise ValueError("Expanded dataset must retain every previous physical scene")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "split-plan.json").write_text(json.dumps({"seed": seed, "scenes": assignments}, indent=2), encoding="utf-8")
    rng = np.random.default_rng(seed)
    rows = []
    for record in sorted(index["records"], key=lambda item: item["instance"]):
        files = record["files"]
        noisy = loadmat(files["noisy"]["path"])["data"]
        clean = loadmat(files["clean"]["path"])["data"]
        if noisy.ndim != 2 or noisy.shape != clean.shape or not np.isfinite(noisy).all() or not np.isfinite(clean).all():
            raise ValueError("Invalid noisy/clean pair")
        metadata = camera_metadata(loadmat(files["metadata"]["path"], simplify_cells=True))
        extent = patch_size + 8
        if min(noisy.shape) < extent:
            raise ValueError("Sensor array is smaller than requested patch")
        for i in range(patches_per_scene):
            y = int(rng.integers((noisy.shape[0] - extent) // 2 + 1)) * 2
            x = int(rng.integers((noisy.shape[1] - extent) // 2 + 1)) * 2
            cut = (slice(y, y + extent), slice(x, x + extent))
            classical, source_stages, stats = process(noisy[cut], metadata)
            _, target_stages, _ = process(clean[cut], metadata, adaptive=False)
            # Exclude interpolation/filter border context from the stored patch.
            inner = (slice(4, -4), slice(4, -4))
            source = source_stages["linear_srgb"][inner].astype(np.float32)
            target = target_stages["linear_srgb"][inner].astype(np.float32)
            display_target = render_linear(target_stages["linear_srgb"], stats["exposure_gain"])[0][inner]
            sample_id = f"{record['instance'][:4]}-{i:03d}"
            path = destination / f"{sample_id}.npz"
            np.savez_compressed(path, source=source, target=target,
                                packed_raw=pack_bayer(source_stages["normalized_mosaic"][inner], metadata["bayer_pattern"]),
                                calibration=np.array(metadata["white_balance"] +
                                    np.asarray(metadata["camera_to_linear_srgb"]).ravel().tolist() +
                                    [float(metadata["bayer_pattern"] == p) for p in ("RGGB", "BGGR", "GRBG", "GBRG")], dtype=np.float32),
                                classical=classical[inner], target_srgb=display_target,
                                conditions=np.array([stats["noise_estimate"], stats["median_linear_luminance"]], dtype=np.float32),
                                exposure=np.float32(stats["exposure_gain"]), sharpen=np.float32(stats["sharpen_strength"]))
            rows.append({"sample_id": sample_id, "scene_id": record["scene_id"],
                         "instance": record["instance"], "split": assignments[record["scene_id"]],
                         "path": path.name, "crop_yxhw": [y+4, x+4, patch_size, patch_size]})
        print(f"Prepared {record['instance']}: {assignments[record['scene_id']]} ({patches_per_scene} patches)", flush=True)
    validate_disjoint(rows)
    manifest = {"schema": 2, "dataset": index["dataset"], "revision": index["revision"],
                "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "seed": seed, "patch_size": patch_size, "scene_assignments": assignments,
                "target": "Clean RAW rendered through fixed approximate camera transform; not captured RGB ground truth",
                "exposure": "Derived from noisy input and shared with target; no clean-target exposure selection",
                "registration": "Dataset correspondence used; independent registration validation pending",
                "scope": "Small pilot; insufficient for benchmark/generalization claims", "samples": rows}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
