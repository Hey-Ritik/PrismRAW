# Dataset contract

Keep downloaded data in `raw/` and derived caches in `processed/`. Do not commit datasets.

Create a manifest with columns:

```csv
sample_id,scene_id,camera_id,split,raw_path,target_rgb_path,metadata_path
```

Metadata must record CFA pattern, whether black subtraction and normalization have already occurred, black/white levels when applicable, white-balance gains, matrix convention, reference colour encoding, and alignment/crop information.

SIDD MAT arrays are already normalized and black-level-subtracted. Verify the actual chosen dataset variant rather than reusing a DNG preprocessing path blindly. Support MAT storage variants explicitly when building the loader.

Group samples by scene before splitting and cropping. Fit preprocessing/calibration only on training data. Never synthesize a Bayer mosaic from a JPEG and call it captured RAW.

## Prepared pilot schema

The training preparation command writes a JSON manifest, not the legacy development CSV. Schema 2 contains scene assignments, dataset revision, index checksum, target-rendering convention, and sample paths relative to the manifest directory. `split-plan.json` is saved before extraction. `--existing-splits` preserves all previous scene assignments and sends added scenes to training; do not reshuffle holdouts after inspecting results.

Each NPZ includes linear noisy input (`source`), linear clean target (`target`), normalized packed Bayer (`packed_raw`), 16 camera features (`calibration`), noise/illumination (`conditions`), and comparison-render inputs/outputs. CFA packing retains semantic R/G/G/B plus explicit pattern phase in calibration. Training currently applies no augmentation. Noise and display exposure come from noisy input; no target-derived exposure is supplied to a model.

Legacy schema 2 uses central neural inputs while classical processing receives a four-pixel RAW border. New `--input-context 4` preparation writes schema 3: for a 128x128 target, hybrid source is 136x136 linear RGB, learned-only source is 68x68 packed Bayer from the same 136x136 sensor crop, and both produce 136x136 RGB predictions. Training and evaluation remove exactly four output pixels on each side. Shape mismatches are rejected rather than broadcast. Classical output and targets remain 128x128. All methods now have the same available RAW boundary context.

Schema 3 adds `input_context` and `context_protocol` to the manifest. Existing recipes remain available with `--input-context 0`; do not mix checkpoints and manifests. Pair dimensions and finite values are validated, while independent geometric registration remains pending.
