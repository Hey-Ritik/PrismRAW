# Dataset contract

Keep downloaded data in `raw/` and derived caches in `processed/`. Do not commit datasets.

Create a manifest with columns:

```csv
sample_id,scene_id,camera_id,split,raw_path,target_rgb_path,metadata_path
```

Metadata must record CFA pattern, whether black subtraction and normalization have already occurred, black/white levels when applicable, white-balance gains, matrix convention, reference colour encoding, and alignment/crop information.

SIDD MAT arrays are already normalized and black-level-subtracted. Verify the actual chosen dataset variant rather than reusing a DNG preprocessing path blindly. Support MAT storage variants explicitly when building the loader.

Group samples by scene before splitting and cropping. Fit preprocessing/calibration only on training data. Never synthesize a Bayer mosaic from a JPEG and call it captured RAW.
