"""Torch dataset for prepared pairs; no Bayer augmentation after packing."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from smartphone_isp.data.splits import validate_disjoint


class PairedDataset(Dataset):
    def __init__(self, manifest, split, model_kind="hybrid", hybrid_input="linear", augment=False):
        if augment and split != "train":
            raise ValueError("Augmentation is only permitted for the training split")
        self.augment = augment
        if model_kind not in {"hybrid", "learned"}:
            raise ValueError("Unknown model kind")
        self.model_kind = model_kind
        if hybrid_input not in {"linear", "denoised"}:
            raise ValueError("Unknown hybrid input stage")
        self.hybrid_input = hybrid_input
        self.root = Path(manifest).parent
        document = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if document.get("input_context", 0) not in (0, 4):
            raise ValueError("Unsupported input context in manifest")
        validate_disjoint(document["samples"])
        self.rows = [row for row in document["samples"] if row["split"] == split]
        if not self.rows:
            raise ValueError(f"Empty split: {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with np.load(self.root / self.rows[index]["path"], allow_pickle=False) as pair:
            key = "packed_raw" if self.model_kind == "learned" else "denoised" if self.hybrid_input == "denoised" else "source"
            condition_key = "calibration" if self.model_kind == "learned" else "conditions"
            source, target, features = pair[key].copy(), pair["target"].copy(), pair[condition_key].copy()
            if self.augment:
                from smartphone_isp.data.augmentation import augment_pair
                source, target, features = augment_pair(source, target, features, self.model_kind, int(np.random.randint(4)), bool(np.random.randint(2)))
            source = torch.from_numpy(source).permute(2, 0, 1)
            target = torch.from_numpy(target).permute(2, 0, 1)
            values = torch.from_numpy(features)
            conditions = values[:, None, None].expand(len(values), *source.shape[1:])
        return source, conditions, target
