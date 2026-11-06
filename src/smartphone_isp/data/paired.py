"""Torch dataset for prepared pairs; no Bayer augmentation after packing."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from smartphone_isp.data.splits import validate_disjoint


class PairedDataset(Dataset):
    def __init__(self, manifest, split, model_kind="hybrid"):
        if model_kind not in {"hybrid", "learned"}:
            raise ValueError("Unknown model kind")
        self.model_kind = model_kind
        self.root = Path(manifest).parent
        document = json.loads(Path(manifest).read_text(encoding="utf-8"))
        validate_disjoint(document["samples"])
        self.rows = [row for row in document["samples"] if row["split"] == split]
        if not self.rows:
            raise ValueError(f"Empty split: {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with np.load(self.root / self.rows[index]["path"], allow_pickle=False) as pair:
            key = "packed_raw" if self.model_kind == "learned" else "source"
            source = torch.from_numpy(pair[key].copy()).permute(2, 0, 1)
            target = torch.from_numpy(pair["target"].copy()).permute(2, 0, 1)
            condition_key = "calibration" if self.model_kind == "learned" else "conditions"
            values = torch.from_numpy(pair[condition_key].copy())
            conditions = values[:, None, None].expand(len(values), *source.shape[1:])
        return source, conditions, target
