"""Torch dataset for prepared pairs; no Bayer augmentation after packing."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from smartphone_isp.data.splits import validate_disjoint


class PairedDataset(Dataset):
    def __init__(self, manifest, split):
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
            source = torch.from_numpy(pair["source"].copy()).permute(2, 0, 1)
            target = torch.from_numpy(pair["target"].copy()).permute(2, 0, 1)
            conditions = torch.from_numpy(pair["conditions"].copy())[:, None, None].expand(2, *source.shape[1:])
        return source, conditions, target
