"""Spatial augmentation that updates Bayer phase and semantic channel packing."""

import numpy as np
from smartphone_isp.data.bayer import pack_bayer, unpack_bayer

PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")


def augment_pair(source, target, features, kind, turns, flip):
    features = features.copy()
    if kind == "learned":
        pattern = PATTERNS[int(np.argmax(features[12:]))]
        raw = unpack_bayer(source, pattern)
        phase = np.array(list(pattern)).reshape(2, 2)
        raw, phase = np.rot90(raw, turns), np.rot90(phase, turns)
        if flip:
            raw, phase = np.fliplr(raw), np.fliplr(phase)
        pattern = "".join(phase.ravel())
        source = pack_bayer(raw, pattern)
        features[12:] = [float(pattern == p) for p in PATTERNS]
    else:
        source = np.rot90(source, turns)
        if flip:
            source = np.fliplr(source)
    target = np.rot90(target, turns)
    if flip:
        target = np.fliplr(target)
    return np.ascontiguousarray(source), np.ascontiguousarray(target), features
