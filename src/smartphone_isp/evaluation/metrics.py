"""Reference metrics with explicit sRGB/D65 conventions."""

from functools import lru_cache
import numpy as np
import torch
from skimage.metrics import structural_similarity
from skimage.color import rgb2lab, deltaE_ciede2000


@lru_cache(maxsize=1)
def perceptual_model():
    import lpips
    return lpips.LPIPS(net="squeeze", version="0.1", verbose=False).eval()


def quality(prediction, target, perceptual=True):
    prediction, target = [np.asarray(value, dtype=np.float32) for value in (prediction, target)]
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("Metrics require matching HxWx3 RGB arrays")
    if min(prediction.shape[:2]) < 8 or not all(np.isfinite(v).all() for v in (prediction, target)):
        raise ValueError("Metrics require finite images at least 8x8")
    if any(v.min() < 0 or v.max() > 1 for v in (prediction, target)):
        raise ValueError("Metric sRGB inputs must be in [0,1]")
    mse = float(np.mean((prediction.astype(np.float64) - target) ** 2))
    result = {"psnr": None if mse == 0 else -10 * np.log10(mse),
              "ssim": float(structural_similarity(prediction, target, channel_axis=-1, data_range=1.0)),
              "delta_e_00": float(np.mean(deltaE_ciede2000(rgb2lab(prediction, illuminant="D65"), rgb2lab(target, illuminant="D65"))))}
    if perceptual:
        if min(prediction.shape[:2]) < 64:
            raise ValueError("LPIPS-Squeeze evaluation requires at least 64x64")
        tensors = [torch.from_numpy(np.ascontiguousarray(v.transpose(2, 0, 1))).unsqueeze(0) * 2 - 1 for v in (prediction, target)]
        with torch.inference_mode():
            result["lpips"] = float(perceptual_model()(*tensors).item())
    return result
