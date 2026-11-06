"""Hybrid inference using an explicitly trained local refinement checkpoint."""

import time
import numpy as np
import torch
from smartphone_isp.models.residual_unet import ResidualUNet
from smartphone_isp.pipelines.classical import process, render_linear


def load_model(checkpoint, device="cpu"):
    document = torch.load(checkpoint, map_location=device, weights_only=True)
    if document.get("format") != "prismraw-residual-v1" or document.get("epoch", 0) < 1:
        raise ValueError("Expected a trained PrismRAW residual checkpoint")
    model = ResidualUNet(document["base_channels"]).to(device)
    model.load_state_dict(document["state_dict"])
    model.eval()
    return model, document


def refine(model, linear, noise, illumination, device="cpu"):
    source = torch.from_numpy(np.ascontiguousarray(linear.transpose(2, 0, 1))).unsqueeze(0).to(device)
    conditions = torch.tensor([noise, illumination], dtype=source.dtype, device=device)
    conditions = conditions.view(1, 2, 1, 1).expand(1, 2, *linear.shape[:2])
    with torch.inference_mode():
        result = model(source, conditions)
    return result[0].permute(1, 2, 0).cpu().numpy()


def process_hybrid(raw, metadata, model, adaptive=True, device="cpu"):
    started = time.perf_counter()
    _, stages, stats = process(raw, metadata, adaptive=adaptive)
    refined = refine(model, stages["linear_srgb"], stats["noise_estimate"], stats["median_linear_luminance"], device)
    output, toned, encoded = render_linear(refined, stats["exposure_gain"], stats["sharpen_strength"])
    stages.update(refined_linear=refined, tone_mapped_linear=toned, encoded_srgb=encoded)
    stats.update(method="hybrid", processing_ms=(time.perf_counter() - started) * 1000,
                 timing_scope="array processing including redundant classical denoising; excludes model load and file I/O",
                 limitation="Pilot checkpoint; classical preparation still computes an unused bilateral result")
    return output, stages, stats
