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
    if document.get("precision") == "float16":
        model.half()
    model.load_state_dict(document["state_dict"])
    model.eval()
    model.input_stage = document.get("config", {}).get("hybrid_input", "linear")
    model.blend = float(document.get("blend", 1.0))
    return model, document


def refine(model, linear, noise, illumination, device="cpu"):
    source = torch.from_numpy(np.ascontiguousarray(linear.transpose(2, 0, 1))).unsqueeze(0).to(device=device, dtype=next(model.parameters()).dtype)
    conditions = torch.tensor([noise, illumination], dtype=source.dtype, device=device)
    conditions = conditions.view(1, 2, 1, 1).expand(1, 2, *linear.shape[:2])
    with torch.inference_mode():
        result = source + getattr(model, "blend", 1.0) * (model(source, conditions) - source)
    return result[0].permute(1, 2, 0).float().cpu().numpy()


def process_hybrid(raw, metadata, model, adaptive=True, device="cpu"):
    started = time.perf_counter()
    use_denoised = getattr(model, "input_stage", "linear") == "denoised"
    _, stages, stats = process(raw, metadata, adaptive=adaptive, denoise=use_denoised, render_output=False)
    source = stages["denoised_linear"] if use_denoised else stages["linear_srgb"]
    refined = refine(model, source, stats["noise_estimate"], stats["median_linear_luminance"], device)
    output, toned, encoded = render_linear(refined, stats["exposure_gain"], stats["sharpen_strength"])
    stages.update(refined_linear=refined, tone_mapped_linear=toned, encoded_srgb=encoded)
    stats.update(method="hybrid", processing_ms=(time.perf_counter() - started) * 1000,
                 timing_scope="complete CPU array pipeline; excludes model load and file I/O",
                 input_stage="denoised" if use_denoised else "linear",
                 precision=str(next(model.parameters()).dtype), blend=getattr(model, "blend", 1.0))
    return output, stages, stats
