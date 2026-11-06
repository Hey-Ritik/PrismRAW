"""Compact RGB refinement model for the hybrid ISP (untrained by default)."""

import torch
from torch import nn
from torch.nn import functional as F


def _block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(),
    )


class ResidualUNet(nn.Module):
    """Refine linear RGB using constant spatial noise and illumination maps.

    Inputs: RGB [N,3,H,W], condition maps [N,2,H,W]. No output clamping:
    the rendering pipeline owns gamut handling and output encoding.
    """

    def __init__(self, base_channels=16):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        c = base_channels
        self.enc1 = _block(5, c)
        self.enc2 = _block(c, c * 2)
        self.bottleneck = _block(c * 2, c * 4)
        self.dec2 = _block(c * 6, c * 2)
        self.dec1 = _block(c * 3, c)
        self.head = nn.Conv2d(c, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, rgb, conditions):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 4:
            raise ValueError("Expected RGB [N,3,H,W] with H,W >= 4")
        if conditions.shape != (rgb.shape[0], 2, *rgb.shape[-2:]):
            raise ValueError("Expected condition maps [N,2,H,W] matching RGB")
        x1 = self.enc1(torch.cat([rgb, conditions], dim=1))
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        x3 = self.bottleneck(F.avg_pool2d(x2, 2))
        y2 = self.dec2(torch.cat([F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2], dim=1))
        y1 = self.dec1(torch.cat([F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1], dim=1))
        return rgb + self.head(y1)
