"""Compact learned-only Bayer-to-linear-RGB reconstruction with camera metadata."""

import torch
from torch import nn
from torch.nn import functional as F
from smartphone_isp.models.residual_unet import _block


class RawToRGBNet(nn.Module):
    """Packed Bayer [N,4,H/2,W/2] + 16 camera metadata maps -> RGB [N,3,H,W].

    Metadata is RGB white-balance gains, flattened camera-to-linear-sRGB matrix,
    and one-hot CFA phase (RGGB/BGGR/GRBG/GBRG). No classical demosaicing,
    white balance, colour correction, or denoising occurs in this network path.
    """

    def __init__(self, base_channels=16):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        c = base_channels
        self.enc1 = _block(20, c)
        self.enc2 = _block(c, c*2)
        self.middle = _block(c*2, c*4)
        self.dec2 = _block(c*6, c*2)
        self.dec1 = _block(c*3, c)
        self.head = nn.Sequential(nn.Conv2d(c, 12, 1), nn.PixelShuffle(2))

    def forward(self, raw, calibration):
        if raw.ndim != 4 or raw.shape[1] != 4 or min(raw.shape[-2:]) < 4:
            raise ValueError("Expected packed Bayer [N,4,H,W], H,W >= 4")
        if calibration.shape != (raw.shape[0], 16, *raw.shape[-2:]):
            raise ValueError("Expected 16 camera metadata maps matching packed input")
        x1 = self.enc1(torch.cat([raw, calibration], dim=1))
        x2 = self.enc2(F.avg_pool2d(x1, 2))
        middle = self.middle(F.avg_pool2d(x2, 2))
        y2 = self.dec2(torch.cat([F.interpolate(middle, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2], dim=1))
        y1 = self.dec1(torch.cat([F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1], dim=1))
        return self.head(y1)
