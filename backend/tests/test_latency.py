"""Preserve denoising arithmetic and hybrid output while skipping unused work."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from smartphone_isp.pipelines.classical import bilateral, process, render_linear
from smartphone_isp.pipelines.hybrid import process_hybrid, refine
from smartphone_isp.models.residual_unet import ResidualUNet


def reference_bilateral(image, sigma):
    if sigma <= 0:
        return image.copy()
    h,w=image.shape[:2]
    padded=np.pad(image,((2,2),(2,2),(0,0)),mode='reflect')
    numerator=np.zeros_like(image)
    denominator=np.zeros((h,w,1),dtype=np.float32)
    for y in range(5):
        for x in range(5):
            neighbor=padded[y:y+h,x:x+w]
            distance=np.mean((neighbor-image)**2,axis=-1,keepdims=True)
            weight=np.exp(-distance/(2*sigma**2)-((y-2)**2+(x-2)**2)/4)
            numerator+=weight*neighbor
            denominator+=weight
    return numerator/denominator


class LatencyTests(unittest.TestCase):
    def test_reused_buffers_preserve_bilateral_results(self):
        rng=np.random.default_rng(42)
        for dtype in (np.float32,np.float64):
            image=rng.random((17,23,3)).astype(dtype)
            for sigma in (0,0.005,0.03,0.08):
                np.testing.assert_array_equal(bilateral(image,sigma),reference_bilateral(image,sigma))

    def test_hybrid_skips_first_render_and_preserves_final_image(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model=ResidualUNet(4).eval()
        model.input_stage='denoised'
        raw=np.random.default_rng(42).random((32,40),dtype=np.float32)
        metadata={'normalized':True,'bayer_pattern':'RGGB','white_balance':[1,1,1],
                  'camera_to_linear_srgb':np.eye(3).tolist()}
        _,stages,stats=process(raw,metadata)
        expected=render_linear(refine(model,stages['denoised_linear'],stats['noise_estimate'],stats['median_linear_luminance']),stats['exposure_gain'],stats['sharpen_strength'])[0]
        with patch('smartphone_isp.pipelines.classical.render_linear',wraps=render_linear) as unused:
            actual,new_stages,new_stats=process_hybrid(raw,metadata,model)
            unused.assert_not_called()
        np.testing.assert_array_equal(actual,expected)
        self.assertIsNotNone(new_stages['encoded_srgb'])
        self.assertEqual(new_stats['shape'],list(actual.shape))
