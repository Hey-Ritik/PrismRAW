"""Reference metric conventions, CFA augmentation and reduced-precision inference."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from smartphone_isp.data.augmentation import augment_pair
from smartphone_isp.data.bayer import pack_bayer, unpack_bayer
from smartphone_isp.evaluation.metrics import quality
from smartphone_isp.models.raw_to_rgb import ResizeRawToRGBNet
from smartphone_isp.optimization.export import export_fp16
from smartphone_isp.pipelines.learned import load_model, reconstruct


class DeliveryTests(unittest.TestCase):
    def test_identical_reference_metrics_and_range_guard(self):
        image = np.random.default_rng(42).random((64, 64, 3), dtype=np.float32)
        metrics = quality(image, image, perceptual=False)
        self.assertIsNone(metrics["psnr"])
        self.assertAlmostEqual(metrics["ssim"], 1)
        self.assertAlmostEqual(metrics["delta_e_00"], 0)
        with self.assertRaises(ValueError):
            quality(image * 255, image, perceptual=False)

    def test_all_cfa_rotations_preserve_sensor_values_and_target_orientation(self):
        raw = np.arange(80, dtype=np.float32).reshape(8, 10)
        target = np.repeat(raw[..., None], 3, axis=-1)
        patterns = ("RGGB", "BGGR", "GRBG", "GBRG")
        for pattern_index, pattern in enumerate(patterns):
            features = np.zeros(16, dtype=np.float32)
            features[12+pattern_index] = 1
            for turns in range(4):
                for flip in (False, True):
                    source, actual_target, calibration = augment_pair(pack_bayer(raw, pattern), target, features, "learned", turns, flip)
                    expected = np.rot90(raw, turns)
                    if flip:
                        expected = np.fliplr(expected)
                    restored = unpack_bayer(source, patterns[int(np.argmax(calibration[12:]))])
                    np.testing.assert_array_equal(restored, expected)
                    np.testing.assert_array_equal(actual_target[..., 0], expected)

    def test_fp16_reduces_size_and_preserves_small_forward_pass(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = ResizeRawToRGBNet(4).eval()
        packed = np.random.default_rng(42).random((9, 11, 4), dtype=np.float32)
        calibration = np.zeros(16, dtype=np.float32)
        expected = reconstruct(model, packed, calibration)
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder)/"model.pt", Path(folder)/"half.pt"
            torch.save({"format": "prismraw-bayer-resize-v2", "epoch": 1, "base_channels": 4, "state_dict": model.state_dict()}, source)
            sizes = export_fp16(source, destination)
            self.assertLess(sizes["optimized_bytes"], sizes["original_bytes"])
            optimized, _ = load_model(destination)
            self.assertEqual(next(optimized.parameters()).dtype, torch.float16)
            np.testing.assert_allclose(reconstruct(optimized, packed, calibration), expected, atol=0.005)
