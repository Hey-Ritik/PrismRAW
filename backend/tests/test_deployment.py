"""Hosting port, crop controls, and phase-preserving DNG decode boundaries."""

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from smartphone_isp.app.engine import crop_sensor
from smartphone_isp.app.settings import runtime_settings
from smartphone_isp.data.dng import load_dng


class DeploymentTests(unittest.TestCase):
    def test_port_environment_and_explicit_override(self):
        spec = importlib.util.spec_from_file_location("serve_test", Path("scripts/serve.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(os.environ, {"PORT": "10000"}):
            self.assertEqual(module.parse_args([]).port, 10000)
            self.assertEqual(module.parse_args(["--port", "7861"]).port, 7861)

    def test_hosted_settings_and_even_bayer_crop(self):
        with patch.dict(os.environ, {"PRISMRAW_MAX_CROP": "256", "PRISMRAW_TORCH_THREADS": "1", "PRISMRAW_HOSTED": "1"}):
            settings = runtime_settings()
            self.assertTrue(settings.hosted)
            self.assertEqual(settings.threads, 1)
            raw = np.arange(300*400).reshape(300, 400)
            cropped, (y, x, h, w) = crop_sensor(raw)
            self.assertEqual(cropped.shape, (256, 256))
            self.assertEqual((y % 2, x % 2), (0, 0))
            np.testing.assert_array_equal(cropped, raw[y:y+h, x:x+w])

    def test_invalid_runtime_settings_fail_early(self):
        for setting in ({"PRISMRAW_MAX_CROP": "255"}, {"PRISMRAW_TORCH_THREADS": "0"}):
            with patch.dict(os.environ, setting), self.assertRaises(ValueError):
                runtime_settings()

    def test_cropped_dng_matches_full_sensor_normalization(self):
        camera = MagicMock()
        camera.__enter__.return_value = camera
        camera.raw_pattern = np.array([[0, 1], [3, 2]])
        camera.raw_image_visible = np.arange(120*160, dtype=np.uint16).reshape(120, 160) + 128
        camera.raw_colors_visible = np.tile(camera.raw_pattern, (60, 80))
        camera.color_desc = b"RGBG"
        camera.black_level_per_channel = [32, 64, 32, 64]
        camera.camera_white_level_per_channel = [65535]*4
        camera.camera_whitebalance = [2, 1, 1.5, 1]
        camera.rgb_xyz_matrix = np.vstack([np.eye(3), np.zeros((1, 3))])
        camera.sizes.flip = 0
        with patch("rawpy.imread", return_value=camera):
            full, _ = load_dng("fixture.dng")
            cropped, metadata = load_dng("fixture.dng", max_side=64)
        y, x, h, w = metadata["sensor_crop_yxhw"]
        self.assertEqual(cropped.shape, (64, 64))
        self.assertEqual(metadata["original_sensor_shape"], [120, 160])
        self.assertEqual((y % 2, x % 2), (0, 0))
        np.testing.assert_array_equal(cropped, full[y:y+h, x:x+w])
