"""Learned RAW shape, calibration, checkpoint, and separation contracts."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from smartphone_isp.models.raw_to_rgb import RawToRGBNet
from smartphone_isp.pipelines.learned import calibration_vector, load_model, process_learned


class LearnedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.metadata = {"normalized": True, "bayer_pattern": "GRBG",
                         "white_balance": [1.1, 1, 1.2], "camera_to_linear_srgb": np.eye(3).tolist()}

    def test_output_reconstructs_full_resolution_and_gradients(self):
        model = RawToRGBNet(4)
        raw = torch.rand(1, 4, 9, 11)
        calibration = torch.zeros(1, 16, 9, 11)
        result = model(raw, calibration)
        self.assertEqual(result.shape, (1, 3, 18, 22))
        result.square().mean().backward()
        self.assertGreater(model.head[0].weight.grad.abs().sum().item(), 0)

    def test_sensor_metadata_keeps_cfa_phase(self):
        for index, pattern in enumerate(("RGGB", "BGGR", "GRBG", "GBRG")):
            self.metadata["bayer_pattern"] = pattern
            vector = calibration_vector(self.metadata)
            self.assertEqual(vector.shape, (16,))
            self.assertEqual(vector[12:].sum(), 1)
            self.assertEqual(vector[12+index], 1)

    def test_checkpoint_and_inference_do_not_call_classical_isp(self):
        model = RawToRGBNet(4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.pt"
            torch.save({"format": "prismraw-bayer-v1", "epoch": 1, "base_channels": 4,
                        "state_dict": model.state_dict()}, path)
            restored, _ = load_model(path)
            with patch("smartphone_isp.pipelines.classical.process", side_effect=AssertionError("Classical ISP called")):
                output, stages, stats = process_learned(np.full((18, 22), 0.2), self.metadata, restored)
            self.assertEqual(output.shape, (18, 22, 3))
            self.assertTrue(np.isfinite(output).all())
            self.assertEqual(stats["method"], "learned")
            self.assertEqual(stages["predicted_linear"].shape, output.shape)

    def test_rejects_odd_sensor_shape(self):
        with self.assertRaises(ValueError):
            process_learned(np.zeros((17, 18)), self.metadata, RawToRGBNet(4))
