"""Context alignment guards and versioned learned-decoder checkpoint loading."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from smartphone_isp.models.raw_to_rgb import ResizeRawToRGBNet
from smartphone_isp.pipelines.learned import load_model, reconstruct
from smartphone_isp.training.alignment import crop_prediction


class ContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_context_crop_trains_only_central_output(self):
        prediction = torch.ones(1, 3, 16, 18, requires_grad=True)
        target = torch.zeros(1, 3, 8, 10)
        center = crop_prediction(prediction, target, 4)
        self.assertEqual(center.shape, target.shape)
        (center - target).abs().mean().backward()
        self.assertEqual(prediction.grad[..., :4, :].abs().sum().item(), 0)
        self.assertGreater(prediction.grad[..., 4:-4, 4:-4].abs().sum().item(), 0)

    def test_rejects_implicit_shape_broadcasting(self):
        with self.assertRaises(ValueError):
            crop_prediction(torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 8, 8), 0)
        with self.assertRaises(ValueError):
            crop_prediction(torch.zeros(1, 1, 8, 8), torch.zeros(1, 3, 8, 8), 0)

    def test_resize_checkpoint_roundtrip_and_full_resolution(self):
        torch.manual_seed(42)
        model = ResizeRawToRGBNet(4).eval()
        packed = np.full((9, 11, 4), 0.2, dtype=np.float32)
        calibration = np.zeros(16, dtype=np.float32)
        expected = reconstruct(model, packed, calibration)
        self.assertEqual(expected.shape, (18, 22, 3))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "v2.pt"
            torch.save({"format": "prismraw-bayer-resize-v2", "epoch": 1, "base_channels": 4,
                        "state_dict": model.state_dict()}, path)
            restored, _ = load_model(path)
            np.testing.assert_allclose(reconstruct(restored, packed, calibration), expected)

    def test_resize_head_does_not_introduce_periodic_bias_on_flat_features(self):
        model = ResizeRawToRGBNet(4)
        result = model.head(torch.ones(1, 4, 12, 14))[..., 4:-4, 4:-4]
        torch.testing.assert_close(result, result[..., :1, :1].expand_as(result))
