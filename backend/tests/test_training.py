"""Check actual gradient updates and trained-checkpoint inference consistency."""

from pathlib import Path
import tempfile
import unittest
import torch
from smartphone_isp.models.residual_unet import ResidualUNet
from smartphone_isp.pipelines.hybrid import load_model
from smartphone_isp.training.train import objective


class TrainingTests(unittest.TestCase):
    def test_update_and_checkpoint_roundtrip(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = ResidualUNet(4)
        source = torch.rand(1, 3, 17, 19)
        conditions = torch.zeros(1, 2, 17, 19)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        objective(model(source, conditions), source * 0.9).backward()
        optimizer.step()
        self.assertGreater(model.head.weight.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save({"format": "prismraw-residual-v1", "base_channels": 4,
                        "epoch": 1, "state_dict": model.state_dict()}, path)
            restored, _ = load_model(path)
            torch.testing.assert_close(model(source, conditions), restored(source, conditions))
