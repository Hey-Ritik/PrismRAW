"""Optional model loading and array lifetime boundaries during comparison."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import weakref

import numpy as np

from smartphone_isp.app import engine, server


class AppMemoryTests(unittest.TestCase):
    def test_warmup_keeps_lpips_but_does_not_load_fp16(self):
        checkpoints = {name: MagicMock() for name in ("learned", "hybrid", "optimized")}
        with patch.object(server, "CHECKPOINTS", checkpoints), \
             patch.object(server, "available_models", return_value={"classical": True, **dict.fromkeys(checkpoints, True)}), \
             patch.object(server, "cached_model") as loader, \
             patch("smartphone_isp.evaluation.metrics.perceptual_model") as perceptual:
            server.warmup()
        self.assertEqual([call.args[0] for call in loader.call_args_list], ["learned", "hybrid"])
        perceptual.assert_called_once_with()

    def test_buffers_released_between_methods_and_fp16_not_cached(self):
        original_process = engine.process
        classical_refs, learned_refs, hybrid_refs = [], [], []

        def classical(*args, **kwargs):
            output, stages, stats = original_process(*args, **kwargs)
            classical_refs.extend(weakref.ref(value) for value in [output, *stages.values()])
            return output, stages, stats

        def learned(*args):
            self.assertTrue(all(ref() is None for ref in classical_refs))
            display = np.full((16, 16, 3), 0.2, dtype=np.float32)
            linear = display.copy()
            learned_refs.extend([weakref.ref(display), weakref.ref(linear)])
            return display, {"predicted_linear": linear}, {}

        def hybrid(*args, **kwargs):
            self.assertTrue(all(ref() is None for ref in learned_refs + hybrid_refs))
            output = np.full((16, 16, 3), 0.3, dtype=np.float32)
            stage = output.copy()
            hybrid_refs.extend([weakref.ref(output), weakref.ref(stage)])
            return output, {"unused": stage}, {}

        metadata = {"normalized": True, "bayer_pattern": "RGGB", "white_balance": [1, 1, 1],
                    "camera_to_linear_srgb": np.eye(3).tolist()}
        with tempfile.TemporaryDirectory() as folder:
            checkpoints = {name: Path(folder) / f"{name}.pt" for name in ("learned", "hybrid", "optimized")}
            for checkpoint in checkpoints.values():
                checkpoint.touch()
            with patch.object(engine, "RESULTS", Path(folder) / "results"), \
                 patch.object(engine, "CHECKPOINTS", checkpoints), \
                 patch.object(engine, "process", side_effect=classical), \
                 patch.object(engine, "process_learned", side_effect=learned), \
                 patch.object(engine, "process_hybrid", side_effect=hybrid), \
                 patch.object(engine, "cached_model", return_value=object()) as cached, \
                 patch.object(engine, "load_hybrid", return_value=(object(), {})) as fp16:
                report = engine.compare(np.full((16, 16), 0.2, dtype=np.float32), metadata)
            self.assertEqual([call.args[0] for call in cached.call_args_list], ["learned", "hybrid"])
            fp16.assert_called_once_with(checkpoints["optimized"])
            self.assertEqual(set(report["processing_ms"]), {"classical", "learned", "hybrid", "optimized"})
            self.assertTrue(all(ref() is None for ref in hybrid_refs))
