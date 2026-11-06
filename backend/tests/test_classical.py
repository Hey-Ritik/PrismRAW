"""Check Bayer semantics, normalization contracts, and complete output behavior."""

import unittest
import numpy as np
from smartphone_isp.pipelines.classical import demosaic, process


def metadata(pattern="RGGB"):
    return {"normalized": True, "bayer_pattern": pattern,
            "white_balance": [1, 1, 1], "camera_to_linear_srgb": np.eye(3).tolist()}


class ClassicalTests(unittest.TestCase):
    def test_all_cfa_patterns_preserve_constant_color(self):
        for pattern in ("RGGB", "BGGR", "GRBG", "GBRG"):
            raw = np.zeros((8, 10), dtype=np.float32)
            for i, color in enumerate(pattern):
                raw[i//2::2, i%2::2] = {"R": 0.2, "G": 0.4, "B": 0.7}[color]
            np.testing.assert_allclose(demosaic(raw, pattern),
                                       np.broadcast_to([0.2, 0.4, 0.7], (8, 10, 3)), atol=1e-6)

    def test_normalized_input_is_not_black_subtracted_again(self):
        meta = metadata()
        meta.update(black_level=100, white_level=1023)
        _, stages, _ = process(np.full((8, 8), 0.3), meta)
        np.testing.assert_allclose(stages["normalized_mosaic"], 0.3)

    def test_integer_and_normalized_paths_agree(self):
        raw = np.arange(64, dtype=np.float32).reshape(8, 8) / 100
        a, _, _ = process(raw, metadata())
        meta = metadata()
        meta.update(normalized=False, black_level=64, white_level=1064)
        b, _, _ = process(raw * 1000 + 64, meta)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_rejects_invalid_inputs_and_missing_metadata(self):
        for raw, meta in ((np.zeros((7, 8)), metadata()),
                          (np.full((8, 8), np.nan), metadata()),
                          (np.zeros((8, 8)), {})):
            with self.assertRaises(ValueError):
                process(raw, meta)

    def test_output_is_finite_display_rgb(self):
        for value in (0, 0.01, 1):
            output, _, stats = process(np.full((8, 8), value), metadata())
            self.assertEqual(output.shape, (8, 8, 3))
            self.assertTrue(np.isfinite(output).all())
            self.assertTrue((output >= 0).all() and (output <= 1).all())
            self.assertGreaterEqual(stats["processing_ms"], 0)


if __name__ == "__main__":
    unittest.main()
