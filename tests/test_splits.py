"""Check physical-scene grouping across camera instances and patches."""

import unittest
from smartphone_isp.data.splits import assign_scenes, validate_disjoint


class SplitTests(unittest.TestCase):
    def test_same_scene_has_one_split_across_cameras(self):
        records = [{"scene_id": scene, "camera": camera}
                   for scene in ("001", "002", "003", "004", "005") for camera in ("GP", "N6")]
        assignment = assign_scenes(records)
        rows = [{"scene_id": record["scene_id"], "split": assignment[record["scene_id"]]} for record in records]
        validate_disjoint(rows)
        self.assertEqual(assignment, assign_scenes(list(reversed(records))))

    def test_rejects_leakage(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_disjoint([{"scene_id": "001", "split": "train"},
                               {"scene_id": "001", "split": "validation"}])

    def test_requires_three_scenes(self):
        with self.assertRaises(ValueError):
            assign_scenes([{"scene_id": "001"}, {"scene_id": "002"}])
