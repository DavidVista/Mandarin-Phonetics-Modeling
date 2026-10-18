import importlib
import sys
import types
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise unittest.SkipTest(f"scientific evaluation dependencies unavailable: {error}")

SOURCE = Path(__file__).resolve().parents[1] / "src" / "phonetics"
phonetics_package = types.ModuleType("phonetics")
phonetics_package.__path__ = [str(SOURCE)]
evaluation_package = types.ModuleType("phonetics.evaluation")
evaluation_package.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics_package)
sys.modules.setdefault("phonetics.evaluation", evaluation_package)
module = importlib.import_module("phonetics.evaluation.representations")


class RepresentationTests(unittest.TestCase):
    def test_occurrence_weighted_group_mean_and_order(self):
        ids = ["a", "b", "c"]
        hidden = np.array([[1, 0], [3, 0], [8, 2]], dtype=np.float32)
        groups = [
            {"group_id": "z", "group_kind": "syllable", "occurrence_ids": ["c"]},
            {"group_id": "g", "group_kind": "syllable", "occurrence_ids": ["a", "b", "c"]},
        ]
        group_ids, means, counts = module.aggregate_occurrences(ids, hidden, groups, "syllable")
        self.assertEqual(group_ids, ["g", "z"])
        self.assertEqual(counts, [3, 1])
        np.testing.assert_allclose(means[0], [4, 2 / 3])

    def test_missing_member_and_zero_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            module.validate_matrix(["a"], np.zeros((1, 2), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "missing occurrence"):
            module.aggregate_occurrences(
                ["a"], np.ones((1, 2), dtype=np.float32),
                [{"group_id": "g", "group_kind": "syllable", "occurrence_ids": ["missing"]}], "syllable",
            )

    def test_random_baseline_stable_and_aligned(self):
        first = module.random_occurrence_baseline(["a", "b"], dimension=5, seed=42)
        second = module.random_occurrence_baseline(["a", "b"], dimension=5, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
