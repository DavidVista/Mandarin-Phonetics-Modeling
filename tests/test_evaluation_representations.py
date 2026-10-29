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

    def test_stable_norm_accepts_tiny_and_huge_finite_rows(self):
        module.validate_matrix(["a", "b"], np.array([[1e-300, 0.0], [1e300, 1e300]]))
        with self.assertRaisesRegex(ValueError, "zero-norm occurrences.*a"):
            module.validate_matrix(["a"], np.zeros((1, 2)))
        with self.assertRaisesRegex(ValueError, "non-finite occurrences.*b"):
            module.validate_matrix(["a", "b"], np.array([[1.0, 0.0], [np.inf, 1.0]]))

    def test_contextual_roundtrip_is_raw_hash_checked_and_immutable(self):
        import tempfile
        provenance = {"model_id": "components", "seed": 7, "checkpoint_sha256": "a" * 64,
                      "bundle_manifest_sha256": "b" * 64,
                      "experiment_hashes": {name: f"{i + 1:x}" * 64 for i, name in enumerate(
                          ("manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json", "acoustic_targets.npy"))},
                      "config_sha256": "2" * 64, "vocab_sha256": "3" * 64,
                      "environment": {"python": "3.12", "numpy": "2.3.5", "torch": "2.6.0",
                                      "platform": "test-platform", "device": "cpu"},
                      "source_sha256": "f" * 64, "evaluation_mode": True}
        hidden = np.array([[1e-30, 4.0], [3.0, 8.0]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "run"
            module.save_contextual_representations(target, occurrence_ids=["a", "b"], hidden=hidden,
                                                   provenance=provenance)
            loaded = module.load_contextual_representations(target)
            np.testing.assert_array_equal(loaded["hidden"], hidden)
            self.assertEqual(loaded["manifest"]["normalization"], "none")
            with self.assertRaises(FileExistsError):
                module.save_contextual_representations(target, occurrence_ids=["a", "b"], hidden=hidden,
                                                       provenance=provenance)
            (target / "occurrence_hidden.npy").write_bytes((target / "occurrence_hidden.npy").read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "checksum"):
                module.load_contextual_representations(target)

    def test_contextual_writer_and_loader_reject_bad_provenance_and_manifest(self):
        import tempfile
        provenance = {"model_id": "components", "seed": 1, "checkpoint_sha256": "a" * 64,
                      "bundle_manifest_sha256": "b" * 64,
                      "experiment_hashes": {name: f"{i + 1:x}" * 64 for i, name in enumerate(
                          ("manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json", "acoustic_targets.npy"))},
                      "config_sha256": "2" * 64, "vocab_sha256": "3" * 64,
                      "environment": {"python": "3.12", "numpy": "2.3.5", "torch": "2.6.0",
                                      "platform": "test-platform", "device": "cpu"},
                      "source_sha256": "f" * 64, "evaluation_mode": True}
        hidden = np.array([[1.0, 2.0]], dtype=np.float32)
        invalids = []
        for key, value in (("model_id", None), ("checkpoint_sha256", None), ("source_sha256", "z" * 64),
                           ("seed", True), ("evaluation_mode", False), ("environment", None),
                           ("experiment_hashes", {"config.json": "c" * 64})):
            invalids.append((key, value))
        for key, value in invalids:
            bad = {**provenance, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.save_contextual_representations("unused", occurrence_ids=["a"], hidden=hidden,
                                                       provenance=bad)
        for bad_matrix in (np.array([[1, 2]], dtype=np.int64), np.array([[True, False]]),
                           np.array([[1 + 0j, 2 + 0j]]), np.array([[object(), object()]])):
            with self.assertRaisesRegex(ValueError, "floating-point"):
                module.validate_matrix(["a"], bad_matrix)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "run"
            module.save_contextual_representations(target, occurrence_ids=["a"], hidden=hidden,
                                                   provenance=provenance)
            manifest_path = target / "manifest.json"
            original = module.json.loads(manifest_path.read_text())
            for mutate in (lambda m: m.update(artifacts={}), lambda m: m.update(dimension=3),
                           lambda m: m.update(dtype="float64"),
                           lambda m: m.update(evaluation_mode="eval")):
                bad = dict(original); mutate(bad)
                manifest_path.write_text(module.json.dumps(bad))
                with self.assertRaises(ValueError):
                    module.load_contextual_representations(target)
            manifest_path.write_text(module.json.dumps(original))
            original["provenance"]["seed"] = True
            manifest_path.write_text(module.json.dumps(original))
            with self.assertRaises(ValueError):
                module.load_contextual_representations(target)


if __name__ == "__main__":
    unittest.main()
