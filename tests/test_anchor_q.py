import hashlib
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "phonetics"
phonetics_package = types.ModuleType("phonetics"); phonetics_package.__path__ = [str(SOURCE)]
evaluation_package = types.ModuleType("phonetics.evaluation"); evaluation_package.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics_package); sys.modules.setdefault("phonetics.evaluation", evaluation_package)
q = importlib.import_module("phonetics.evaluation.anchor_q")
fixture = importlib.import_module("test_contextual_bundle")


class AnchorQTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        helper = fixture.ContextualBundleTests()
        bundle_path = helper.prepare(self.root, "score", q_draws=2)
        self.bundle = importlib.import_module("phonetics.evaluation.bundle").load_evaluation_bundle(bundle_path)
        self.bundle_path = bundle_path
        self.bundle_manifest_sha256 = hashlib.sha256((bundle_path / "manifest.json").read_bytes()).hexdigest()
        self.ids = [row["occurrence_id"] for row in self.bundle["occurrences"]]
        self.hidden = np.arange(1, len(self.ids) * 3 + 1, dtype=np.float32).reshape(len(self.ids), 3)
        self.prov = {"model_id": "fixture", "seed": 7,
                     "bundle_manifest_sha256": self.bundle_manifest_sha256}

    def tearDown(self):
        self.tmp.cleanup()

    def score(self, **kwargs):
        return q.score_anchor_q(self.bundle, self.ids, self.hidden, self.prov,
                                bundle_manifest_sha256=self.bundle_manifest_sha256, **kwargs)

    def test_full_draw_preservation_reordered_inventory_and_aggregation(self):
        # All sampled rows, including repeated triplet identities, remain separate draws.
        result = self.score()
        self.assertEqual(len(result["draw_scores"]), 24)
        original_row = self.bundle["triplets"][1]
        self.bundle["triplets"][1] = {**self.bundle["triplets"][0], "draw_index": 1,
                                       "draw_id": self.bundle["triplets"][0]["draw_id"].rsplit(":", 1)[0] + ":000001"}
        duplicated = self.score()
        self.assertEqual(len(duplicated["draw_scores"]), 24)
        self.assertEqual(duplicated["draw_scores"][0]["triplet_id"], duplicated["draw_scores"][1]["triplet_id"])
        self.bundle["triplets"][1] = original_row
        self.assertEqual(set(result["edges"]), set(self.bundle["config"]["sampling"]["anchor_q"]["edge_order"]))
        self.assertAlmostEqual(result["overall"]["q"], np.mean([x["q"] for x in result["edges"].values()]))
        order = np.arange(len(self.ids))[::-1]
        permuted = q.score_anchor_q(self.bundle, [self.ids[i] for i in order], self.hidden[order], self.prov,
                                    bundle_manifest_sha256=self.bundle_manifest_sha256)
        self.assertEqual(result["draw_scores"], permuted["draw_scores"])

    def test_tie_boundary_and_order_decisions(self):
        row = self.bundle["triplets"][0]
        vector = {x: np.array([1.0, 0.0]) for x in (row["anchor_id"], row["high_id"], row["low_id"])}
        matrix = np.vstack([vector.get(oid, np.array([1.0, 0.0])) for oid in self.ids])
        cosine_results = [1e-12, 0.0, 0.0, 1e-12, np.nextafter(1e-12, np.inf), 0.0,
                          0.0, np.nextafter(1e-12, np.inf)] + [0.5, 0.5] * 20
        with patch.object(q, "stable_cosine", side_effect=cosine_results):
            result = q.score_anchor_q(self.bundle, self.ids, matrix, self.prov, bundle_manifest_sha256=self.bundle_manifest_sha256)
        self.assertEqual(result["draw_scores"][0]["score"], 0.5)
        self.assertEqual(result["draw_scores"][1]["score"], 0.5)
        self.assertEqual(result["draw_scores"][2]["score"], 1.0)
        self.assertEqual(result["draw_scores"][3]["score"], 0.0)
        with patch.object(q, "stable_cosine", side_effect=[0.2, 0.8] + [0.5, 0.5] * 23):
            result = q.score_anchor_q(self.bundle, self.ids, matrix, self.prov, bundle_manifest_sha256=self.bundle_manifest_sha256)
        self.assertEqual(result["draw_scores"][0]["score"], 0.0)

    def test_hand_computed_edge_q_and_sample_sd_mcse(self):
        outcomes = [1.0, 0.0] * 12
        cosines = [value for outcome in outcomes for value in ((0.8, 0.2) if outcome else (0.2, 0.8))]
        before = self.hidden.copy()
        with patch.object(q, "stable_cosine", side_effect=cosines):
            result = self.score()
        np.testing.assert_array_equal(self.hidden, before)
        for edge in result["edges"].values():
            self.assertEqual(edge["q"], 0.5)
            self.assertEqual(edge["mc_se"], 0.5)
        self.assertEqual(result["overall"]["q"], 0.5)
        self.assertAlmostEqual(result["overall"]["mc_se"], 0.5 / (12 ** 0.5), places=15)
        json.dumps(result, allow_nan=False)

    def test_rejects_bad_inventory_hash_tolerance_and_matrix(self):
        with self.assertRaisesRegex(ValueError, "inventory"):
            q.score_anchor_q(self.bundle, self.ids[:-1], self.hidden[:-1], self.prov, bundle_manifest_sha256=self.bundle_manifest_sha256)
        bad = dict(self.prov, bundle_manifest_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            q.score_anchor_q(self.bundle, self.ids, self.hidden, bad, bundle_manifest_sha256=self.bundle_manifest_sha256)
        for value in (True, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): q._validate_tolerance(value)
        with self.assertRaisesRegex(ValueError, "floating-point"):
            q.score_anchor_q(self.bundle, self.ids, self.hidden.astype(int), self.prov, bundle_manifest_sha256=self.bundle_manifest_sha256)

    def test_duplicate_draw_id_and_incomplete_edge_rejected(self):
        original = self.bundle["triplets"]
        self.bundle["triplets"] = [dict(r) for r in original]
        self.bundle["triplets"][1]["draw_id"] = self.bundle["triplets"][0]["draw_id"]
        with self.assertRaisesRegex(ValueError, "duplicate anchor-Q draw ID"):
            self.score()
        self.bundle["triplets"] = original[:-1]
        with self.assertRaisesRegex(ValueError, "draw count"):
            self.score()
        self.bundle["triplets"] = original[::-1]
        with self.assertRaisesRegex(ValueError, "global edge block order"):
            self.score()
        self.bundle["triplets"] = [dict(r) for r in original]
        self.bundle["triplets"][0]["draw_index"] = 1
        with self.assertRaisesRegex(ValueError, "draw index/order"):
            self.score()

    def test_raw_vectors_zero_nonfinite_and_extreme_norms(self):
        ids = self.ids
        values = np.tile(np.array([1e-300, 0.0]), (len(ids), 1))
        result = q.score_anchor_q(self.bundle, ids, values, self.prov, bundle_manifest_sha256=self.bundle_manifest_sha256)
        self.assertEqual(result["overall"]["q"], 0.5)
        values[0] = 0
        with self.assertRaisesRegex(ValueError, ids[0]):
            q.score_anchor_q(self.bundle, ids, values, self.prov, bundle_manifest_sha256=self.bundle_manifest_sha256)
        values[0] = np.nan
        with self.assertRaisesRegex(ValueError, ids[0]):
            q.score_anchor_q(self.bundle, ids, values, self.prov, bundle_manifest_sha256=self.bundle_manifest_sha256)
        values[0] = np.inf
        with self.assertRaisesRegex(ValueError, ids[0]):
            q.score_anchor_q(self.bundle, ids, values, self.prov, bundle_manifest_sha256=self.bundle_manifest_sha256)

    def test_path_wrapper_roundtrip_with_writer_and_full_provenance(self):
        representations = importlib.import_module("phonetics.evaluation.representations")
        names = ("manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json", "acoustic_targets.npy")
        hashes = {name: f"{i + 1:x}" * 64 for i, name in enumerate(names)}
        provenance = {"model_id": "components", "seed": 17, "checkpoint_sha256": "a" * 64,
                      "bundle_manifest_sha256": self.bundle_manifest_sha256,
                      "experiment_hashes": hashes, "config_sha256": hashes["config.json"],
                      "vocab_sha256": hashes["vocab.json"], "source_sha256": "f" * 64,
                      "environment": {"python": "3.12", "numpy": "2.3.5", "torch": "2.6.0",
                                      "platform": "test-platform", "device": "cpu"},
                      "evaluation_mode": True}
        order = np.argsort(self.ids)
        ids = [self.ids[i] for i in order]
        hidden = self.hidden[order]
        rep_path = self.root / "contextual"
        representations.save_contextual_representations(rep_path, occurrence_ids=ids,
                                                        hidden=hidden, provenance=provenance)
        path_result = q.score_anchor_q_paths(self.bundle_path, rep_path)
        pure_result = q.score_anchor_q(self.bundle, ids, hidden, provenance,
                                       bundle_manifest_sha256=self.bundle_manifest_sha256)
        self.assertEqual(path_result, pure_result)

    def test_single_draw_precision_status(self):
        helper = fixture.ContextualBundleTests()
        one_path = helper.prepare(self.root, "single", q_draws=1)
        one = importlib.import_module("phonetics.evaluation.bundle").load_evaluation_bundle(one_path)
        digest = hashlib.sha256((one_path / "manifest.json").read_bytes()).hexdigest()
        provenance = dict(self.prov, bundle_manifest_sha256=digest)
        result = q.score_anchor_q(one, self.ids, self.hidden, provenance, bundle_manifest_sha256=digest)
        self.assertIsNone(result["overall"]["mc_se"])
        self.assertEqual(result["overall"]["mc_se_status"], "not_available_single_draw")
if __name__ == "__main__":
    unittest.main()
