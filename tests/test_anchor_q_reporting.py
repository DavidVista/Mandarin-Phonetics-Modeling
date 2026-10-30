"""Unittest coverage for anchor-Q report statistics and batch interoperability."""
from __future__ import annotations

import copy
import importlib
import json
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "phonetics"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(ROOT.parent / "meta" / ".venv" / "Lib" / "site-packages"))
phonetics = types.ModuleType("phonetics")
phonetics.__path__ = [str(SOURCE)]
evaluation = types.ModuleType("phonetics.evaluation")
evaluation.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics)
sys.modules.setdefault("phonetics.evaluation", evaluation)

sampling = importlib.import_module("phonetics.evaluation.anchor_sampling")
reporting = importlib.import_module("phonetics.evaluation.anchor_q_reporting")


def make_batch(seed_profiles=None, models=("syllable", "components"), draws=4):
    """Create a canonical batch with fixed draw identities and configurable values."""
    edge_ids = [f"{component}:{low}->{high}" for component, low, high in sampling.EDGES]
    records = []
    for model in models:
        for seed in range(5):
            values = (seed_profiles or {}).get((model, seed), [0.0, 0.5, 1.0, 0.5][:draws])
            rows = []
            for edge in edge_ids:
                for index in range(draws):
                    rows.append({"draw_id": f"anchor-q:v1:{edge}:{index:06d}", "edge_id": edge,
                                 "triplet_id": f"triplet-{index}", "score": values[index]})
            edge_summaries = {}
            coverage_edges = {}
            for edge in edge_ids:
                edge_values = [row["score"] for row in rows if row["edge_id"] == edge]
                edge_summaries[edge] = {"draw_count": draws, "q": float(np.mean(edge_values))}
                coverage_edges[edge] = {"draw_count": draws, "unique_triplet_count": draws}
            score = {"model_id": model, "seed": seed, "bundle_manifest_sha256": "a" * 64,
                     "tie_tolerance": 1e-12, "draw_scores": rows, "edges": edge_summaries,
                     "provenance": {"model_id": model, "seed": seed, "bundle_manifest_sha256": "a" * 64},
                     "coverage": {"edges": coverage_edges}}
            records.append({"model_id": model, "seed": seed, "score": score})
    return {"schema_version": 1, "bundle_manifest_sha256": "a" * 64, "results": records}


def rows_for(result, table, **keys):
    return next(row for row in result["tables"][table] if all(row[key] == value for key, value in keys.items()))


class AnchorQReportingTests(unittest.TestCase):
    def test_heterogeneous_seed_mean_sd_mcse_and_seed_labels(self):
        profiles = {}
        for seed, vals in enumerate(([0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0], [.5, .5, .5, .5])):
            profiles[("syllable", seed)] = vals
            profiles[("components", seed)] = vals
        result = reporting.build_anchor_q_report(make_batch(profiles, models=("syllable",), draws=4))
        row = rows_for(result, "model_edge_seed_summary", model_id="syllable", edge_id="initial:000->100")
        values = [0, 1, .5, .5, .5]
        self.assertAlmostEqual(row["mean"], .5)
        self.assertAlmostEqual(row["sample_sd"], np.std(values, ddof=1))
        self.assertAlmostEqual(row["mc_se"], np.std([.5, .5, .5, .5], ddof=1) / 2)
        self.assertEqual([item["seed"] for item in row["seed_values"]], [0, 1, 2, 3, 4])

    def test_correlated_seeds_do_not_shrink_mcse_by_sqrt_five(self):
        profiles = {(model, seed): [0, 1, 0, 1] for model in ("syllable",) for seed in range(5)}
        result = reporting.build_anchor_q_report(make_batch(profiles, models=("syllable",), draws=4))
        row = rows_for(result, "model_edge_seed_summary", model_id="syllable", edge_id="initial:000->100")
        self.assertAlmostEqual(row["mc_se"], np.std([0, 1, 0, 1], ddof=1) / 2)

    def test_single_draw_has_explicit_unavailable_mcse(self):
        result = reporting.build_anchor_q_report(make_batch(draws=1))
        edge = rows_for(result, "model_edge_seed_summary", model_id="syllable", edge_id="initial:000->100")
        paired = rows_for(result, "paired_deltas", model_id="components", metric="overall")
        self.assertIsNone(edge["mc_se"])
        self.assertEqual(edge["mc_se_status"], "not_available_single_draw")
        self.assertIsNone(paired["mc_se"])

    def test_paired_mcse_sign_identical_models_and_seed_permutation(self):
        result = reporting.build_anchor_q_report(make_batch(draws=4))
        paired = rows_for(result, "paired_deltas", model_id="components", metric="overall")
        self.assertAlmostEqual(paired["mean_delta_q"], 0)
        self.assertAlmostEqual(paired["mc_se"], 0)
        profiles = {}
        for seed in range(5):
            profiles[("syllable", seed)] = [0, 0, 0, 0]
            profiles[("components", seed)] = [1, 1, 1, 1]
        value = make_batch(profiles, draws=4)
        forward = reporting.build_anchor_q_report(value)
        value["results"].reverse()
        reverse = reporting.build_anchor_q_report(value)
        delta = rows_for(forward, "paired_deltas", model_id="components", metric="overall")
        self.assertEqual(delta["mean_delta_q"], 1)
        self.assertEqual(delta["seed_deltas"], rows_for(reverse, "paired_deltas", model_id="components", metric="overall")["seed_deltas"])

    def test_shares_normalize_after_meaning_seed_edge_values_and_all_zero(self):
        profiles = {("syllable", seed): ([0, 0, 0, 0] if seed < 4 else [1, 1, 1, 1]) for seed in range(5)}
        result = reporting.build_anchor_q_report(make_batch(profiles, models=("syllable",), draws=4))
        edge_means = [row["mean"] for row in result["tables"]["model_edge_seed_summary"]]
        self.assertAlmostEqual(sum(row["share"] for row in result["tables"]["transition_shares"]), 1)
        self.assertAlmostEqual(result["tables"]["transition_shares"][0]["share"], edge_means[0] / sum(edge_means))
        zeros = {("syllable", seed): [0, 0, 0, 0] for seed in range(5)}
        zero_result = reporting.build_anchor_q_report(make_batch(zeros, models=("syllable",), draws=4))
        self.assertTrue(all(row["share"] is None and row["status"] == "undefined_zero_total" for row in zero_result["tables"]["transition_shares"]))

    def test_report_serializes_without_nan_and_preserves_input(self):
        batch = make_batch(draws=4)
        before = copy.deepcopy(batch)
        result = reporting.build_anchor_q_report(batch)
        self.assertEqual(batch, before)
        json.dumps(result, allow_nan=False)

    def test_sorted_json_object_keys_preserve_canonical_edge_rows(self):
        batch = make_batch(draws=4)
        restored = json.loads(json.dumps(batch, sort_keys=True))
        self.assertEqual(reporting.build_anchor_q_report(batch), reporting.build_anchor_q_report(restored))

    def test_actual_batch_result_integration_with_small_draw_count(self):
        from test_anchor_q_batch import AnchorQBatchTests
        helper = AnchorQBatchTests()
        helper.setUp()
        try:
            batch_module = importlib.import_module("phonetics.evaluation.anchor_q_batch")
            result = batch_module.score_anchor_q_batch(helper.bundle_path, helper.runs, helper.paths)
            report = reporting.build_anchor_q_report(result, reference_model="components")
            self.assertEqual(len(report["tables"]["model_seed_edges"]), 120)
            edge = report["tables"]["model_seed_edges"][0]
            self.assertEqual(edge["draw_count"], 2)
            self.assertIsNotNone(edge["mc_se"])
        finally:
            helper.tearDown()

    def test_rejects_bad_identity_schema_counts_and_boolean_scores(self):
        mutations = []
        value = make_batch(draws=2); value["results"][0]["score"]["draw_scores"][0]["score"] = True; mutations.append(value)
        value = make_batch(draws=2); value["results"][0]["score"]["edges"].pop(next(iter(value["results"][0]["score"]["edges"]))); mutations.append(value)
        value = make_batch(draws=2); value["results"][0]["score"]["draw_scores"][0]["triplet_id"] = "wrong"; mutations.append(value)
        value = make_batch(draws=2); value["results"][0]["score"]["edges"][next(iter(value["results"][0]["score"]["edges"]))]["draw_count"] = 9; mutations.append(value)
        for invalid in mutations:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    reporting.build_anchor_q_report(invalid)


if __name__ == "__main__":
    unittest.main()
