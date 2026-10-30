"""Synthetic integration tests for the pure two-report coordinator."""
from __future__ import annotations

import copy
import importlib
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "phonetics"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(ROOT.parent / "meta" / ".venv" / "Lib" / "site-packages"))
phonetics = types.ModuleType("phonetics"); phonetics.__path__ = [str(SOURCE)]
evaluation = types.ModuleType("phonetics.evaluation"); evaluation.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics); sys.modules.setdefault("phonetics.evaluation", evaluation)

batch_tests = importlib.import_module("test_anchor_q_batch")
q_batcher = importlib.import_module("phonetics.evaluation.anchor_q_batch")
d_batcher = importlib.import_module("phonetics.evaluation.dimsim_reporting")
contract = importlib.import_module("phonetics.evaluation.report_contract")
RUN_IDS = {"contextual-anchor-q": "q-review-01", "contextual-dimsim": "dimsim-review-01"}


class EvaluationReportContractTests(unittest.TestCase):
    def setUp(self):
        self.helper = batch_tests.AnchorQBatchTests()
        self.helper.setUp()
        self.q = q_batcher.score_anchor_q_batch(self.helper.bundle_path, self.helper.runs, self.helper.paths)
        self.d = d_batcher.score_contextual_dimsim_batch(self.helper.bundle_path, self.helper.runs, self.helper.paths,
            {"boundary_min_side_count": 1, "min_unique_pairs_per_bin": 1})

    def tearDown(self):
        self.helper.tearDown()

    def test_same_fixture_builds_two_independent_reports_and_five_figures(self):
        before_q, before_d = copy.deepcopy(self.q), copy.deepcopy(self.d)
        result = contract.build_evaluation_report(self.q, self.d, reference_model="components", run_ids=RUN_IDS)
        self.assertEqual(set(result["reports"]), {"contextual-anchor-q", "contextual-dimsim"})
        self.assertEqual([x["name"] for x in result["figures"]], [
            "Anchor-Q transitions", "Transition shares", "Component shares", "DIMSIM geometry", "DIMSIM correlation"])
        self.assertTrue(all(item["data"] is not None for item in result["figures"]))
        self.assertEqual(result["figures"][0]["axis_right_counts"], False)
        geometry, correlation = result["figures"][3:]
        self.assertEqual(set(geometry["data"]), {"curves", "boundaries", "counts"})
        self.assertTrue(geometry["axis_right_counts"] and geometry["ticks_labels_every_panel_both_columns"])
        self.assertEqual(set(correlation["data"]), {"model_rho", "seed_rho"})
        for figure in result["figures"]:
            for key, path in figure["data_keys"].items():
                root = result["reports"][figure["analysis"]]
                value = root
                for part in path.split("."): value = value[part]
                self.assertEqual(figure["data"][key], value)
            self.assertEqual(figure["export_format"], "unresolved")
        self.assertTrue(result["readiness"]["numeric_contract_valid"])
        self.assertFalse(result["readiness"]["end_to_end_ready"])
        self.assertEqual(json.dumps(self.q, sort_keys=True), json.dumps(before_q, sort_keys=True))
        self.assertEqual(json.dumps(self.d, sort_keys=True), json.dumps(before_d, sort_keys=True))
        self.assertEqual(json.dumps(result, allow_nan=False, sort_keys=True), json.dumps(
            contract.build_evaluation_report(self.q, self.d, reference_model="components", run_ids=RUN_IDS), allow_nan=False, sort_keys=True))
        self.assertEqual(json.loads(json.dumps(result["raw_batches"]))["contextual-anchor-q"], self.q)
        self.assertEqual(json.loads(json.dumps(result["raw_batches"]))["contextual-dimsim"], self.d)
        sorted_q = json.loads(json.dumps(self.q, sort_keys=True))
        sorted_d = json.loads(json.dumps(self.d, sort_keys=True))
        sorted_result = contract.build_evaluation_report(sorted_q, sorted_d, "components", run_ids=RUN_IDS)
        self.assertEqual(json.dumps(result, sort_keys=True), json.dumps(sorted_result, sort_keys=True))

    def test_bundle_inventory_reference_and_seed_mismatches_fail(self):
        bad = copy.deepcopy(self.d); bad["bundle_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bundle hashes"):
            contract.build_evaluation_report(self.q, bad, "components")
        bad = copy.deepcopy(self.d); bad["results"] = [r for r in bad["results"] if not (r["model_id"] == "acoustic" and r["seed"] == 4)]
        with self.assertRaisesRegex(ValueError, "all five seeds"):
            contract.build_evaluation_report(self.q, bad, "components")
        with self.assertRaises(ValueError):
            contract.build_evaluation_report(self.q, self.d, "unknown")

    def test_run_checkpoint_and_experiment_identity_mismatches_fail(self):
        for field, value in (("run_id", "other-run"), ("run_manifest_sha256", "0" * 64), ("checkpoint_sha256", "1" * 64)):
            bad = copy.deepcopy(self.d)
            next(row for row in bad["results"] if row["model_id"] == "acoustic" and row["seed"] == 0)[field] = value
            with self.assertRaisesRegex(ValueError, "mismatch|differs"):
                contract.build_evaluation_report(self.q, bad, "components")

    def test_missing_provenance_and_unsafe_non_json_payload_fail(self):
        bad = copy.deepcopy(self.q); del bad["results"][0]["checkpoint_sha256"]
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
            contract.build_evaluation_report(bad, self.d, "components")
        bad = copy.deepcopy(self.d); bad["results"][0]["score"]["modes"]["dimsim-natural"]["summary"]["rho"]["rho"] = float("nan")
        with self.assertRaises(ValueError):
            contract.build_evaluation_report(self.q, bad, "components")

    def test_matching_wrong_outer_checkpoint_is_rejected_against_q_score_provenance(self):
        q, d = copy.deepcopy(self.q), copy.deepcopy(self.d)
        for batch in (q, d):
            row = next(r for r in batch["results"] if r["model_id"] == "acoustic" and r["seed"] == 0)
            row["checkpoint_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "outer/nested checkpoint_sha256"):
            contract.build_evaluation_report(q, d, "components")
        q = copy.deepcopy(self.q)
        qrow = next(r for r in q["results"] if r["model_id"] == "acoustic" and r["seed"] == 0)
        qrow["experiment_hashes"]["config.json"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "outer/nested experiment_hashes"):
            contract.build_evaluation_report(q, self.d, "components")

    def test_nested_experiment_hash_conflict_rejected_when_dimsim_outer_has_effective_pins(self):
        q, d = copy.deepcopy(self.q), copy.deepcopy(self.d)
        q_by_id = {(row["model_id"], row["seed"]): row for row in q["results"]}
        for row in q["results"]:
            row.pop("experiment_hashes", None)
        for row in d["results"]:
            row["experiment_hashes"] = copy.deepcopy(q_by_id[(row["model_id"], row["seed"])]
                ["score"]["provenance"]["experiment_hashes"])
        q["results"][0]["score"]["provenance"]["experiment_hashes"]["config.json"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "outer/nested experiment_hashes"):
            contract.build_evaluation_report(q, d, "components")

    def test_resolved_nested_fallback_hashes_must_agree_across_seeds(self):
        q, d = copy.deepcopy(self.q), copy.deepcopy(self.d)
        for batch in (q, d):
            for row in batch["results"]:
                row.pop("experiment_hashes", None)
        bad = copy.deepcopy(q)
        bad["results"][0]["score"]["provenance"]["experiment_hashes"]["config.json"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "resolved experiment_hashes differ across seeds"):
            contract.build_evaluation_report(bad, d, "components")

        mixed_bad_q, mixed_bad_d = copy.deepcopy(q), copy.deepcopy(d)
        for row in mixed_bad_d["results"]:
            if not (row["model_id"] == "acoustic" and row["seed"] == 0):
                source = next(r for r in self.q["results"] if (r["model_id"], r["seed"]) == (row["model_id"], row["seed"]))
                row["experiment_hashes"] = copy.deepcopy(source["experiment_hashes"])
        mixed_bad_q["results"][0]["score"]["provenance"]["experiment_hashes"]["config.json"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "resolved experiment_hashes differ across seeds"):
            contract.build_evaluation_report(mixed_bad_q, mixed_bad_d, "components")

        mixed_q, mixed_d = copy.deepcopy(q), copy.deepcopy(d)
        fallback_row = next(r for r in mixed_d["results"] if r["model_id"] == "acoustic" and r["seed"] == 0)
        fallback_source = next(r for r in self.q["results"] if r["model_id"] == "acoustic" and r["seed"] == 0)
        fallback_row["experiment_hashes"] = copy.deepcopy(fallback_source["experiment_hashes"])
        valid = contract.build_evaluation_report(mixed_q, mixed_d, "components")
        acoustic_pins = [p for p in valid["scientific_input_pins"]["model_seed_provenance"] if p["model_id"] == "acoustic"]
        self.assertEqual(len(acoustic_pins), 5)
        self.assertEqual(len({json.dumps(p["experiment_hashes"], sort_keys=True) for p in acoustic_pins}), 1)

    def test_artifact_plan_requires_explicit_safe_ids(self):
        unresolved = contract.build_evaluation_report(self.q, self.d, "components")
        self.assertTrue(all(item["run_id"] is None and item["filenames"] is None for item in unresolved["artifact_plan"]))
        for bad in ({"contextual-anchor-q": "../escape", "contextual-dimsim": "valid"},
                    {"contextual-anchor-q": "one"}):
            with self.assertRaises(ValueError):
                contract.build_evaluation_report(self.q, self.d, "components", run_ids=bad)

    def test_plan_is_relative_safe_and_does_not_claim_exports(self):
        result = contract.build_evaluation_report(self.q, self.d, "components", run_ids=RUN_IDS)
        self.assertEqual(len(result["artifact_plan"]), 2)
        for item in result["artifact_plan"]:
            for path in [*item["directories"], *item["filenames"].values()]:
                self.assertFalse(path.startswith("/")); self.assertNotIn("..", Path(path).parts)
            self.assertEqual(item["run_id"], RUN_IDS[item["analysis"]])
        self.assertEqual(result["readiness"]["figure_exports"], "pending")
        self.assertNotIn("model_state", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
