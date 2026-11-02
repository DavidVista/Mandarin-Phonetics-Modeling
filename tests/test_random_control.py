import hashlib
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent / "meta" / ".venv" / "Lib" / "site-packages"))
SOURCE = ROOT / "src" / "phonetics"
phonetics = types.ModuleType("phonetics"); phonetics.__path__ = [str(SOURCE)]
evaluation = types.ModuleType("phonetics.evaluation"); evaluation.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics); sys.modules.setdefault("phonetics.evaluation", evaluation)
fixture = importlib.import_module("test_contextual_bundle")
bundle_mod = importlib.import_module("phonetics.evaluation.bundle")
reps = importlib.import_module("phonetics.evaluation.representations")
control = importlib.import_module("phonetics.evaluation.random_control")


class RandomControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        helper = fixture.ContextualBundleTests()
        self.path = helper.prepare(self.root, "random-control", q_draws=2)
        self.bundle = bundle_mod.load_evaluation_bundle(self.path)
        self.bundle_hash = hashlib.sha256((self.path / "manifest.json").read_bytes()).hexdigest()
        self.ids = sorted(x["occurrence_id"] for x in self.bundle["occurrences"])

    def tearDown(self): self.tmp.cleanup()

    def test_generation_is_reproducible_sorted_and_unaggregated(self):
        before = list(self.ids)
        global_state = np.random.get_state()
        a = control.generate_random_control(self.ids, dimension=4)
        b = control.generate_random_control(self.ids, dimension=4)
        after_state = np.random.get_state()
        self.assertEqual(a["occurrence_ids"], before)
        self.assertEqual(a["matrices"].keys(), b["matrices"].keys())
        for seed in range(5):
            self.assertEqual(a["matrices"][seed].shape, (len(self.ids), 4))
            self.assertEqual(a["matrices"][seed].dtype, np.float32)
            self.assertTrue(np.array_equal(a["matrices"][seed], b["matrices"][seed]))
            self.assertGreater(np.unique(a["matrices"][seed], axis=0).shape[0], 1)
        self.assertFalse(np.array_equal(a["matrices"][0], a["matrices"][1]))
        self.assertEqual(global_state[0], after_state[0])
        self.assertTrue(np.array_equal(global_state[1], after_state[1]))
        self.assertEqual(global_state[2:], after_state[2:])
        self.assertEqual(self.ids, before)

    def test_params_strict_and_random_seed_sequence_fixed(self):
        for bad in (True, 0, 1.5, "4"):
            with self.subTest(dimension=bad), self.assertRaises(ValueError):
                control.generate_random_control(self.ids, dimension=bad)
        for seeds in ([0, 1, 2, 3, True], [0, 1, 2, 3], [1, 2, 3, 4, 5]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                control.generate_random_control(self.ids, dimension=4, random_seeds=seeds)

    def test_scoring_reuses_draws_for_both_core_scorers_and_is_json_finite(self):
        generated = control.generate_random_control(self.ids, dimension=3)
        original = {s: x.copy() for s, x in generated["matrices"].items()}
        resolution = self._resolution(self.ids, 3)
        source_hash = "b" * 64
        result = control.evaluate_random_control(self.bundle, generated, dimension_resolution=resolution,
            source_sha256=source_hash, bundle_manifest_sha256=self.bundle_hash,
            analysis_config={"boundary_min_side_count": 1})
        report = result["report"]
        self.assertEqual(report["model_id"], "random-control")
        self.assertEqual(report["seed_kind"], "random")
        self.assertEqual(report["random_seeds"], [0, 1, 2, 3, 4])
        for seed in range(5):
            rec = report["per_seed"][seed]
            self.assertEqual(rec["anchor_q"]["provenance"]["seed_kind"], "random")
            self.assertEqual(rec["provenance"]["adapter"], "iid_standard_normal")
            self.assertEqual(rec["provenance"]["source_sha256"], source_hash)
            self.assertEqual(len(rec["anchor_q"]["draw_scores"]), 24)
            for mode in ("dimsim-natural", "dimsim-balanced"):
                self.assertEqual(len(rec["dimsim"]["modes"][mode]["rows"]), len(self.bundle["pairs"][mode]))
            self.assertTrue(np.array_equal(result["matrices"][seed], original[seed]))
        overall = [r["anchor_q"]["overall"]["q"] for r in report["per_seed"]]
        self.assertAlmostEqual(report["anchor_q"]["overall"]["mean"], float(np.mean(overall)))
        self.assertAlmostEqual(report["anchor_q"]["overall"]["sample_sd"], float(np.std(overall, ddof=1)))
        edge = next(iter(report["anchor_q"]["edges"]))
        edge_rows = [[x for x in r["anchor_q"]["draw_scores"] if x["edge_id"] == edge] for r in report["per_seed"]]
        averaged_draws = np.mean([[x["score"] for x in rows] for rows in edge_rows], axis=0)
        expected_mcse = float(np.std(averaged_draws, ddof=1) / np.sqrt(len(averaged_draws)))
        self.assertAlmostEqual(report["anchor_q"]["edges"][edge]["mc_se_seed_mean"], expected_mcse)
        json.dumps(report, allow_nan=False)
        roundtrip = json.loads(json.dumps(report, allow_nan=False))
        self.assertEqual(roundtrip["random_seeds"], [0, 1, 2, 3, 4])
        sorted_records = json.loads(json.dumps(report["per_seed"], sort_keys=True, allow_nan=False))
        rebuilt = control.summarize_random_control(sorted_records, self.bundle_hash)
        self.assertEqual(json.dumps(rebuilt, sort_keys=True, allow_nan=False),
                         json.dumps(report, sort_keys=True, allow_nan=False))
        self.assertIsNone(control._mcse([0.5])["mc_se_seed_mean"])
        self.assertEqual(control._mcse([0.5])["mc_se_status"], "not_available_single_draw")
        self.assertEqual(control._mcse([0.5, 0.5])["mc_se_seed_mean"], 0.0)

    def test_dimension_resolver_requires_complete_matching_learned_runs(self):
        exp_hashes = dict(self.bundle["manifest"]["source_experiment"]["hashes"])
        exp_hashes["acoustic_targets.npy"] = "a" * 64
        runs = {}
        for model, dimension in (("syllable", 3), ("components", 3)):
            runs[model] = {}
            for seed in range(5):
                path = self.root / "reps" / model / str(seed)
                provenance = {"model_id": model, "seed": seed, "checkpoint_sha256": hashlib.sha256(f"{model}{seed}".encode()).hexdigest(),
                    "bundle_manifest_sha256": self.bundle_hash, "config_sha256": exp_hashes["config.json"],
                    "vocab_sha256": exp_hashes["vocab.json"], "source_sha256": exp_hashes["manifest.json"],
                    "experiment_hashes": exp_hashes,
                    "environment": {k: "fixture" for k in ("python", "numpy", "torch", "platform", "device")},
                    "evaluation_mode": True}
                reps.save_contextual_representations(path, occurrence_ids=self.ids,
                    hidden=np.arange(1, len(self.ids)*dimension+1, dtype=np.float32).reshape(-1, dimension), provenance=provenance)
                runs[model][seed] = path
        resolved = control.resolve_contextual_dimension(runs, bundle_manifest_sha256=self.bundle_hash, bundle=self.bundle)
        self.assertEqual(resolved["dimension"], 3)
        self.assertEqual(set(resolved["models"]), set(runs))
        resolved = json.loads(json.dumps(resolved))
        generated = control.generate_random_control(self.ids, dimension=3)
        evaluated = control.evaluate_random_control(self.bundle, generated, bundle_manifest_sha256=self.bundle_hash,
            dimension_resolution=resolved, source_sha256="c" * 64)
        self.assertEqual(evaluated["report"]["random_seeds"], [0, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "exactly"):
            control.resolve_contextual_dimension({"syllable": {0: runs["syllable"][0]}}, bundle_manifest_sha256=self.bundle_hash, bundle=self.bundle)
        bad = {**runs, "components": dict(runs["components"])}
        # A verified but different-dimension run remains a real run, and must fail resolution.
        path = self.root / "reps" / "components" / "different"
        p = dict(reps.load_contextual_representations(runs["components"][0])["manifest"]["provenance"])
        reps.save_contextual_representations(path, occurrence_ids=self.ids,
            hidden=np.arange(1, len(self.ids)*4+1, dtype=np.float32).reshape(-1, 4), provenance=p)
        bad["components"][0] = path
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            control.resolve_contextual_dimension(bad, bundle_manifest_sha256=self.bundle_hash, bundle=self.bundle)

    def _resolution(self, ids, dimension):
        return {"schema_version": 1, "dimension": dimension, "bundle_manifest_sha256": self.bundle_hash,
                "occurrence_ids": list(ids), "models": {"syllable": {str(seed): {
                    "manifest_sha256": hashlib.sha256(f"m{seed}".encode()).hexdigest(),
                    "checkpoint_sha256": hashlib.sha256(f"c{seed}".encode()).hexdigest(),
                    "dimension": dimension, "occurrence_ids": list(ids)} for seed in range(5)}}}

    def test_high_level_rejects_unmatched_or_tampered_input_before_scoring(self):
        generated = control.generate_random_control(self.ids, dimension=3)
        resolution = self._resolution(self.ids, 3)
        for bad in ({**resolution, "dimension": 999}, {**resolution, "bundle_manifest_sha256": "0" * 64},
                    {**resolution, "occurrence_ids": self.ids[:-1]}):
            with self.subTest(resolution=bad.get("dimension")), self.assertRaises(ValueError):
                control.evaluate_random_control(self.bundle, generated, bundle_manifest_sha256=self.bundle_hash,
                    dimension_resolution=bad, source_sha256="b" * 64)
        altered = {"occurrence_ids": self.ids, "matrices": dict(generated["matrices"])}
        altered["matrices"][0] = np.zeros((len(self.ids), 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "matrix 0"):
            control.evaluate_random_control(self.bundle, altered, bundle_manifest_sha256=self.bundle_hash,
                dimension_resolution=resolution, source_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "source SHA"):
            control.evaluate_random_control(self.bundle, generated, bundle_manifest_sha256=self.bundle_hash,
                dimension_resolution=resolution, source_sha256="invalid")

    def test_summary_requires_all_five_defined_rhos_and_strict_identities(self):
        generated = control.generate_random_control(self.ids, dimension=3)
        result = control.evaluate_random_control(self.bundle, generated, bundle_manifest_sha256=self.bundle_hash,
            dimension_resolution=self._resolution(self.ids, 3), source_sha256="b" * 64,
            analysis_config={"boundary_min_side_count": 1})
        records = result["report"]["per_seed"]
        mode = "dimsim-natural"
        records[2]["dimsim"]["modes"][mode]["summary"]["rho"] = {"rho": None, "undefined_reason": "constant_distance"}
        summary = control.summarize_random_control(records, self.bundle_hash)
        self.assertIsNone(summary["dimsim"][mode]["rho_mean"])
        self.assertIsNone(summary["dimsim"][mode]["rho_sample_sd"])
        self.assertEqual(summary["dimsim"][mode]["status"], "requires_all_five_defined")
        self.assertEqual(summary["dimsim"][mode]["per_seed_rho"][2]["undefined_reason"], "constant_distance")
        json.dumps(summary, allow_nan=False)
        bad = list(records); bad[0] = {**bad[0], "random_seed": False}
        with self.assertRaisesRegex(ValueError, "ordered integer"):
            control.summarize_random_control(bad, self.bundle_hash)
        bad = list(records); bad[1] = {**bad[1], "anchor_q": dict(bad[1]["anchor_q"])}
        bad[1]["anchor_q"]["draw_scores"] = list(bad[1]["anchor_q"]["draw_scores"])
        bad[1]["anchor_q"]["draw_scores"][0] = {**bad[1]["anchor_q"]["draw_scores"][0], "triplet_id": "wrong"}
        with self.assertRaisesRegex(ValueError, "Q triplet identity"):
            control.summarize_random_control(bad, self.bundle_hash)

    def test_matched_q_mcse_uses_correlated_seed_average_and_recomputes_q(self):
        generated = control.generate_random_control(self.ids, dimension=3)
        result = control.evaluate_random_control(self.bundle, generated, bundle_manifest_sha256=self.bundle_hash,
            dimension_resolution=self._resolution(self.ids, 3), source_sha256="b" * 64)
        records = result["report"]["per_seed"]
        edge = next(iter(records[0]["anchor_q"]["edges"]))
        patterns = ((0, 0), (0, 0), (0, 1), (1, 1), (1, 1))
        for record, pattern in zip(records, patterns):
            rows = [row for row in record["anchor_q"]["draw_scores"] if row["edge_id"] == edge]
            self.assertEqual(len(rows), 2)
            for row, score in zip(rows, pattern):
                row["score"] = score
            record["anchor_q"]["edges"][edge]["q"] = 999.0
            record["anchor_q"]["overall"]["q"] = 999.0
        summary = control.summarize_random_control(records, self.bundle_hash)
        self.assertAlmostEqual(summary["anchor_q"]["edges"][edge]["mc_se_seed_mean"], .1)
        self.assertAlmostEqual(summary["anchor_q"]["edges"][edge]["mean"], .5)
        component = edge.split(":", 1)[0]
        component_errors = [summary["anchor_q"]["edges"][key]["mc_se_seed_mean"]
                            for key in summary["anchor_q"]["edges"] if key.startswith(component + ":")]
        overall_errors = [value["mc_se_seed_mean"] for value in summary["anchor_q"]["edges"].values()]
        self.assertAlmostEqual(summary["anchor_q"]["components"][component]["mc_se_seed_mean"],
                               np.sqrt(sum(x*x for x in component_errors)) / len(component_errors))
        self.assertAlmostEqual(summary["anchor_q"]["overall"]["mc_se_seed_mean"],
                               np.sqrt(sum(x*x for x in overall_errors)) / len(overall_errors))
        self.assertNotEqual(summary["anchor_q"]["overall"]["mean"], 999.0)
        bad = json.loads(json.dumps(records))
        target = next(row for row in bad[0]["anchor_q"]["draw_scores"] if row["edge_id"] == edge)
        target["score"] = .4
        with self.assertRaisesRegex(ValueError, "Q draw values"):
            control.summarize_random_control(bad, self.bundle_hash)

    def test_single_q_draw_mcse_statuses_are_explicit_and_json_safe(self):
        helper = fixture.ContextualBundleTests()
        path = helper.prepare(self.root, "random-control-single-q", q_draws=1)
        bundle = bundle_mod.load_evaluation_bundle(path)
        bundle_hash = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
        ids = sorted(row["occurrence_id"] for row in bundle["occurrences"])
        generated = control.generate_random_control(ids, dimension=3)
        resolution = {**self._resolution(ids, 3), "bundle_manifest_sha256": bundle_hash}
        result = control.evaluate_random_control(bundle, generated, bundle_manifest_sha256=bundle_hash,
            dimension_resolution=resolution, source_sha256="d" * 64)
        q = result["report"]["anchor_q"]
        for edge in q["edges"].values():
            self.assertIsNone(edge["mc_se_seed_mean"])
            self.assertEqual(edge["mc_se_status"], "not_available_single_draw")
        for value in (*q["components"].values(), q["overall"]):
            self.assertIsNone(value["mc_se_seed_mean"])
            self.assertEqual(value["mc_se_status"], "not_available_single_draw")
        json.dumps(result["report"], allow_nan=False)

    def test_summary_rejects_cross_seed_dimsim_metric_changes(self):
        generated = control.generate_random_control(self.ids, dimension=3)
        result = control.evaluate_random_control(self.bundle, generated, bundle_manifest_sha256=self.bundle_hash,
            dimension_resolution=self._resolution(self.ids, 3), source_sha256="b" * 64)
        records = json.loads(json.dumps(result["report"]["per_seed"]))
        row = records[1]["dimsim"]["modes"]["dimsim-natural"]["rows"][0]
        row["overlap_mask"] = "bad"
        with self.assertRaisesRegex(ValueError, "DIMSIM pair identity"):
            control.summarize_random_control(records, self.bundle_hash)

    def test_summary_rejects_reversed_actual_q_draw_order(self):
        generated = control.generate_random_control(self.ids, dimension=3)
        result = control.evaluate_random_control(self.bundle, generated, bundle_manifest_sha256=self.bundle_hash,
            dimension_resolution=self._resolution(self.ids, 3), source_sha256="b" * 64)
        records = json.loads(json.dumps(result["report"]["per_seed"], sort_keys=True))
        records[0]["anchor_q"]["draw_scores"].reverse()
        with self.assertRaisesRegex(ValueError, "canonical edge order"):
            control.summarize_random_control(records, self.bundle_hash)


if __name__ == "__main__": unittest.main()
