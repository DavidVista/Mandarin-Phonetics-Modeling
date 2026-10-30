import hashlib
import importlib
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "phonetics"
phonetics = types.ModuleType("phonetics"); phonetics.__path__ = [str(SOURCE)]
evaluation = types.ModuleType("phonetics.evaluation"); evaluation.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics); sys.modules.setdefault("phonetics.evaluation", evaluation)
fixture = importlib.import_module("test_contextual_bundle")
bundle_mod = importlib.import_module("phonetics.evaluation.bundle")
reps = importlib.import_module("phonetics.evaluation.representations")
dim = importlib.import_module("phonetics.evaluation.contextual_dimsim")
sampling = importlib.import_module("phonetics.evaluation.dimsim_sampling")
report = importlib.import_module("phonetics.evaluation.dimsim_reporting")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class ContextualDimsimTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        helper = fixture.ContextualBundleTests()
        self.path = helper.prepare(self.root, "dimsim", q_draws=2)
        self.bundle = bundle_mod.load_evaluation_bundle(self.path)
        self.ids = sorted(r["occurrence_id"] for r in self.bundle["occurrences"])
        self.hidden = np.arange(1, len(self.ids) * 3 + 1, dtype=np.float32).reshape(-1, 3)
        self.prov = {"model_id": "syllable", "seed": 0, "bundle_manifest_sha256": sha(self.path / "manifest.json")}

    def tearDown(self): self.tmp.cleanup()

    def score(self, cfg=None):
        return dim.score_contextual_dimsim(self.bundle, self.ids, self.hidden, self.prov,
                                           self.prov["bundle_manifest_sha256"], cfg)

    def test_modes_preserve_draws_and_have_fixed_bins(self):
        snapshot = self.hidden.copy()
        result = self.score({"boundary_min_side_count": 1, "min_unique_pairs_per_bin": 1})
        self.assertEqual(list(result["modes"]), list(dim.MODES))
        for mode in dim.MODES:
            summary = result["modes"][mode]["summary"]
            self.assertEqual(len(summary["bins"]), 13)
            self.assertEqual(summary["draw_count"], len(self.bundle["pairs"][mode]))
            self.assertEqual(len(result["modes"][mode]["rows"]), summary["draw_count"])
        self.assertTrue(np.array_equal(snapshot, self.hidden))

    def test_invalid_distance_is_retained_but_filtered(self):
        pair = self.bundle["pairs"]["dimsim-natural"][0]
        pair["distance_status"], pair["dimsim_distance"] = "invalid:Test", None
        result = self.score()
        summary = result["modes"]["dimsim-natural"]["summary"]
        self.assertEqual(summary["invalid_metric_count"], 1)
        self.assertEqual(len(result["modes"]["dimsim-natural"]["rows"]), summary["draw_count"])

    def test_occurrence_mismatch_and_bundle_hash_fail(self):
        with self.assertRaisesRegex(ValueError, "inventory"):
            dim.score_contextual_dimsim(self.bundle, self.ids[:-1], self.hidden[:-1], self.prov,
                                        self.prov["bundle_manifest_sha256"])
        with self.assertRaisesRegex(ValueError, "provenance"):
            dim.score_contextual_dimsim(self.bundle, self.ids, self.hidden, self.prov, "0" * 64)

    def test_boundary_tie_break_and_undefined(self):
        # Isolate selection behavior with identical valid rows and a restricted threshold grid.
        result = self.score({"boundary_min_side_count": 1, "boundary_thresholds": [10, 20]})
        boundary = result["modes"]["dimsim-natural"]["summary"]["boundary"]
        if boundary["threshold"] is not None:
            tied = [x for x in boundary["curve"] if x["right_rho"] is not None]
            best = min(abs(x["right_rho"]) for x in tied)
            self.assertEqual(boundary["threshold"], min(x["threshold"] for x in tied if abs(x["right_rho"]) == best))
        undefined = self.score({"boundary_min_side_count": 10000, "boundary_thresholds": [10]})["modes"]["dimsim-natural"]["summary"]["boundary"]
        self.assertIsNone(undefined["threshold"])

    def synthetic_score(self, distances, dissimilarities, masks=None, config=None, model="syllable", seed=0, repeat_pair=False):
        count = len(distances)
        ids = [f"o{i:04d}" for i in range(2 * count)]
        hidden = np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (2 * count, 1))
        pairs = {mode: [] for mode in dim.MODES}
        for i, (distance, dissim) in enumerate(zip(distances, dissimilarities)):
            if repeat_pair:
                dissim = .1
            cosine = 1.0 - dissim
            j = 0 if repeat_pair else i
            hidden[2*j] = [1.0, 0.0]
            hidden[2*j+1] = [cosine, math.sqrt(max(0.0, 1.0-cosine*cosine))]
            for mode in dim.MODES:
                pair_j = 0 if repeat_pair and mode == "dimsim-balanced" else i
                left, right = ids[2*pair_j], ids[2*pair_j+1]
                row = {"draw_id": f"{mode}:v1:{i:06d}", "draw_index": i,
                    "pair_id": sampling.pair_identity(left, right), "left_id": left, "right_id": right,
                    "mode": "natural" if mode.endswith("natural") else "balanced",
                    "dimsim_distance": distance, "distance_status": "ok", "overlap_mask": (masks or ["000"]*count)[i],
                    "no_zero_initial_diagnostic": True}
                if mode == "dimsim-balanced":
                    row.update({"first_id": left, "second_id": right,
                                "first_syllable_id": f"syllable:{i}a", "second_syllable_id": f"syllable:{i}b"})
                pairs[mode].append(row)
        bundle = {"manifest": {"schema_version": 3}, "occurrences": [{"occurrence_id": x} for x in ids], "pairs": pairs}
        provenance = {"model_id": model, "seed": seed, "bundle_manifest_sha256": "b"*64}
        return dim.score_contextual_dimsim(bundle, ids, hidden, provenance, "b"*64,
                                           {"boundary_min_side_count": 1, **(config or {})})

    def test_known_average_rank_ties_and_constant_111_stratum(self):
        score = self.synthetic_score([1, 1, 2, 3], [.1, .1, .8, .4], ["111", "111", "010", "001"],
                                     {"boundary_thresholds": [10]})
        mode = score["modes"]["dimsim-natural"]["summary"]
        # Average ranks are [1.5, 1.5, 3, 4] and [1.5, 1.5, 4, 3].
        self.assertAlmostEqual(mode["rho"]["rho"], 7/9)
        self.assertIsNone(mode["strata"]["111"]["rho"])
        self.assertEqual(mode["strata"]["111"]["undefined_reason"], "constant_distance")

    def test_balanced_multiplicity_and_bin_support_99_100_gap(self):
        repeated = self.synthetic_score([4]*100, [.1 + i*.001 for i in range(100)], config={"boundary_thresholds": [10]}, repeat_pair=True)
        balanced = repeated["modes"]["dimsim-balanced"]["summary"]
        self.assertEqual(balanced["draw_count"], 100)
        self.assertEqual(balanced["unique_pair_count"], 1)
        self.assertEqual(balanced["bins"][0]["unique_pair_count"], 1)
        self.assertFalse(balanced["bins"][0]["supported"])
        one_short = self.synthetic_score([4]*99, [.1 + i*.001 for i in range(99)], config={"boundary_thresholds": [10]})
        self.assertEqual(one_short["modes"]["dimsim-natural"]["summary"]["bins"][0]["unique_pair_count"], 99)
        self.assertFalse(one_short["modes"]["dimsim-natural"]["summary"]["bins"][0]["supported"])

    def test_boundary_known_winner_tie_and_support_fraction(self):
        distances = [5, 25, 35, 45, 55, 65]
        score = self.synthetic_score(distances, [.1, .2, .3, .4, .5, .6],
                                     config={"boundary_thresholds": [10, 20, 30]})
        boundary = score["modes"]["dimsim-natural"]["summary"]["boundary"]
        self.assertEqual(boundary["threshold"], 10)  # Tied right tails choose the smaller threshold.
        self.assertEqual(boundary["min_side_count"], 1)
        self.assertEqual(score["contract"]["boundary_min_side_fraction"], .05)
        # Explicitly verify ceil(.05*N) raises the one-pair override to two.
        larger = self.synthetic_score(list(range(21)), [.1 + i*.01 for i in range(21)],
                                      config={"boundary_thresholds": [10]})
        self.assertEqual(larger["modes"]["dimsim-natural"]["summary"]["boundary"]["min_side_count"], 2)

    def test_boundary_median_requires_five_valid_boundaries(self):
        scores = []
        for seed in range(5):
            value = self.synthetic_score([5, 25, 35, 45, 55, 65], [.1, .2, .3, .4, .5, .6],
                config={"boundary_thresholds": [10, 20, 30]}, seed=seed)
            scores.append({"model_id": "syllable", "seed": seed, "score": value})
        good = report.build_dimsim_report({"schema_version": 1, "bundle_manifest_sha256": "b"*64, "results": scores})
        boundary = good["tables"]["model_boundary"][0]
        self.assertEqual(boundary["median"], 10)
        self.assertEqual([x["threshold"] for x in boundary["seed_values"]], [10]*5)
        undefined = []
        for seed in range(5):
            value = self.synthetic_score([5, 25, 35, 45, 55, 65], [.1, .2, .3, .4, .5, .6],
                config={"boundary_min_side_count": 20, "boundary_thresholds": [10, 20, 30]}, seed=seed)
            undefined.append({"model_id": "syllable", "seed": seed, "score": value})
        bad = report.build_dimsim_report({"schema_version": 1, "bundle_manifest_sha256": "b"*64, "results": undefined})
        boundary = bad["tables"]["model_boundary"][0]
        self.assertIsNone(boundary["median"])
        self.assertEqual(boundary["median_status"], "requires_all_five_valid")

    def test_report_missing_seed_keeps_values_and_status(self):
        scores = []
        for model in ("syllable", "components"):
            for seed in range(5):
                value = self.score({"boundary_min_side_count": 1, "min_unique_pairs_per_bin": 1})
                value["model_id"], value["seed"] = model, seed
                scores.append({"model_id": model, "seed": seed, "score": value})
        scores = [r for r in scores if not (r["model_id"] == "components" and r["seed"] == 4)]
        result = report.build_dimsim_report({"schema_version": 1, "bundle_manifest_sha256": self.prov["bundle_manifest_sha256"], "results": scores})
        row = next(x for x in result["tables"]["model_rho"] if x["model_id"] == "components")
        self.assertEqual(row["status"], "incomplete_five_valid_values")
        self.assertIsNone(row["mean"])
        json.dumps(result, allow_nan=False)

    def test_report_rejects_pair_identity_or_shared_contract_mismatch(self):
        records = []
        for model in ("syllable", "components"):
            for seed in range(5):
                value = self.synthetic_score([1, 2, 3, 4], [.1, .3, .2, .5],
                                             config={"boundary_thresholds": [10]}, model=model, seed=seed)
                records.append({"model_id": model, "seed": seed, "score": value})
        batch = {"schema_version": 1, "bundle_manifest_sha256": "b"*64, "results": records}
        report.build_dimsim_report(batch)
        bad = json.loads(json.dumps(batch))
        bad["results"][-1]["score"]["modes"]["dimsim-natural"]["rows"][0]["pair_id"] = "wrong-pair"
        with self.assertRaisesRegex(ValueError, "pair identity"):
            report.build_dimsim_report(bad)
        bad = json.loads(json.dumps(batch))
        bad["results"][-1]["score"]["contract"]["min_unique_pairs_per_bin"] = 999
        with self.assertRaisesRegex(ValueError, "analysis contract"):
            report.build_dimsim_report(bad)

    def test_invalid_norm_reports_occurrence_id_and_does_not_mutate(self):
        hidden = self.hidden.copy()
        before = hidden.copy()
        hidden[0] = 0
        with self.assertRaisesRegex(ValueError, self.ids[0]):
            dim.score_contextual_dimsim(self.bundle, self.ids, hidden, self.prov,
                                        self.prov["bundle_manifest_sha256"])
        self.assertTrue(np.array_equal(hidden[1:], before[1:]))
        self.assertTrue(np.array_equal(self.hidden, np.arange(1, len(self.ids)*3+1, dtype=np.float32).reshape(-1, 3)))

    def test_report_bin_gap_does_not_connect_across_unsupported_bins(self):
        distances = [4]*100 + [24]*99
        dissim = [.1 + (i % 50)*.001 for i in range(len(distances))]
        scores = []
        for seed in range(5):
            value = self.synthetic_score(distances, dissim,
                config={"min_unique_pairs_per_bin": 100, "boundary_thresholds": [10, 20]}, seed=seed)
            scores.append({"model_id": "syllable", "seed": seed, "score": value})
        result = report.build_dimsim_report({"schema_version": 1, "bundle_manifest_sha256": "b"*64, "results": scores})
        curves = result["plot_data"]["curves"]
        first = next(x for x in curves if x["bin"] == "[0,10)")
        gap = next(x for x in curves if x["bin"] == "[10,20)")
        second = next(x for x in curves if x["bin"] == "[20,30)")
        self.assertTrue(first["supported"])
        self.assertFalse(gap["supported"])
        self.assertFalse(first["connect_to_next"])
        self.assertFalse(second["supported"])

    def test_small_bundle_path_batch_two_models_five_seeds(self):
        source_hashes = self.bundle["manifest"]["source_experiment"]["hashes"]
        exp_hashes = dict(source_hashes, **{"acoustic_targets.npy": "a" * 64})
        runs, paths = {}, {}
        for model in ("syllable", "components"):
            run = self.root / "runs" / model / "run-explicit"; run.mkdir(parents=True)
            artifacts, statuses, seed_paths = {}, {}, {}
            for seed in range(5):
                sd = run / f"seed-{seed}"; sd.mkdir()
                ckpt = sd / "best.pt"; ckpt.write_bytes(f"{model}:{seed}".encode())
                artifacts[f"seed-{seed}/best.pt"] = sha(ckpt); statuses[str(seed)] = "complete"
                rp = self.root / "representations" / model / str(seed)
                provenance = {"model_id": model, "seed": seed, "checkpoint_sha256": sha(ckpt),
                              "bundle_manifest_sha256": sha(self.path / "manifest.json"),
                              "config_sha256": exp_hashes["config.json"], "vocab_sha256": exp_hashes["vocab.json"],
                              "source_sha256": exp_hashes["manifest.json"], "experiment_hashes": exp_hashes,
                              "environment": {k: "fixture" for k in ("python", "numpy", "torch", "platform", "device")},
                              "evaluation_mode": True}
                reps.save_contextual_representations(rp, occurrence_ids=self.ids, hidden=self.hidden, provenance=provenance)
                seed_paths[seed] = rp
            manifest = {"schema_version": 1, "run_type": "training_run", "status": "complete", "run_id": run.name,
                        "model_id": model, "seeds": list(range(5)), "seed_status": statuses,
                        "experiment_hashes": exp_hashes, "artifacts": artifacts}
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            runs[model], paths[model] = run, seed_paths
        batch = report.score_contextual_dimsim_batch(self.path, runs, paths, {"boundary_min_side_count": 1, "min_unique_pairs_per_bin": 1})
        self.assertEqual(len(batch["results"]), 10)
        final = report.build_dimsim_report(batch)
        self.assertEqual(len(final["tables"]["model_rho"]), 4)
        json.dumps(final, allow_nan=False)


if __name__ == "__main__": unittest.main()
