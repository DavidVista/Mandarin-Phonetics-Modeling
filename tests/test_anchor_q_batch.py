import hashlib
import importlib
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "phonetics"
phonetics = types.ModuleType("phonetics"); phonetics.__path__ = [str(SOURCE)]
evaluation = types.ModuleType("phonetics.evaluation"); evaluation.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics); sys.modules.setdefault("phonetics.evaluation", evaluation)
fixture = importlib.import_module("test_contextual_bundle")
reps = importlib.import_module("phonetics.evaluation.representations")
batch = importlib.import_module("phonetics.evaluation.anchor_q_batch")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(run_dir):
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(run_dir, value):
    (run_dir / "manifest.json").write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class AnchorQBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        helper = fixture.ContextualBundleTests()
        self.bundle_path = helper.prepare(self.root, "batch", q_draws=2)
        self.bundle = importlib.import_module("phonetics.evaluation.bundle").load_evaluation_bundle(self.bundle_path)
        source_hashes = self.bundle["manifest"]["source_experiment"]["hashes"]
        self.exp_hashes = dict(source_hashes, **{"acoustic_targets.npy": "a" * 64})
        self.runs, self.paths = {}, {}
        for model in ("components", "acoustic"):
            run_dir = self.root / "runs" / model / "run-explicit"
            run_dir.mkdir(parents=True)
            artifacts, seed_status, seeds = {}, {}, list(range(5))
            model_paths = {}
            for seed in seeds:
                seed_dir = run_dir / f"seed-{seed}"; seed_dir.mkdir()
                checkpoint = seed_dir / "best.pt"; checkpoint.write_bytes(f"fake checkpoint {model} {seed}".encode())
                artifacts[f"seed-{seed}/best.pt"] = sha(checkpoint)
                seed_status[str(seed)] = "complete"
                rep_path = self.root / "representations" / model / str(seed)
                provenance = {"model_id": model, "seed": seed, "checkpoint_sha256": sha(checkpoint),
                              "bundle_manifest_sha256": sha(self.bundle_path / "manifest.json"), "config_sha256": self.exp_hashes["config.json"],
                              "vocab_sha256": self.exp_hashes["vocab.json"], "source_sha256": self.exp_hashes["manifest.json"],
                              "experiment_hashes": self.exp_hashes,
                              "environment": {key: "fixture" for key in ("python", "numpy", "torch", "platform", "device")},
                              "evaluation_mode": True}
                reps.save_contextual_representations(rep_path,
                    occurrence_ids=sorted(row["occurrence_id"] for row in self.bundle["occurrences"]),
                    hidden=np.arange(1, len(self.bundle["occurrences"]) * 3 + 1, dtype=np.float32).reshape(-1, 3),
                    provenance=provenance)
                model_paths[seed] = rep_path
            manifest = {"schema_version": 1, "run_type": "training_run", "status": "complete", "run_id": run_dir.name,
                        "model_id": model, "seeds": seeds, "seed_status": seed_status,
                        "experiment_hashes": self.exp_hashes, "artifacts": artifacts}
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.runs[model] = run_dir
            self.paths[model] = model_paths

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_explicit_runs_score_all_ten_seed_representations(self):
        run_snapshot = dict(self.runs); path_snapshot = {k: dict(v) for k, v in self.paths.items()}
        result = batch.score_anchor_q_batch(self.bundle_path, self.runs, self.paths)
        self.assertEqual(len(result["results"]), 10)
        self.assertEqual([(r["model_id"], r["seed"]) for r in result["results"]],
                         [(model, seed) for model in sorted(self.runs) for seed in range(5)])
        self.assertTrue(all(len(row["score"]["draw_scores"]) == 24 for row in result["results"]))
        self.assertEqual(json.dumps(result, sort_keys=True), json.dumps(batch.score_anchor_q_batch(self.bundle_path, self.runs, self.paths), sort_keys=True))
        self.assertEqual(self.runs, run_snapshot); self.assertEqual(self.paths, path_snapshot)

    def test_missing_seed_and_wrong_provenance_rejected(self):
        with self.assertRaises(ValueError):
            batch.score_anchor_q_batch(self.bundle_path, self.runs, {**self.paths, "components": {0: self.paths["components"][0]}})
        original = self.paths["components"][0]
        self.paths["components"][0] = self.paths["components"][1]
        with self.assertRaises(ValueError):
            batch.score_anchor_q_batch(self.bundle_path, self.runs, self.paths)
        self.paths["components"][0] = original

    def test_checkpoint_hash_or_run_completion_mismatch_rejected(self):
        checkpoint = self.runs["components"] / "seed-0" / "best.pt"
        checkpoint.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            batch.score_anchor_q_batch(self.bundle_path, self.runs, self.paths)

    def test_discovery_rejects_bool_schema_and_seed_ids(self):
        from phonetics.evaluation.seed_runs import discover_seed_runs
        run = self.runs["components"]
        manifest = read_manifest(run); manifest["schema_version"] = True; write_manifest(run, manifest)
        with self.assertRaisesRegex(ValueError, "identity"):
            discover_seed_runs(run, "components")
        manifest["schema_version"] = 1; manifest["seeds"][0] = True; write_manifest(run, manifest)
        with self.assertRaisesRegex(ValueError, "seed inventory"):
            discover_seed_runs(run, "components")

    def test_discovery_rejects_bad_seed_directories_status_model_and_missing_checkpoint(self):
        from phonetics.evaluation.seed_runs import discover_seed_runs
        run = self.runs["components"]
        (run / "seed-9").mkdir()
        with self.assertRaisesRegex(ValueError, "directory inventory"):
            discover_seed_runs(run, "components")
        (run / "seed-9").rmdir()
        (run / "seed-3").rename(run / "seed-3-hidden")
        with self.assertRaisesRegex(ValueError, "directory inventory"):
            discover_seed_runs(run, "components")
        (run / "seed-3-hidden").rename(run / "seed-3")
        manifest = read_manifest(run); manifest["seed_status"]["4"] = "failed"; write_manifest(run, manifest)
        with self.assertRaisesRegex(ValueError, "complete status"):
            discover_seed_runs(run, "components")
        manifest["seed_status"]["4"] = "complete"; manifest["model_id"] = "acoustic"; write_manifest(run, manifest)
        with self.assertRaisesRegex(ValueError, "identity"):
            discover_seed_runs(run, "components")
        manifest["model_id"] = "components"; write_manifest(run, manifest)
        manifest["status"] = "partial"; write_manifest(run, manifest)
        with self.assertRaisesRegex(ValueError, "completion status"):
            discover_seed_runs(run, "components")
        manifest["status"] = "complete"; write_manifest(run, manifest)
        checkpoint = run / "seed-2" / "best.pt"; checkpoint.unlink()
        with self.assertRaisesRegex(ValueError, "model components, seed 2"):
            discover_seed_runs(run, "components")

    def test_missing_representation_error_names_model_seed_and_path(self):
        paths = {"components": dict(self.paths["components"])}
        missing = self.root / "absent-representation"
        paths["components"][2] = missing
        with self.assertRaisesRegex(ValueError, r"model components, seed 2, path .*absent-representation"):
            batch.score_anchor_q_batch(self.bundle_path, {"components": self.runs["components"]}, paths)

    def test_duplicate_canonical_model_keys_rejected_in_both_inputs(self):
        aliases = {"components-acoustic-lambda-1": self.runs["components"],
                   "components-acoustic-lambda-1.0": self.runs["acoustic"]}
        with self.assertRaisesRegex(ValueError, "inventories"):
            batch.score_anchor_q_batch(self.bundle_path, aliases, aliases)
        reps_aliases = {"components": self.paths["components"], **{
            "components-acoustic-lambda-1": self.paths["components"],
            "components-acoustic-lambda-1.0": self.paths["acoustic"]}}
        with self.assertRaisesRegex(ValueError, "inventories"):
            batch.score_anchor_q_batch(self.bundle_path, {"components": self.runs["components"],
                "components-acoustic-lambda-1": self.runs["acoustic"]}, reps_aliases)

    def test_source_mismatch_and_valid_wrong_checkpoint_provenance_rejected(self):
        run = self.runs["components"]
        manifest = read_manifest(run); manifest["experiment_hashes"]["config.json"] = "c" * 64; write_manifest(run, manifest)
        with self.assertRaisesRegex(ValueError, "source hashes"):
            batch.score_anchor_q_batch(self.bundle_path, {"components": run}, {"components": self.paths["components"]})
        # Independently make a checksum-valid representation with the wrong checkpoint pin.
        manifest["experiment_hashes"] = self.exp_hashes; write_manifest(run, manifest)
        loaded = reps.load_contextual_representations(self.paths["components"][0])
        provenance = dict(loaded["manifest"]["provenance"], checkpoint_sha256="d" * 64)
        wrong_path = self.root / "representations" / "wrong-checkpoint"
        reps.save_contextual_representations(wrong_path, occurrence_ids=loaded["occurrence_ids"],
                                              hidden=loaded["hidden"], provenance=provenance)
        paths = {"components": dict(self.paths["components"])}; paths["components"][0] = wrong_path
        with self.assertRaisesRegex(ValueError, "checkpoint mismatch: components seed 0"):
            batch.score_anchor_q_batch(self.bundle_path, {"components": run}, paths)
        bad_bundle_path = self.root / "representations" / "wrong-bundle"
        provenance["checkpoint_sha256"] = sha(run / "seed-0" / "best.pt")
        provenance["bundle_manifest_sha256"] = "e" * 64
        reps.save_contextual_representations(bad_bundle_path, occurrence_ids=loaded["occurrence_ids"],
                                              hidden=loaded["hidden"], provenance=provenance)
        paths["components"][0] = bad_bundle_path
        with self.assertRaisesRegex(ValueError, "bundle manifest hash mismatch"):
            batch.score_anchor_q_batch(self.bundle_path, {"components": run}, paths)

    def test_path_escape_is_rejected(self):
        from phonetics.evaluation.seed_runs import discover_seed_runs
        run = self.runs["components"]
        checkpoint = run / "seed-0" / "best.pt"
        outside = self.root / "outside.pt"; outside.write_bytes(checkpoint.read_bytes())
        original = Path.resolve
        def resolve_with_escape(path, *args, **kwargs):
            if path == checkpoint:
                return outside
            return original(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolve_with_escape):
            with self.assertRaisesRegex(ValueError, "escapes training run"):
                discover_seed_runs(run, "components")


if __name__ == "__main__":
    unittest.main()
