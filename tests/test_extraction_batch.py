"""Synthetic batch contract tests; no checkpoint tensors or model inference are used."""
import ast
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phonetics.evaluation.extraction_batch import (
    execute_contextual_extraction, plan_contextual_extraction, resolve_encoder_source_hash,
)
from phonetics.evaluation.bundle import prepare_evaluation_bundle
from phonetics.evaluation.representations import load_contextual_representations, save_contextual_representations


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExtractionBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp.name)
        sys.path.insert(0, str(ROOT / "tests"))
        from test_contextual_bundle import ContextualBundleTests
        fixture = ContextualBundleTests()
        self.experiment = fixture.make_experiment(self.root, "experiment")
        np.save(self.experiment / "acoustic_targets.npy", np.arange(4, dtype=np.float32), allow_pickle=False)
        manifest_path = self.experiment / "manifest.json"
        experiment_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        experiment_manifest["artifacts"]["acoustic_targets.npy"] = sha(self.experiment / "acoustic_targets.npy")
        manifest_path.write_text(json.dumps(experiment_manifest), encoding="utf-8")
        self.exp_hash = sha(manifest_path)
        self.source_six = {"manifest.json": self.exp_hash, **experiment_manifest["artifacts"]}
        self.bundle = prepare_evaluation_bundle("synthetic-bundle", experiment_path=self.experiment,
            output_root=self.root / "bundles", distance_fn=lambda a, b: 0.0 if a == b else 20.0,
            metric_provenance={"metric_name": "synthetic"}, code_provenance={"fixture": "0" * 64},
            protocol_config={"dimsim_natural_count": 20, "dimsim_balanced_count": 24,
                             "anchor_q_draws_per_edge": 2})
        self.bundle_hash = sha(self.bundle / "manifest.json")
        self.bundle_data = None
        self.bundle_schema = json.loads((self.bundle / "manifest.json").read_text(encoding="utf-8"))["schema_version"]
        self.bundle_ids = sorted(json.loads(row)["occurrence_id"] for row in
                                 (self.bundle / "occurrences.jsonl").read_text(encoding="utf-8").splitlines() if row)
        self.models = ("syllable", "components")
        self.runs, self.ids = {}, {}
        for model in self.models:
            run = self.root / "train" / model / f"run-{model}"
            run.mkdir(parents=True)
            artifacts = {}
            for seed in range(5):
                folder = run / f"seed-{seed}"; folder.mkdir()
                checkpoint = folder / "best.pt"; checkpoint.write_bytes(f"{model}-{seed}".encode())
                artifacts[f"seed-{seed}/best.pt"] = sha(checkpoint)
            manifest = {"schema_version": 1, "run_type": "training_run", "status": "complete",
                        "run_id": run.name, "model_id": model, "seeds": list(range(5)),
                        "seed_status": {str(i): "complete" for i in range(5)}, "artifacts": artifacts,
                        "experiment_hashes": self.source_six}
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.runs[model] = run
            self.ids[model] = {seed: f"{model}-seed-{seed}" for seed in range(5)}
        self.plan = self.make_plan()

    def tearDown(self):
        self.temp.cleanup()

    def make_plan(self, *, runs=None, ids=None, output=None):
        stub = {"manifest": {"schema_version": 3, "source_experiment": {"hashes":
                    {k: v for k, v in self.source_six.items() if k != "acoustic_targets.npy"}}},
                "occurrences": [{"occurrence_id": x} for x in self.bundle_ids]}
        with patch("phonetics.evaluation.extraction_batch.load_evaluation_bundle", return_value=stub):
            return plan_contextual_extraction(bundle_path=self.bundle, bundle_manifest_sha256=self.bundle_hash,
                experiment_path=self.experiment, experiment_manifest_sha256=self.exp_hash,
                training_runs=runs or self.runs, run_ids=ids or self.ids,
                output_root=output or self.root / "outputs", source_sha256="f" * 64)

    def test_plan_is_deterministic_json_compatible_and_has_separate_data_paths(self):
        one = self.make_plan(); two = self.make_plan()
        self.assertEqual(one, two)
        self.assertEqual(json.loads(json.dumps(one)), one)
        self.assertEqual(len(one["inventory"]), 10)
        for item in one["inventory"]:
            self.assertEqual(Path(item["output_dir"]), Path(item["output_run_dir"]) / "data")
            self.assertEqual(item["training_run_path"], str(self.runs[item["model_id"]].resolve()))
            self.assertTrue(item["training_manifest_sha256"])
        self.assertFalse((self.root / "outputs").exists())

    def test_tiny_synthetic_fixture_is_schema_three(self):
        self.assertEqual(self.bundle_schema, 3)

    def test_successful_two_model_five_seed_export_and_shared_raw_representations(self):
        calls = []
        def fake_extract(**kwargs):
            calls.append(kwargs)
            model, seed = kwargs["expected_model_id"], kwargs["expected_seed"]
            provenance = {"model_id": model, "seed": seed,
                "checkpoint_sha256": sha(kwargs["checkpoint_path"]), "bundle_manifest_sha256": self.bundle_hash,
                "experiment_hashes": self.source_six, "config_sha256": self.source_six["config.json"],
                "vocab_sha256": self.source_six["vocab.json"], "source_sha256": "f" * 64,
                "environment": {"python": "3.12", "numpy": np.__version__, "torch": "synthetic",
                                "platform": "test", "device": "cpu"}, "evaluation_mode": True}
            values = np.arange(len(self.bundle_ids) * 3, dtype=np.float64).reshape(len(self.bundle_ids), 3) + 1 + seed
            save_contextual_representations(kwargs["output_dir"], occurrence_ids=self.bundle_ids,
                                            hidden=values, provenance=provenance)
        stub = {"manifest": {"schema_version": 3, "source_experiment": {"hashes":
                    {k: v for k, v in self.source_six.items() if k != "acoustic_targets.npy"}}},
                "occurrences": [{"occurrence_id": x} for x in self.bundle_ids]}
        metadata = execute_contextual_extraction(self.plan, extractor=fake_extract)
        self.assertEqual(len(calls), 10)
        self.assertEqual([(c["expected_model_id"], c["expected_seed"]) for c in calls],
                         [(i["model_id"], i["seed"]) for i in self.plan["inventory"]])
        for call in calls:
            self.assertTrue(call["contextual_only"])
            self.assertEqual(call["batch_size"], 32)
            self.assertEqual(call["source_sha256"], "f" * 64)
        result = metadata["representation_dirs"]
        self.assertEqual(set(result), set(self.models))
        for model in self.models:
            self.assertEqual(set(result[model]), set(range(5)))
            q_input = result[model][0]
            dim_input = result[model][0]
            self.assertEqual(q_input, dim_input)
            loaded = load_contextual_representations(q_input)
            self.assertEqual(loaded["hidden"].dtype, np.float64)
            self.assertEqual(loaded["occurrence_ids"], self.bundle_ids)
        from phonetics.evaluation.anchor_q_batch import score_anchor_q_batch
        from phonetics.evaluation.dimsim_reporting import score_contextual_dimsim_batch
        q_batch = score_anchor_q_batch(self.bundle, self.runs, result)
        dim_batch = score_contextual_dimsim_batch(self.bundle, self.runs, result)
        self.assertEqual(len(q_batch["results"]), 10)
        self.assertEqual(len(dim_batch["results"]), 10)
        self.assertEqual(len(metadata["exports"]), 10)
        for export in metadata["exports"]:
            self.assertEqual(export["manifest_sha256"], sha(Path(export["representation_dir"]) / "manifest.json"))
        self.assertNotIn("representation_manifest_sha256", self.plan["inventory"][0])

    def test_missing_last_seed_and_stale_run_manifest_fail_before_any_callback(self):
        calls = []
        last = self.runs["components"] / "seed-4/best.pt"
        last.unlink()
        with self.assertRaises(ValueError):
            self.make_plan()
        self.assertEqual(calls, [])
        # Restore checkpoint then mutate run-manifest pin after creating a valid plan.
        last.write_bytes(b"components-4")
        manifest = self.runs["components"] / "manifest.json"
        manifest.write_text(manifest.read_text() + " ")
        stub = {"manifest": {"schema_version": 3, "source_experiment": {"hashes":
                    {k: v for k, v in self.source_six.items() if k != "acoustic_targets.npy"}}}, "occurrences": []}
        with patch("phonetics.evaluation.extraction_batch.load_evaluation_bundle", return_value=stub):
            with self.assertRaises(ValueError):
                execute_contextual_extraction(self.plan, extractor=lambda **kw: calls.append(kw))
        self.assertEqual(calls, [])

    def test_collision_unsafe_id_preexisting_run_and_input_overlap_rejected(self):
        ids = {m: dict(v) for m, v in self.ids.items()}
        ids["components"][4] = ids["syllable"][4]
        with self.assertRaisesRegex(ValueError, "collision"):
            self.make_plan(ids=ids)
        ids["components"][4] = "CON"
        with self.assertRaisesRegex(ValueError, "unsafe"):
            self.make_plan(ids=ids)
        first = self.plan["inventory"][0]["output_run_dir"]
        Path(first).mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.make_plan()
        with self.assertRaisesRegex(ValueError, "overlaps"):
            self.make_plan(output=self.experiment / "sub")
        overlap = self.make_plan(output=self.root / "fresh-outputs")
        # Mutating output_root to an experiment ancestor is caught by plan reconstruction.
        overlap["output_root"] = str(self.experiment.parent)
        calls = []
        stub = {"manifest": {"schema_version": 3, "source_experiment": {"hashes":
                    {k: v for k, v in self.source_six.items() if k != "acoustic_targets.npy"}}}, "occurrences": []}
        with patch("phonetics.evaluation.extraction_batch.load_evaluation_bundle", return_value=stub):
            with self.assertRaises(ValueError):
                execute_contextual_extraction(overlap, extractor=lambda **kw: calls.append(kw))
        self.assertEqual(calls, [])

    def test_json_string_seed_keys_and_model_aliases_rejected(self):
        ids = {m: dict(v) for m, v in self.ids.items()}
        ids["syllable"] = {str(k): v for k, v in ids["syllable"].items()}
        with self.assertRaisesRegex(ValueError, "integer seed keys"):
            self.make_plan(ids=ids)
        alias = dict(self.runs); alias["components-acoustic-lambda-1.0"] = alias.pop("components")
        with self.assertRaises(ValueError):
            self.make_plan(runs=alias)

    def test_wrong_export_source_hash_and_occurrence_inventory_are_rejected(self):
        for bad_kind in ("source_hash", "occurrence_ids"):
            with self.subTest(bad_kind=bad_kind):
                plan = self.make_plan(output=self.root / f"bad-{bad_kind}")
                calls = []
                stub = {"manifest": {"schema_version": 3, "source_experiment": {"hashes":
                            {k: v for k, v in self.source_six.items() if k != "acoustic_targets.npy"}}},
                        "occurrences": [{"occurrence_id": x} for x in self.bundle_ids]}
                def fake(**kwargs):
                    calls.append(kwargs)
                    prov = {"model_id": kwargs["expected_model_id"], "seed": kwargs["expected_seed"],
                        "checkpoint_sha256": sha(kwargs["checkpoint_path"]), "bundle_manifest_sha256": self.bundle_hash,
                        "experiment_hashes": self.source_six, "config_sha256": self.source_six["config.json"],
                        "vocab_sha256": self.source_six["vocab.json"],
                        "source_sha256": ("e" * 64 if bad_kind == "source_hash" else "f" * 64),
                        "environment": {"python": "3.12", "numpy": np.__version__, "torch": "synthetic",
                                        "platform": "test", "device": "cpu"}, "evaluation_mode": True}
                    ids = self.bundle_ids if bad_kind != "occurrence_ids" else self.bundle_ids[:-1]
                    mat = np.ones((len(ids), 2), dtype=np.float32)
                    save_contextual_representations(kwargs["output_dir"], occurrence_ids=ids, hidden=mat,
                                                    provenance=prov)
                with patch("phonetics.evaluation.extraction_batch.load_evaluation_bundle", return_value=stub):
                    with self.assertRaisesRegex(ValueError, "export provenance or occurrence inventory"):
                        execute_contextual_extraction(plan, extractor=fake)
                self.assertEqual(len(calls), 1)

    def test_source_hash_helper_and_encoder_explicit_branch_need_no_file_lookup(self):
        self.assertEqual(resolve_encoder_source_hash({"source_sha256": {"evaluation/encoders.py": "a" * 64}}), "a" * 64)
        source = (ROOT / "src/phonetics/evaluation/encoders.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "extract_bilstm")
        source_lines = source.splitlines()
        body = "\n".join(source_lines[fn.lineno - 1:fn.end_lineno])
        self.assertIn('source_sha256 or _sha256(Path(__file__).resolve())', body)
        self.assertIn('if source_sha256 is not None', body)


if __name__ == "__main__":
    unittest.main()
