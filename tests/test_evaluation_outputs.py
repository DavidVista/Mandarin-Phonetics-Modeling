"""Synthetic persistence and recovery checks for numerical evaluation outputs."""
from __future__ import annotations

import importlib
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch

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
contract_module = importlib.import_module("phonetics.evaluation.report_contract")
outputs = importlib.import_module("phonetics.evaluation.evaluation_outputs")
extraction_tests = importlib.import_module("test_extraction_batch")
representation_module = importlib.import_module("phonetics.evaluation.representations")
random_tests = importlib.import_module("test_random_control")


class EvaluationOutputsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = batch_tests.AnchorQBatchTests(); self.fixture.setUp()
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        self.q = q_batcher.score_anchor_q_batch(self.fixture.bundle_path, self.fixture.runs, self.fixture.paths)
        self.d = d_batcher.score_contextual_dimsim_batch(self.fixture.bundle_path, self.fixture.runs, self.fixture.paths,
            {"boundary_min_side_count": 1, "min_unique_pairs_per_bin": 1})
        self.ids = {"contextual-anchor-q": "q-out-1", "contextual-dimsim": "d-out-1"}
        self.contract = contract_module.build_evaluation_report(self.q, self.d, "components", self.ids)

    def tearDown(self):
        self.fixture.tearDown(); self.temp.cleanup()

    def _write(self):
        return outputs.write_evaluation_outputs(self.root, self.contract, run_ids=self.ids,
            input_roots=[self.fixture.bundle_path], source_hashes={"evaluation.py": "a" * 64},
            resolved_config={"reference_model": "components", "export_format": "png"})

    def test_roundtrip_exact_contract_and_no_overwrite(self):
        before_q, before_d, before_config = copy.deepcopy(self.q), copy.deepcopy(self.d), {"reference_model": "components", "export_format": "png"}
        paths = self._write()
        for analysis, root in paths.items():
            loaded = outputs.load_evaluation_outputs(root)
            self.assertEqual(loaded["report"], self.contract["reports"][analysis])
            self.assertEqual(loaded["batch"], self.contract["raw_batches"][analysis])
            self.assertEqual(loaded["tables"], loaded["report"]["tables"])
            self.assertEqual(loaded["manifest"]["status"], "complete")
            self.assertEqual(loaded["manifest"]["figure_exports"], "pending")
            with self.assertRaises(FileExistsError): self._write()
        self.assertEqual(self.q, before_q); self.assertEqual(self.d, before_d)
        self.assertEqual(before_config, {"reference_model": "components", "export_format": "png"})

    def test_tamper_and_unexpected_file_are_rejected(self):
        root = self._write()["contextual-anchor-q"]
        batch = root / "data/anchor_q_batch.json"
        batch.write_text(batch.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
            outputs.load_evaluation_outputs(root)

    def test_input_overlap_preflight_leaves_no_outputs(self):
        bad_inputs = [self.fixture.bundle_path, self.root]
        with self.assertRaises(ValueError):
            outputs.write_evaluation_outputs(self.root, self.contract, run_ids=self.ids, input_roots=bad_inputs,
                source_hashes={"evaluation.py": "a" * 64}, resolved_config={})
        self.assertFalse((self.root / "outputs").exists())

    def test_bad_bundle_identity_preflight_creates_no_output(self):
        wrong = self.root / "wrong-input"; wrong.mkdir(); (wrong / "manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bundle manifest pin"):
            outputs.write_evaluation_outputs(self.root, self.contract, run_ids=self.ids, input_roots=[wrong],
                source_hashes={"evaluation.py": "a" * 64}, resolved_config={})
        self.assertFalse((self.root / "outputs").exists())

    def test_second_analysis_write_failure_keeps_first_complete_and_second_failed(self):
        original = outputs._write
        def fail_second(path, value):
            if Path(path).name == "dimsim_batch.json": raise OSError("injected write failure")
            return original(path, value)
        with patch.object(outputs, "_write", side_effect=fail_second), self.assertRaisesRegex(OSError, "injected"):
            self._write()
        first = self.root / "outputs/contextual-anchor-q/q-out-1/manifest.json"
        second = self.root / "outputs/contextual-dimsim/d-out-1/manifest.json"
        self.assertEqual(json.loads(first.read_text(encoding="utf-8"))["status"], "complete")
        self.assertEqual(json.loads(second.read_text(encoding="utf-8"))["status"], "failed")
        with self.assertRaises(FileExistsError): self._write()

    def test_inventory_rejects_unexpected_missing_and_empty_files(self):
        for name, operation in (("unexpected", lambda p: p.write_text("x", encoding="utf-8")),
                                ("missing", lambda p: p.unlink()),
                                ("empty", lambda p: p.write_bytes(b""))):
            with self.subTest(kind=name):
                root = self._write()["contextual-anchor-q"]
                target = root / "data/anchor_q_batch.json"
                if name == "unexpected": target = root / "unexpected.bin"
                operation(target)
                with self.assertRaises(ValueError): outputs.load_evaluation_outputs(root)
                # Unique destinations for the next subcase.
                self.ids["contextual-anchor-q"] += "x"
                self.ids["contextual-dimsim"] += "x"


class ExtractionPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = extraction_tests.ExtractionBatchTests(); self.fixture.setUp()

    def tearDown(self): self.fixture.tearDown()

    def _fake(self, *, wrong=False):
        f = self.fixture
        def extract(**kw):
            model, seed = kw["expected_model_id"], kw["expected_seed"]
            provenance = {"model_id": model, "seed": seed, "checkpoint_sha256": extraction_tests.sha(kw["checkpoint_path"]),
                "bundle_manifest_sha256": f.bundle_hash, "experiment_hashes": f.source_six,
                "config_sha256": f.source_six["config.json"], "vocab_sha256": f.source_six["vocab.json"],
                "source_sha256": "0" * 64 if wrong else "f" * 64,
                "environment": {"python": "3.12", "numpy": __import__("numpy").__version__, "torch": "fake", "platform": "test", "device": "cpu"},
                "evaluation_mode": True}
            representation_module.save_contextual_representations(kw["output_dir"], occurrence_ids=f.bundle_ids,
                hidden=__import__("numpy").ones((len(f.bundle_ids), 3), dtype="float32"), provenance=provenance)
        return extract

    def test_successful_export_is_gated_after_nested_manifest_validation(self):
        result = outputs.run_contextual_extraction_outputs(self.fixture.plan, extractor=self._fake())
        first = result["exports"][0]["representation_dir"]
        loaded = outputs.load_contextual_extraction_outputs(Path(first).parent)
        self.assertEqual(loaded["manifest"]["status"], "complete")
        self.assertEqual(loaded["diagnostics"]["nested_manifest_sha256"], extraction_tests.sha(Path(first) / "manifest.json"))

    def test_postvalidation_failure_preserves_failed_manifest_and_raw_export(self):
        with self.assertRaisesRegex(ValueError, "provenance or occurrence inventory mismatch"):
            outputs.run_contextual_extraction_outputs(self.fixture.plan, extractor=self._fake(wrong=True))
        item = self.fixture.plan["inventory"][0]
        root = Path(item["output_run_dir"])
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue((root / "data/manifest.json").is_file())

    def test_nested_manifest_tamper_and_outer_identity_mismatch_rejected(self):
        result = outputs.run_contextual_extraction_outputs(self.fixture.plan, extractor=self._fake())
        root = Path(result["exports"][0]["representation_dir"]).parent
        nested = root / "data/manifest.json"
        value = json.loads(nested.read_text(encoding="utf-8")); value["provenance"]["model_id"] = "components"
        nested.write_text(json.dumps(value), encoding="utf-8")
        outer_path = root / "manifest.json"
        outer = json.loads(outer_path.read_text(encoding="utf-8"))
        row = next(x for x in outer["artifacts"] if x["path"] == "data/manifest.json")
        row["sha256"] = outputs.file_sha256(nested); row["bytes"] = nested.stat().st_size
        outer_path.write_text(json.dumps(outer), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nested contextual manifest pin mismatch"):
            outputs.load_contextual_extraction_outputs(root)


class RandomPersistenceTests(unittest.TestCase):
    def test_control_recovery_preserves_arrays_without_learned_loading(self):
        fixture = random_tests.RandomControlTests(); fixture.setUp()
        temp = tempfile.TemporaryDirectory(dir=ROOT)
        try:
            control_mod = importlib.import_module("phonetics.evaluation.random_control")
            generated = control_mod.generate_random_control(fixture.ids, dimension=3)
            evaluated = control_mod.evaluate_random_control(fixture.bundle, generated,
                bundle_manifest_sha256=fixture.bundle_hash,
                dimension_resolution=fixture._resolution(fixture.ids, 3), source_sha256="b" * 64,
                analysis_config={"boundary_min_side_count": 1})
            original_ids = list(generated["occurrence_ids"])
            original_matrix = generated["matrices"][0].copy()
            root = outputs.write_random_control_outputs(Path(temp.name), "random-01", evaluated,
                input_roots=[fixture.path], source_hashes={"random_control.py": "c" * 64},
                resolved_config={"analysis": "matched_random_control"})
            recovered = outputs.load_random_control_outputs(root)
            self.assertEqual(recovered["report"], evaluated["report"])
            for seed in range(5): self.assertTrue(np.array_equal(recovered["matrices"][seed], evaluated["matrices"][seed]))
            self.assertEqual(generated["occurrence_ids"], original_ids)
            self.assertTrue(np.array_equal(generated["matrices"][0], original_matrix))
            # A finite, shape-correct but stale matrix must fail the deterministic PCG64 binding.
            bad = {"occurrence_ids": list(evaluated["occurrence_ids"]), "matrices": dict(evaluated["matrices"]), "report": evaluated["report"]}
            bad["matrices"][0] = np.ones_like(bad["matrices"][0])
            with self.assertRaisesRegex(ValueError, "PCG64"):
                outputs.write_random_control_outputs(Path(temp.name), "random-bad", bad, input_roots=[fixture.path],
                    source_hashes={"random_control.py": "c" * 64}, resolved_config={})
            stale = dict(evaluated); stale["report"] = dict(evaluated["report"]); stale["report"]["anchor_q"] = {}
            with self.assertRaisesRegex(ValueError, "stale"):
                outputs.write_random_control_outputs(Path(temp.name), "random-stale", stale, input_roots=[fixture.path],
                    source_hashes={"random_control.py": "c" * 64}, resolved_config={})
        finally:
            fixture.tearDown(); temp.cleanup()


if __name__ == "__main__": unittest.main()
