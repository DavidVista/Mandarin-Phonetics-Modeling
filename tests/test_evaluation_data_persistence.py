"""Numerical persistence is independent of plotting, fonts and model inference."""
from __future__ import annotations

import builtins
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from test_extraction_batch import ExtractionBatchTests
from phonetics.evaluation import notebook_runtime as runtime
from phonetics.evaluation.evaluation_outputs import (
    _verify_inventory, load_evaluation_outputs, load_random_control_outputs,
    load_random_control_summary,
)
from phonetics.evaluation.representations import save_contextual_representations


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DataPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = f = ExtractionBatchTests()
        f.setUp()
        for model in runtime.MODEL_ORDER:
            if model in f.runs:
                continue
            path = f.root / "train" / model / f"run-{model}"
            shutil.copytree(f.runs["syllable"], path)
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            manifest.update(model_id=model, run_id=path.name)
            (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            f.runs[model] = path
        config = {
            "bundle_path": f.bundle, "bundle_manifest_sha256": f.bundle_hash,
            "experiment_path": f.experiment, "experiment_manifest_sha256": f.exp_hash,
            "model_run_paths": f.runs, "extraction_mode": "extract",
            "extraction_prefix": "data-stage", "existing_representation_runs": None,
            "project_root": f.root / "prepared", "reference_model": "syllable",
            "run_ids": {name: "data-run" for name in
                        ("contextual-anchor-q", "contextual-dimsim", "contextual-random-control")},
            "device": "cpu", "batch_size": 32,
            "dimsim_boundary_start": 10, "dimsim_boundary_stop": 110, "dimsim_boundary_step": 1,
            "dimsim_min_side_count": 1, "dimsim_min_unique_pairs_per_bin": 1,
        }
        build = {"source_sha256": {"evaluation/encoders.py": "f" * 64,
                                    "evaluation/random_control.py": "b" * 64}}
        cls.context = runtime.preflight_evaluation(config, build)

        def extractor(**kw):
            provenance = {
                "model_id": kw["expected_model_id"], "seed": kw["expected_seed"],
                "checkpoint_sha256": sha(kw["checkpoint_path"]),
                "bundle_manifest_sha256": f.bundle_hash, "experiment_hashes": f.source_six,
                "config_sha256": f.source_six["config.json"], "vocab_sha256": f.source_six["vocab.json"],
                "source_sha256": "f" * 64, "evaluation_mode": True,
                "environment": {key: "synthetic" for key in ("python", "numpy", "torch", "platform", "device")},
            }
            matrix = np.arange(1, len(f.bundle_ids) * 3 + 1, dtype=np.float32).reshape(-1, 3)
            save_contextual_representations(kw["output_dir"], occurrence_ids=f.bundle_ids,
                                            hidden=matrix, provenance=provenance)

        cls.reps = runtime.extract_or_reuse(cls.context, extractor=extractor)
        cls.q = runtime.score_anchor_q(cls.context, cls.reps)
        cls.d = runtime.score_dimsim(cls.context, cls.reps)
        cls.random = runtime.score_random_control(cls.context, cls.reps)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDown()

    def setUp(self):
        self.context = {**self.__class__.context, "config": copy.deepcopy(self.__class__.context["config"])}
        self.context["project_root"] = self.fixture.root / self._testMethodName
        self.context["config"]["project_root"] = self.context["project_root"]
        # Equal names in different directories must never be treated as path overlap.
        self.context["config"]["run_ids"] = {
            name: self.fixture.bundle.name for name in self.context["config"]["run_ids"]
        }

    def persist(self):
        return runtime.persist_results(self.context, self.q, self.d, self.random)

    def test_data_stage_needs_no_plotting_settings_or_imports(self):
        self.assertNotIn("presentation_root", self.context)
        original = builtins.__import__

        def no_plotting(name, *args, **kwargs):
            if name.startswith(("matplotlib", "phonetics.visualization")):
                raise ImportError("Plotting is unavailable in this calculation environment")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_plotting):
            result = self.persist()
        self.assertEqual(set(result), {"numerical", "random_control"})
        for path in result["numerical"].values():
            loaded = load_evaluation_outputs(path)
            self.assertEqual(loaded["manifest"]["status"], "complete")
            self.assertNotIn("presentation_id", loaded["config"])
        random = load_random_control_outputs(result["random_control"])
        self.assertEqual(random["manifest"]["status"], "complete")
        self.assertFalse((self.context["project_root"] / "outputs/contextual-evaluation-report").exists())

    def test_existing_random_target_fails_before_either_numerical_write(self):
        target = self.context["project_root"] / "outputs/contextual-random-control" / self.fixture.bundle.name
        target.mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.persist()
        self.assertFalse((self.context["project_root"] / "outputs/contextual-anchor-q").exists())
        self.assertFalse((self.context["project_root"] / "outputs/contextual-dimsim").exists())

    def test_real_input_nesting_is_still_rejected(self):
        self.context["project_root"] = self.fixture.bundle / "bad-output"
        with self.assertRaises(ValueError):
            self.persist()
        self.assertFalse(self.context["project_root"].exists())

    def test_random_summary_never_loads_vector_matrices(self):
        result = self.persist()
        with patch("numpy.load", side_effect=AssertionError("No vector loading in visualization")):
            summary = load_random_control_summary(result["random_control"])
        self.assertNotIn("matrices", summary)
        self.assertNotIn("per_seed", summary["report"])
        self.assertEqual(summary["tables"]["anchor_q"], self.random["report"]["anchor_q"])

    def report(self, result, run_id):
        from phonetics.visualization.evaluation_report import write_evaluation_report
        return write_evaluation_report(
            self.context["project_root"], run_id,
            anchor_q_output=result["numerical"]["contextual-anchor-q"],
            dimsim_output=result["numerical"]["contextual-dimsim"],
            random_control_output=result["random_control"], source_hashes={"plot.py": "a" * 64},
            source_digest="b" * 64,
        )

    def test_missing_font_does_not_affect_saved_numerical_runs(self):
        result = self.persist()
        before = {path: sha(path / "manifest.json") for path in result["numerical"].values()}
        with patch("phonetics.visualization.evaluation_report.apply_style", side_effect=ValueError("Arial absent")):
            with self.assertRaisesRegex(ValueError, "Arial absent"):
                self.report(result, "font-check")
        self.assertFalse((self.context["project_root"] / "outputs/contextual-evaluation-report/font-check").exists())
        self.assertEqual(before, {path: sha(path / "manifest.json") for path in before})

    def test_report_separately_gates_all_five_metadata_files(self):
        from phonetics.visualization.evaluation import FIGURE_STEMS
        result = self.persist()

        def renderer(q, d, output, digest, **kwargs):
            for stem in FIGURE_STEMS:
                for suffix in ("png", "eps", "metadata.json"):
                    (output / f"{stem}.{suffix}").write_text("synthetic render fixture", encoding="utf-8")

        with patch("phonetics.visualization.evaluation_report.apply_style"), patch(
            "phonetics.visualization.evaluation_report.render_evaluation", side_effect=renderer
        ):
            root = self.report(result, "separate-report")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        expected = {row["path"] for row in manifest["artifacts"]}
        self.assertEqual(sum(path.endswith(".metadata.json") for path in expected), 5)
        _verify_inventory(root, "contextual-evaluation-report", root.name, expected,
                          profile="contextual_evaluation_report", figure_exports="complete")
        self.assertEqual(manifest["inputs"][0]["path"], str(result["numerical"]["contextual-anchor-q"]))
        self.assertTrue((root / "data/random_control_summary.json").is_file())
        with self.assertRaises(FileExistsError):
            self.report(result, "separate-report")

    def test_failed_report_keeps_numerical_runs_complete(self):
        result = self.persist()
        with patch("phonetics.visualization.evaluation_report.apply_style"), patch(
            "phonetics.visualization.evaluation_report.render_evaluation", side_effect=RuntimeError("renderer failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "renderer failed"):
                self.report(result, "failed-report")
        root = self.context["project_root"] / "outputs/contextual-evaluation-report/failed-report"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        for path in result["numerical"].values():
            self.assertEqual(load_evaluation_outputs(path)["manifest"]["status"], "complete")


class ImmutableRunGuardTests(unittest.TestCase):
    def test_finished_presentation_is_not_a_parent_for_new_outputs(self):
        from phonetics.visualization.evaluation import _guard_not_in_immutable_run
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            for status in ("complete", "failed"):
                (root / "manifest.json").write_text(json.dumps({
                    "analysis": "contextual-evaluation-report", "status": status,
                    "scope": "contextual_evaluation_report",
                }), encoding="utf-8")
                for guard in (runtime._reject_immutable_ancestors, _guard_not_in_immutable_run):
                    with self.assertRaisesRegex(ValueError, "immutable"):
                        guard(root / "outputs" / "new-run")
            self.assertFalse((root / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
