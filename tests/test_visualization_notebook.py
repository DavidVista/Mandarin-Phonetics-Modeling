"""Exercise the artifact-only visualization notebook, including real PNG/EPS export."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import nbformat
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
TARGET = PROJECT / "notebooks/Visualization/evaluation.ipynb"
GENERATOR = WORKSPACE / "meta/tools/build_visualization_notebook.py"
sys.path[:0] = [str(PROJECT / "tests"), str(PROJECT / "src")]


def source(cell):
    return "".join(cell["source"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class VisualizationNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(TARGET.read_text(encoding="utf-8"))
        cls.cells = {cell["id"]: cell for cell in cls.notebook["cells"]}
        cls.module = types.ModuleType("visualization_notebook_under_test")
        sys.modules[cls.module.__name__] = cls.module
        cls.ns = cls.module.__dict__
        cls.temp = tempfile.TemporaryDirectory(dir=PROJECT)
        exec(source(cls.cells["parameters"]), cls.ns)
        cls.ns["EVALUATION_OUTPUT_ROOT"] = Path(cls.temp.name)
        for cell in cls.notebook["cells"]:
            if cell["cell_type"] != "code" or cell["id"] in {"parameters", "input-paths", "save-report"}:
                continue
            exec(compile(source(cell), cell["id"], "exec"), cls.ns)
        spec = importlib.util.spec_from_file_location("visualization_notebook_builder", GENERATOR)
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    @classmethod
    def tearDownClass(cls):
        for name in ("_style_resources", "_plot_cache"):
            cls.ns[name].cleanup()
        cls.temp.cleanup()
        sys.modules.pop(cls.module.__name__, None)

    def test_current_schema_syntax_and_no_sdk_or_display(self):
        nbformat.validate(nbformat.from_dict(self.notebook))
        self.builder.build_notebook(check=True)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(source(cell))
            compile(tree, cell["id"], "exec")
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    self.assertFalse((node.module or "").startswith(("phonetics", "torch", "dimsim", "pypinyin")))
                    self.assertNotEqual(node.module, "IPython.display")
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"display", "exec", "_publish_module"})
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotEqual(node.func.attr, "show")

    def test_builder_preserves_parameters_and_other_notebooks(self):
        siblings = [PROJECT / "notebooks/Evaluation" / name for name in
                    ("contextual-evaluation-composite.ipynb", "contextual-evaluation.ipynb")]
        before = {path: sha(path) for path in siblings}
        sentinel = 'from pathlib import Path\nEVALUATION_OUTPUT_ROOT = Path("/custom/artifacts")\nPRESENTATION_ID = "keep-this-id"\n'
        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            target = Path(directory) / "visualization.ipynb"
            target.write_text(json.dumps({"cells": [{"id": "parameters", "source": sentinel.splitlines(keepends=True)}]}), encoding="utf-8")
            with patch.object(self.builder, "TARGET", target):
                self.builder.build_notebook()
                self.builder.build_notebook(check=True)
            built = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(source(next(c for c in built["cells"] if c["id"] == "parameters")), sentinel)
        self.assertEqual(before, {path: sha(path) for path in siblings})

    def test_unique_run_selection_is_not_latest_selection(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            root = Path(directory)
            for analysis in ("contextual-anchor-q", "contextual-dimsim", "contextual-random-control"):
                path = root / analysis / "run-01"
                path.mkdir(parents=True)
                (path / "manifest.json").write_text(json.dumps({
                    "status": "complete", "analysis": analysis, "scope": "numerical_outputs", "run_id": path.name,
                }), encoding="utf-8")
            selected = self.ns["resolve_evaluation_runs"](root)
            self.assertEqual({path.name for path in selected.values()}, {"run-01"})
            extra = root / "contextual-anchor-q/run-02"
            extra.mkdir()
            (extra / "manifest.json").write_text(json.dumps({
                "status": "complete", "analysis": "contextual-anchor-q", "scope": "numerical_outputs", "run_id": extra.name,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "set EVALUATION_RUN_IDS explicitly"):
                self.ns["resolve_evaluation_runs"](root)
            explicit = self.ns["resolve_evaluation_runs"](root, {name: "run-01" for name in selected})
            self.assertEqual(explicit, selected)

    def test_missing_font_fails_before_report_creation(self):
        from matplotlib import font_manager
        with patch.object(font_manager.fontManager, "ttflist", []):
            with self.assertRaisesRegex(RuntimeError, "Set FONT_FILES"):
                exec(source(self.cells["fonts"]), dict(self.ns, FONT_FILES=[]))

    def test_actual_report_from_saved_outputs_without_vector_loading(self):
        import test_evaluation_data_persistence as fixtures
        suite = fixtures.DataPersistenceTests
        suite.setUpClass()
        try:
            case = suite("test_data_stage_needs_no_plotting_settings_or_imports")
            case.setUp()
            saved = case.persist()
            inputs = [*saved["numerical"].values(), saved["random_control"]]
            before = {path: sha(path) for root in inputs for path in root.rglob("*") if path.is_file()}
            ns = self.ns
            ns["EVALUATION_OUTPUT_ROOT"] = case.context["project_root"] / "outputs"
            ns["EVALUATION_RUN_IDS"] = None
            ns["OUTPUT_PROJECT_ROOT"] = suite.fixture.root / "visualization-output"
            ns["PRESENTATION_ID"] = "actual-export"
            with patch("numpy.load", side_effect=AssertionError("Plotting must not load vector matrices")):
                exec(source(self.cells["input-paths"]), ns)
                exec(source(self.cells["save-report"]), ns)
            report = ns["REPORT_PATH"]
            for suffix in ("png", "eps", "metadata.json"):
                files = list((report / "figures").glob(f"*.{suffix}"))
                self.assertEqual(len(files), 5)
                self.assertTrue(all(path.stat().st_size > 0 for path in files))
            manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["figure_exports"], "complete")
            for row in manifest["artifacts"]:
                self.assertEqual(sha(report / row["path"]), row["sha256"])
            self.assertEqual(before, {path: sha(path) for path in before})
            self.assertEqual(ns["plt"].get_fignums(), [])
            with self.assertRaises(FileExistsError):
                exec(source(self.cells["save-report"]), ns)
            altered = suite.fixture.root / "changed/contextual-anchor-q" / saved["numerical"]["contextual-anchor-q"].name
            import shutil
            shutil.copytree(saved["numerical"]["contextual-anchor-q"], altered)
            with (altered / "tables/tables.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
                ns["write_evaluation_report"](
                    ns["OUTPUT_PROJECT_ROOT"], "bad-input", anchor_q_output=altered,
                    dimsim_output=saved["numerical"]["contextual-dimsim"],
                    random_control_output=saved["random_control"], source_hashes=ns["SOURCE_HASHES"],
                    source_digest=ns["SOURCE_DIGEST"], protocol=ns["PROTOCOL"],
                )
        finally:
            suite.tearDownClass()


if __name__ == "__main__":
    unittest.main()
