import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise unittest.SkipTest(f"scientific evaluation dependencies unavailable: {error}")

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
PREPARE = PROJECT / "notebooks" / "Datasets" / "Evaluation Setup" / "prepare-evaluation.ipynb"
COMPOSITE = PROJECT / "notebooks" / "Evaluation" / "Geometry" / "geometry-analysis-composite.ipynb"
STANDALONE = PROJECT / "notebooks" / "Evaluation" / "Geometry" / "geometry-analysis.ipynb"
GENERATOR = WORKSPACE / "meta" / "tools" / "build_evaluation_notebooks.py"


def code(cell):
    return "".join(cell["source"])


def execute_embedded(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "portable_evaluation_test"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and (cell["id"].endswith("build") or "-module-" in cell["id"]):
            exec(compile(code(cell), f"{path}:{cell['id']}", "exec"), namespace)
    return notebook, namespace


class NotebookTests(unittest.TestCase):
    def test_standalone_cells_execute_without_project_runtime_imports_and_match_composite(self):
        notebook, standalone = execute_embedded(STANDALONE)
        rendered = STANDALONE.read_text(encoding="utf-8")
        self.assertNotIn("from phonetics", rendered)
        self.assertNotIn("exec(", rendered)
        self.assertNotIn("pip install", rendered)
        self.assertNotIn("urlopen", rendered)

        composite_notebook = json.loads(COMPOSITE.read_text(encoding="utf-8"))
        composite = {"__name__": "composite_evaluation_test"}
        exec(compile(code(composite_notebook["cells"][1]), "composite-build", "exec"), composite)
        setup = code(composite_notebook["cells"][2]).replace('Path("/EDIT/project")', f"Path({str(PROJECT)!r})")
        for name in list(sys.modules):
            if name == "phonetics" or name.startswith("phonetics."):
                del sys.modules[name]
        exec(compile(setup, "composite-imports", "exec"), composite)

        ids = ["a", "b", "c"]
        portable_matrix = standalone["random_occurrence_baseline"](ids, dimension=8, seed=17)
        composite_module = sys.modules["phonetics.evaluation.representations"]
        canonical_matrix = composite_module.random_occurrence_baseline(ids, dimension=8, seed=17)
        np.testing.assert_array_equal(portable_matrix, canonical_matrix)
        pairs = [{"pair_id": "p", "left_id": "a", "right_id": "b", "dimsim_distance": 1.0,
                  "distance_status": "ok", "overlap_mask": "000", "no_zero_initial_diagnostic": True}]
        portable_rows = standalone["cosine_rows"](pairs, ids, portable_matrix)
        canonical_analysis = sys.modules["phonetics.evaluation.analysis"]
        self.assertEqual(portable_rows, canonical_analysis.cosine_rows(pairs, ids, canonical_matrix))
        self.assertTrue(all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code"))

    def test_preparation_embedded_cells_execute_and_use_explicit_metric(self):
        notebook, namespace = execute_embedded(PREPARE)
        self.assertIn("DimsimDistance", namespace)
        self.assertIn("prepare_evaluation_bundle", namespace)
        run_source = code(next(cell for cell in notebook["cells"] if cell["id"] == "prep-run"))
        self.assertIn("metric = DimsimDistance()", run_source)
        self.assertIn("metric_provenance=metric.provenance", run_source)
        self.assertIn('code_provenance=BUILD_METADATA["source_sha256"]', run_source)

    def test_builder_check_detects_modified_temporary_output(self):
        if not GENERATOR.is_file():
            self.skipTest("workspace notebook generator is unavailable in this project snapshot")
        spec = importlib.util.spec_from_file_location("evaluation_notebook_builder_test", GENERATOR)
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            builder.PREPARE_PATH = root / "prepare.ipynb"
            builder.COMPOSITE_PATH = root / "composite.ipynb"
            builder.STANDALONE_PATH = root / "standalone.ipynb"
            builder.build_notebooks()
            builder.build_notebooks(check=True)
            builder.STANDALONE_PATH.write_text(builder.STANDALONE_PATH.read_text(encoding="utf-8") + "edited", encoding="utf-8")
            with self.assertRaises(SystemExit):
                builder.build_notebooks(check=True)


if __name__ == "__main__":
    unittest.main()
