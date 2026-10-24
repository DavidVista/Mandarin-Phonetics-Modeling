import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise unittest.SkipTest(f"scientific evaluation dependencies unavailable: {error}")

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
PREPARE = PROJECT / "notebooks" / "Datasets" / "Evaluation Setup" / "prepare-evaluation.ipynb"
COMPOSITE = PROJECT / "notebooks" / "Evaluation" / "geometry-analysis-composite.ipynb"
STANDALONE = PROJECT / "notebooks" / "Evaluation" / "geometry-analysis.ipynb"
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
    def test_checkpoint_paths_match_training_outputs_and_fail_before_execution(self):
        for path, prefix in ((COMPOSITE, "composite"), (STANDALONE, "standalone")):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            config = code(next(cell for cell in notebook["cells"] if cell["id"] == prefix + "-config"))
            namespace = {"find_project_root": lambda: PROJECT}
            exec(compile(config, "analysis-config", "exec"), namespace)
            runs = namespace["RUNS"]
            self.assertEqual(runs[0]["checkpoint_path"], namespace["RUNS_ROOT"] / "ce_only/seed-42/best.pt")
            self.assertEqual(runs[1]["checkpoint_path"], namespace["RUNS_ROOT"] / "combined/lambda-1/seed-42/best.pt")
            self.assertNotIn('"models"', config)
            run_source = code(next(cell for cell in notebook["cells"] if cell["id"] == prefix + "-run"))
            namespace["run_evaluation"] = mock.Mock()
            with mock.patch.object(Path, "is_file", return_value=False):
                with self.assertRaisesRegex(FileNotFoundError, "Training checkpoints"):
                    exec(compile(run_source, "analysis-run", "exec"), namespace)
            namespace["run_evaluation"].assert_not_called()

    def test_standalone_cells_execute_without_project_runtime_imports_and_match_composite(self):
        notebook, standalone = execute_embedded(STANDALONE)
        rendered = STANDALONE.read_text(encoding="utf-8")
        self.assertNotIn("from phonetics", rendered)
        self.assertNotIn("exec(", rendered)
        self.assertNotIn("pip install", rendered)
        self.assertNotIn("urlopen", rendered)

        composite_notebook = json.loads(COMPOSITE.read_text(encoding="utf-8"))
        composite = {"__name__": "composite_evaluation_test"}
        exec(compile(code(next(cell for cell in composite_notebook["cells"] if cell["id"] == "composite-build")), "composite-build", "exec"), composite)
        setup = code(next(cell for cell in composite_notebook["cells"] if cell["id"] == "composite-imports"))
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

    def test_dependency_setup_paths_and_storage_contract(self):
        prepare = json.loads(PREPARE.read_text(encoding="utf-8"))
        dependency_source = code(next(cell for cell in prepare["cells"] if cell["id"] == "prep-dependencies"))
        for enabled in (False, True):
            with self.subTest(install_dependencies=enabled):
                tree = ast.parse(dependency_source)
                assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name) and target.id == "INSTALL_DEPENDENCIES"
                                       for target in node.targets)]
                self.assertEqual(len(assignments), 1)
                assignments[0].value = ast.Constant(value=enabled)
                namespace = {}
                with mock.patch("subprocess.check_call") as install:
                    exec(compile(ast.fix_missing_locations(tree), "prep-dependencies", "exec"), namespace)
                if enabled:
                    install.assert_called_once_with(
                        [sys.executable, "-m", "pip", "install", *namespace["DEPENDENCIES"]])
                else:
                    install.assert_not_called()
        self.assertIn('"dimsim==0.2.2"', dependency_source)
        self.assertIn('"pypinyin==0.55.0"', dependency_source)
        self.assertNotIn('"torch', dependency_source)

        config_source = code(next(cell for cell in prepare["cells"] if cell["id"] == "prep-config"))
        namespace = {}; exec(compile(config_source, "prep-config", "exec"), namespace)
        self.assertEqual(namespace["EXPERIMENT_PATH"], PROJECT / "datasets" / "experiments" / "1.0.0")
        self.assertEqual(namespace["OUTPUT_ROOT"], PROJECT / "datasets" / "evaluations")
        self.assertEqual(namespace["EVALUATION_ID"], "1.0.0")
        custom = config_source.replace("CUSTOM_EXPERIMENT_PATH = None", 'CUSTOM_EXPERIMENT_PATH = Path("/custom/experiment")').replace(
            "CUSTOM_OUTPUT_ROOT = None", 'CUSTOM_OUTPUT_ROOT = Path("/custom/output")')
        custom_namespace = {}; exec(compile(custom, "prep-custom", "exec"), custom_namespace)
        self.assertEqual(custom_namespace["EXPERIMENT_PATH"], Path("/custom/experiment"))
        self.assertEqual(custom_namespace["OUTPUT_ROOT"], Path("/custom/output"))

        requirements = (PROJECT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("dimsim==0.2.2", requirements); self.assertIn("pypinyin==0.55.0", requirements)
        self.assertFalse((PROJECT / "requirements-evaluation.txt").exists())
        active = [line for line in requirements.splitlines() if line and not line.lstrip().startswith("#")]
        self.assertFalse(any(line.startswith("torch") for line in active))
        ignore = (PROJECT / ".gitignore").read_text(encoding="utf-8")
        for rule in ("datasets/evaluations/**", "!datasets/evaluations/**/manifest.json", "!datasets/evaluations/**/config.json",
                     "!datasets/evaluations/**/coverage.json", "datasets/evaluation-runs/**"):
            self.assertIn(rule, ignore)

    def test_project_root_discovery_from_workspace_and_descendant(self):
        composite = json.loads(COMPOSITE.read_text(encoding="utf-8"))
        setup = code(next(cell for cell in composite["cells"] if cell["id"] == "composite-imports"))
        function_source = setup[setup.index("def find_project_root"):setup.index("CODE_ROOT =")]
        namespace = {"Path": Path}; exec(compile(function_source, "root-discovery", "exec"), namespace)
        self.assertEqual(namespace["find_project_root"](WORKSPACE), PROJECT)
        self.assertEqual(namespace["find_project_root"](PROJECT / "notebooks" / "Evaluation"), PROJECT)

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
