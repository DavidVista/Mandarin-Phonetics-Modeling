"""The calculation composite contains no presentation or plotting stage."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT.parent / "meta/tools/build_evaluation_notebooks.py"
COMPOSITE = PROJECT / "notebooks/Evaluation/contextual-evaluation-composite.ipynb"


def source(cell):
    return "".join(cell["source"])


class EvaluationCompositeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("data_composite_builder", BUILDER)
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)
        cls.notebook = json.loads(COMPOSITE.read_text(encoding="utf-8"))
        cls.cells = {cell["id"]: cell for cell in cls.notebook["cells"]}

    def test_composite_is_current_valid_and_data_only(self):
        import nbformat
        nbformat.validate(nbformat.from_dict(self.notebook))
        self.builder.build_notebooks(only_composite=True, check=True)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            code = source(cell)
            compile(code, cell["id"], "exec")
            for forbidden in ("matplotlib", "IPython.display", "display(", "render_evaluation",
                              "PRESENTATION_ID", "EXPORT_FORMATS", "EXPORT_DPI"):
                self.assertNotIn(forbidden, code)
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
        build = self.notebook["metadata"]["evaluation_build"]
        self.assertEqual(build["scope"], "numerical_evaluation")
        self.assertNotIn("resource_sha256", build)
        self.assertFalse(any(path.startswith("visualization/") or path.endswith("evaluation_tables.py")
                             for path in build["source_sha256"]))
        self.assertEqual(self.notebook["cells"][-1]["id"], "analysis-interpretation")

    def test_composite_only_generation_preserves_other_notebooks_and_data_settings(self):
        sentinel = '''from pathlib import Path
BUNDLE_PATH = Path("/custom/evaluation")
DEVICE = "cuda"
BATCH_SIZE = 7
PRESENTATION_ID = "old-report"
EXPORT_FORMATS, EXPORT_DPI = ["png", "eps"], 300
DEPENDENCIES = ["numpy==2.1.3", "matplotlib>=3.8", "pypinyin==0.55.0"]
CONFIG = {"bundle_path": BUNDLE_PATH, "device": DEVICE, "batch_size": BATCH_SIZE,
          "presentation_id": PRESENTATION_ID, "export_formats": EXPORT_FORMATS, "export_dpi": EXPORT_DPI}
'''
        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            root = Path(directory)
            composite, standalone, prepare = (root / name for name in ("composite.ipynb", "standalone.ipynb", "prepare.ipynb"))
            composite.write_text(json.dumps({"cells": [{"id": "analysis-parameters", "source": sentinel.splitlines(keepends=True)}]}), encoding="utf-8")
            standalone.write_bytes(b"preserve standalone exactly")
            prepare.write_bytes(b"preserve preparation exactly")
            with patch.multiple(self.builder, COMPOSITE_PATH=composite, STANDALONE_PATH=standalone, PREPARE_PATH=prepare):
                self.assertEqual(self.builder.build_notebooks(only_composite=True), [composite])
                self.builder.build_notebooks(only_composite=True, check=True)
            notebook = json.loads(composite.read_text(encoding="utf-8"))
            parameters = source(next(cell for cell in notebook["cells"] if cell["id"] == "analysis-parameters"))
            ns = {}
            exec(parameters, ns)
            self.assertEqual(ns["CONFIG"], {"bundle_path": Path("/custom/evaluation"), "device": "cuda", "batch_size": 7})
            self.assertEqual(ns["DEPENDENCIES"], ["numpy==2.1.3", "pypinyin==0.55.0"])
            self.assertNotIn("PRESENTATION_ID", ns)
            self.assertEqual(standalone.read_bytes(), b"preserve standalone exactly")
            self.assertEqual(prepare.read_bytes(), b"preserve preparation exactly")

    def test_generation_options_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.builder.build_notebooks(only_composite=True, only_geometry=True)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.builder.build_notebooks(only_composite=True, only_prepare=True)

    def test_runtime_import_in_fresh_process_needs_no_plotting_or_torch(self):
        program = '''import builtins,json,sys
from pathlib import Path
nb=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ns={"CANONICAL_SOURCE_ROOT":Path(sys.argv[2])}
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.startswith(("matplotlib","phonetics.visualization","torch")):
        raise ImportError("Unavailable optional plotting/ML dependency: "+name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
for cell in nb["cells"]:
    if cell["id"] in {"analysis-build-metadata","analysis-verify-canonical","analysis-import-runtime"}:
        exec(compile("".join(cell["source"]),cell["id"],"exec"),ns)
assert callable(ns["persist_results"])
assert not any(name.startswith(("matplotlib","phonetics.visualization","torch")) for name in sys.modules)
print("Runtime import: PASS")
'''
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in sys.path if path)
        result = subprocess.run([sys.executable, "-c", program, str(COMPOSITE), str(PROJECT / "src/phonetics")],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_pipeline_stops_after_numerical_persistence(self):
        calls = []
        context, representations, random = {}, {}, {}
        q, d = {"results": [None] * 25}, {"results": [None] * 25}
        results = {"numerical": {"q": "q-path", "d": "d-path"}, "random_control": "random-path"}
        def callback(name, value):
            def run(*args):
                calls.append(name)
                return value
            return Mock(side_effect=run)
        ns = {
            "CONFIG": {}, "BUILD_METADATA": {}, "EVALUATION_STAGE": "scores",
            "preflight_evaluation": callback("preflight", context),
            "extract_or_reuse": callback("representations", representations),
            "score_anchor_q": callback("q", q), "score_dimsim": callback("dimsim", d),
            "score_random_control": callback("random", random),
            "persist_results": callback("persist", results),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            for cell in self.notebook["cells"]:
                if cell["cell_type"] == "code" and cell["id"] in {
                    "analysis-preflight", "analysis-representations", "analysis-q",
                    "analysis-dimsim", "analysis-random", "analysis-output",
                }:
                    exec(compile(source(cell), cell["id"], "exec"), ns)
        self.assertEqual(calls, ["preflight", "representations", "q", "dimsim", "random", "persist"])
        ns["persist_results"].assert_called_once_with(context, q, d, random)
        self.assertEqual(ns["RESULT_PATHS"], results)

    def test_dependency_cell_never_installs_or_requires_graphics(self):
        import importlib.metadata
        code = source(self.cells["analysis-dependencies"])
        versions = {"numpy": "2.1.3", "dimsim": "0.2.2", "pypinyin": "0.55.0", "torch": "synthetic"}
        for mode in ("extract", "reuse"):
            with patch.object(importlib.metadata, "version", side_effect=versions.__getitem__) as version, \
                    patch("subprocess.check_call") as install, contextlib.redirect_stdout(io.StringIO()):
                exec(code, {"INSTALL_DEPENDENCIES": False, "EXTRACTION_MODE": mode, "EVALUATION_STAGE": "scores"})
            install.assert_not_called()
            self.assertNotIn("matplotlib", [call.args[0] for call in version.call_args_list])
            self.assertEqual("torch" in [call.args[0] for call in version.call_args_list], mode == "extract")


if __name__ == "__main__":
    unittest.main()
