"""Portable source closure, guards and build freshness without ML execution."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pipeline_builder", PROJECT.parent / "meta/tools/build_pipeline_notebooks.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class PipelineNotebookTests(unittest.TestCase):
    def test_current_notebooks_are_fresh_valid_and_parse(self):
        import nbformat
        paths = BUILDER.build(check=True)
        self.assertEqual(len(paths), 9)
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                source = cell.source
                if source.startswith("%%writefile"):
                    source = source.split("\n", 1)[1]
                ast.parse(source)
            config = next(c.source for c in notebook.cells if c.id == "configuration")
            self.assertIn("RUN_TRAINING = False" if "Training" in path.parts else "RUN_PREPARATION = False", config)

    def test_snapshot_is_exact_and_importable_without_checkout_or_dependencies(self):
        notebook = BUILDER.notebook("Fixture", "prepare-data")
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as temporary:
            root = Path(temporary)
            count = 0
            for cell in notebook["cells"]:
                source = "".join(cell["source"])
                if not source.startswith("%%writefile"):
                    continue
                first, body = source.split("\n", 1)
                relative = first.split("/phonetics/", 1)[1].rstrip('"')
                path = root / "phonetics" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")
                expected = notebook["metadata"]["source_sha256"][relative]
                self.assertEqual(hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest(), expected)
                count += 1
            self.assertEqual(count, len(BUILDER.MODULES))
            # -S removes site packages, including any installed project/ML SDK.
            code = "import sys; from phonetics.cli import main; main(['--help'])"
            result = subprocess.run([sys.executable, "-S", "-c", code], cwd=root,
                                    capture_output=True, text=True, check=True)
            self.assertIn("prepare-experiment", result.stdout)

    def test_scoped_composites_removed(self):
        self.assertFalse(list((PROJECT / "notebooks/Training").rglob("*-composite.ipynb")))
        self.assertFalse((PROJECT / "notebooks/Datasets/Data Preparation/prepare-pinyin-embeddings-composite.ipynb").exists())


if __name__ == "__main__":
    unittest.main()
