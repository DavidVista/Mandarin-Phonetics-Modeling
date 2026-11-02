"""Current contextual notebooks build without retired notebook templates."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
PREPARE = PROJECT / "notebooks/Datasets/Evaluation Setup/prepare-evaluation.ipynb"
COMPOSITE = PROJECT / "notebooks/Evaluation/contextual-evaluation-composite.ipynb"
STANDALONE = PROJECT / "notebooks/Evaluation/contextual-evaluation.ipynb"
GENERATOR = WORKSPACE / "meta/tools/build_evaluation_notebooks.py"


def source(cell):
    return "".join(cell["source"])


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("evaluation_notebook_builder", GENERATOR)
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def test_contextual_notebooks_are_current(self):
        self.builder.build_notebooks(only_geometry=True, check=True)

    def test_schema_and_no_retired_importers_or_plotting(self):
        import nbformat
        for path in (STANDALONE, COMPOSITE):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            nbformat.validate(nbformat.from_dict(notebook))
            self.assertEqual(notebook['cells'][-1]['id'], 'analysis-output')
            self.assertEqual(notebook['metadata']['evaluation_build']['scope'], 'numerical_evaluation')
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                code = source(cell)
                compile(code, cell['id'], 'exec')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
                for token in ('%%', '_publish_module', 'MetaPathFinder', 'render_evaluation', 'import matplotlib'):
                    self.assertNotIn(token, code)
            for relative, digest in notebook['metadata']['evaluation_build']['source_sha256'].items():
                self.assertEqual(hashlib.sha256((PROJECT/'src/phonetics'/relative).read_bytes()).hexdigest(), digest)

    def test_build_without_existing_notebooks_preserves_prepare(self):
        before = PREPARE.read_bytes()
        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            root = Path(directory)
            with patch.multiple(self.builder, COMPOSITE_PATH=root/'composite.ipynb', STANDALONE_PATH=root/'standalone.ipynb'):
                self.builder.build_notebooks(only_geometry=True)
                self.builder.build_notebooks(only_geometry=True, check=True)
                for path in (self.builder.COMPOSITE_PATH, self.builder.STANDALONE_PATH):
                    self.assertTrue(path.is_file())
                self.assertEqual(sorted(p.name for p in root.iterdir()), ['composite.ipynb', 'standalone.ipynb'])
        self.assertEqual(before, PREPARE.read_bytes())

    def test_build_preserves_each_notebooks_own_parameters(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            root = Path(directory)
            with patch.multiple(self.builder, COMPOSITE_PATH=root/'composite.ipynb', STANDALONE_PATH=root/'standalone.ipynb'):
                for path, identifier, value in ((self.builder.COMPOSITE_PATH, 'analysis-parameters', 'composite'), (self.builder.STANDALONE_PATH, 'parameters', 'standalone')):
                    path.write_text(json.dumps({'cells': [{'id': identifier, 'source': [f'USER_SENTINEL = "{value}"\n']}]}), encoding='utf-8')
                self.builder.build_notebooks(only_geometry=True)
                self.builder.build_notebooks(only_geometry=True, check=True)
                for path, value in ((self.builder.COMPOSITE_PATH, 'composite'), (self.builder.STANDALONE_PATH, 'standalone')):
                    notebook = json.loads(path.read_text(encoding='utf-8'))
                    cell = next(c for c in notebook['cells'] if c['id'] in {'parameters', 'analysis-parameters'})
                    ns = {}
                    exec(source(cell), ns)
                    self.assertEqual(ns['USER_SENTINEL'], value)

    def test_standalone_definitions_without_torch_or_repository_imports(self):
        notebook = json.loads(STANDALONE.read_text(encoding='utf-8'))
        module = types.ModuleType('standalone_definitions_test')
        sys.modules[module.__name__] = module
        try:
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code' or cell['id'].startswith('analysis-'):
                    continue
                if cell['id'] in {'data', 'model', 'reproducibility', 'evaluation-encoders'}:
                    continue
                exec(compile(source(cell), cell['id'], 'exec'), module.__dict__)
            self.assertEqual(module.stable_cosine([1e-300,0], [1e-300,0]), 1.0)
            self.assertTrue(callable(module.persist_results))
        finally:
            sys.modules.pop(module.__name__, None)

    def test_preparation_notebook_contract_is_preserved(self):
            notebook = json.loads(PREPARE.read_text(encoding="utf-8"))
            cells = {cell.get("id"): source(cell) for cell in notebook["cells"]}
            self.assertIn("metric = DimsimDistance()", cells["prep-run"])
            self.assertIn('code_provenance=BUILD_METADATA["source_sha256"]', cells["prep-run"])
            self.assertIn('"dimsim==0.2.2"', cells["prep-dependencies"])
            self.assertIn('"pypinyin==0.55.0"', cells["prep-dependencies"])
            self.assertNotIn("torch", cells["prep-dependencies"].lower())


if __name__ == "__main__":
    unittest.main()
