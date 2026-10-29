import importlib.util
import ast
import builtins
import symtable
import sys
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).parents[2]
spec=importlib.util.spec_from_file_location("builder",ROOT/"meta/tools/build_training_notebooks.py")
builder=importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)


class BuilderTests(unittest.TestCase):
    def test_portable_pinyin_cells_execute_in_fresh_namespace(self):
        spec=importlib.util.spec_from_file_location("preparation_builder",ROOT/"meta/tools/build_preparation_notebook.py")
        preparation_builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(preparation_builder)
        saved=json.loads((ROOT/"project/notebooks/Datasets/Data Preparation/prepare-pinyin-embeddings.ipynb").read_text(encoding="utf-8"))
        for notebook in (preparation_builder.portable_notebook(),saved):
            cells={cell.get("id"):cell for cell in notebook["cells"]};namespace={}
            for cell_id in ("dependencies","pinyin-contract"):
                exec("".join(cells[cell_id]["source"]),namespace)
            self.assertEqual(namespace["canonical_label"]("ma1"),"ma1")
            self.assertEqual(namespace["canonical_label"]("ma5",tone_convention="explicit"),"ma5")
            parsed=namespace["parse_components"]("qu2")
            self.assertEqual((parsed.initial,parsed.final,parsed.tone),("q","v","2"))
            self.assertEqual(namespace["parse_components"]("lu:").tone, "5")
            self.assertEqual(namespace["parse_components"]("weng1").final, "ueng")
            for invalid in ("lü4", "Zhong1", "mā", "shong1", "m"):
                with self.assertRaises(ValueError): namespace["canonical_label"](invalid)

    def test_pinyin_definitions_are_standalone_and_selected_from_canonical_module(self):
        source=(builder.SOURCE/"pinyin.py").read_text(encoding="utf-8-sig")
        namespace={}; exec(source,namespace)
        self.assertEqual(namespace["canonical_labels"](["ma"]),["ma5"])
        self.assertEqual(namespace["parse_components"]("yue4").final,"ve")

    def test_current_training_notebooks_embed_dependency_closure_and_guard_runs(self):
        notebook=builder.notebook("components-acoustic",composite=False)
        cells={cell["id"]:"".join(cell["source"]) for cell in notebook["cells"]}
        self.assertIn("experiment.py", " ".join(builder.MODULES))
        self.assertIn("acoustic_targets.npy", cells["experiment-loading"])
        self.assertIn("def train_experiment", cells["checkpoints-summaries"])
        self.assertNotIn("embedded-sources", cells)
        self.assertIn("RUN_REAL_TRAINING = False", cells["settings"])
        self.assertIn("LAMBDA_VALUE = 1", cells["settings"])
        self.assertNotIn("LAMBDA_VALUE not in", cells["settings"])
        self.assertNotIn("exec(", "\n".join(cells.values()))
        composite=builder.notebook("syllable",composite=True)
        composite_cells={cell["id"]:"".join(cell["source"]) for cell in composite["cells"]}
        self.assertIn("CODE_ROOT", composite_cells["runtime"])
        self.assertIn("RUN_REAL_TRAINING = False", composite_cells["settings"])


    def test_visible_definitions_match_canonical_sources(self):
        notebook = builder.notebook("syllable", composite=False)
        definitions = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code": continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    self.assertNotIn(node.name, definitions)
                    definitions[node.name] = node
        for relative in builder.MODULES:
            tree = ast.parse((builder.SOURCE / relative).read_text(encoding="utf-8-sig"))
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) or node.name not in definitions: continue
                expected = node
                if relative == "training.py" and node.name == "train_experiment":
                    expected = ast.parse(builder._adapt_trainer(ast.get_source_segment((builder.SOURCE / relative).read_text(encoding="utf-8-sig"), node))).body[0]
                self.assertEqual(ast.dump(definitions[node.name]), ast.dump(expected), node.name)

    def test_visible_cells_resolve_globals_without_sdk_or_exec(self):
        notebook = builder.notebook("syllable", composite=False)
        source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
        table = symtable.symtable(ast.unparse(tree), "notebook", "exec")
        available = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
        available.update(dir(builtins)); available.add("__name__")
        def unresolved(scope):
            names = {s.get_name() for s in scope.get_symbols() if s.is_global() and s.is_referenced()} - available
            for child in scope.get_children(): names.update(unresolved(child))
            return names
        self.assertEqual(unresolved(table), set())
        self.assertNotIn("__file__", source)
        self.assertNotIn("from .", source)

    def test_visible_bundle_loader_runs_without_torch(self):
        from project.tests.test_experiment_bundle import ExperimentBundleTests
        fixture = ExperimentBundleTests("test_writer_loader_preserve_data_and_portability")
        fixture.setUp()
        try:
            notebook = builder.notebook("syllable", composite=False)
            cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}
            namespace = {}
            imports = ast.parse(cells["dependencies"])
            imports.body = [n for n in imports.body if not (
                isinstance(n, ast.Import) and any(a.name == "torch" or a.name.startswith("torch.") for a in n.names)
                or isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("torch"))]
            exec(compile(imports, "notebook-imports-without-torch", "exec"), namespace)
            for cell_id in ("artifact-helpers", "pinyin-contract", "ids-validation", "experiment-loading"):
                exec(compile(cells[cell_id], cell_id, "exec"), namespace)
            loaded = namespace["load_experiment"](fixture.bundle)
            self.assertEqual(len(loaded["samples"]), 12)
            self.assertEqual(loaded["config"]["training_seeds"], [0, 1, 2, 3, 4])
        finally:
            fixture.tearDown()


if __name__ == "__main__": unittest.main()
