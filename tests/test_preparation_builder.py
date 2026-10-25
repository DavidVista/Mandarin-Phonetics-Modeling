import importlib.util
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

    def test_standalone_build_embeds_executable_loader_definition(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/"training.ipynb"
            builder.build(path,"test","mse_only","1.0","standalone")
            notebook=json.loads(path.read_text(encoding="utf-8"))
            generated=next(cell for cell in notebook["cells"] if cell["cell_type"]=="code" and "BUILD_METADATA" in "".join(cell["source"]))
            namespace={}; exec("".join(generated["source"]),namespace)
            self.assertIn("load_experiment",namespace)


if __name__ == "__main__": unittest.main()
