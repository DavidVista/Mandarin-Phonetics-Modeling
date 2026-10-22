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
    def test_pinyin_definitions_are_standalone_and_selected_from_canonical_module(self):
        source=builder.source_definitions(builder.SOURCE/"pinyin.py",{"INITIALS","FINALS","TONES","ZERO_INITIAL","_canonical_labels","_decompose"})
        namespace={}; exec(source,namespace)
        self.assertEqual(namespace["_canonical_labels"](["ma"]),["ma5"])
        self.assertEqual(namespace["_decompose"]("yue4"),(0,32,3))

    def test_standalone_build_embeds_executable_loader_definition(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path=Path(directory)/"training.ipynb"
            builder.build(path,"test","mse_only","1.0","standalone")
            notebook=json.loads(path.read_text(encoding="utf-8"))
            generated=next(cell for cell in notebook["cells"] if cell["cell_type"]=="code" and "BUILD_METADATA" in "".join(cell["source"]))
            namespace={}; exec("".join(generated["source"]),namespace)
            self.assertIn("load_experiment",namespace)


if __name__ == "__main__": unittest.main()
