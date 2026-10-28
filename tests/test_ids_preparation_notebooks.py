"""Check that preparation candidates require explicit execution settings."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


class IDSPreparationNotebookTests(unittest.TestCase):
    def test_disabled_candidates_do_not_read_or_write_datasets(self):
        project = Path(__file__).resolve().parents[1]
        for name in ("prepare-corpus", "prepare-data"):
            with self.subTest(notebook=name):
                notebook = json.loads((project / "notebooks/Datasets/Data Preparation" /
                                       f"{name}.ipynb").read_text(encoding="utf-8"))
                blocked = Mock(side_effect=AssertionError("Unexpected dataset operation"))
                package = types.ModuleType("phonetics")
                package.prepare_corpus = blocked
                package.assemble_dataset = blocked
                artifacts = types.ModuleType("phonetics.artifacts")
                artifacts.sha256_file = blocked
                source = types.ModuleType("phonetics.source_preparation")
                source.export_duyu = blocked
                modules = {"phonetics": package, "phonetics.artifacts": artifacts,
                           "phonetics.source_preparation": source}
                namespace = {}
                previous = Path.cwd()
                original_path = list(sys.path)
                try:
                    os.chdir(project)
                    with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
                        for index, cell in enumerate(notebook["cells"]):
                            if cell["cell_type"] == "code":
                                code = "".join(cell["source"])
                                exec(compile(code, f"{name}:cell-{index}", "exec"), namespace)
                        blocked.assert_not_called()
                        if name == "prepare-corpus":
                            self.assertFalse(namespace["RUN_EXPORT"])
                            self.assertFalse(namespace["RUN_PROCESSING"])
                            self.assertEqual(namespace["IDS_PIN"]["sha256"],
                                             "744bb05d5b0742e9ee35c37791f94d56a173349b3367569e7ca11e510364d203")
                            namespace["RUN_PROCESSING"] = True
                        else:
                            self.assertFalse(namespace["RUN_PREPARATION"])
                            namespace["RUN_PREPARATION"] = True
                        with self.assertRaisesRegex(ValueError, "Select the output ID"):
                            exec("".join(notebook["cells"][-1]["source"]), namespace)
                        blocked.assert_not_called()
                finally:
                    os.chdir(previous)
                    sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
