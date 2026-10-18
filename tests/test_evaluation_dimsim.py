import importlib
import importlib.metadata
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / "src" / "phonetics"
phonetics_package = types.ModuleType("phonetics"); phonetics_package.__path__ = [str(SOURCE)]
evaluation_package = types.ModuleType("phonetics.evaluation"); evaluation_package.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics_package); sys.modules.setdefault("phonetics.evaluation", evaluation_package)
module = importlib.import_module("phonetics.evaluation.dimsim")


class DimsimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.metric = module.DimsimDistance()
        except importlib.metadata.PackageNotFoundError as error:
            raise unittest.SkipTest(f"pinned DIMSIM dependencies unavailable: {error}")

    def test_canonical_examples_symmetry_and_identity(self):
        examples = {
            ("shi4", "shi4"): 0.0,
            ("ji1", "qi1"): 0.47619047619047616,
            ("ma1", "ma4"): 0.0014285714285714284,
            ("mai4", "ai4"): 67324.81696874699,
            ("yi1", "shi4"): 68003.92173809969,
            ("qu4", "ju4"): 19.07266437184804,
        }
        for pair, expected in examples.items():
            with self.subTest(pair=pair):
                self.assertAlmostEqual(self.metric(*pair), expected)
                self.assertAlmostEqual(self.metric(pair[1], pair[0]), expected)
        self.assertEqual(self.metric("ma1", "ma1"), 0.0)
        self.assertTrue(self.metric.provenance["package_file_sha256"])

    def test_version_mismatch_rejected_before_import(self):
        with mock.patch.object(module.metadata, "version", side_effect=lambda name: "9.9.9"):
            with self.assertRaisesRegex(RuntimeError, "do not match pinned"):
                module.DimsimDistance()


if __name__ == "__main__":
    unittest.main()
