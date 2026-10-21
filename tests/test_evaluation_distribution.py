import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "src" / "phonetics"
phonetics_package = types.ModuleType("phonetics"); phonetics_package.__path__ = [str(SOURCE)]
evaluation_package = types.ModuleType("phonetics.evaluation"); evaluation_package.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics_package); sys.modules.setdefault("phonetics.evaluation", evaluation_package)
module = importlib.import_module("phonetics.evaluation.distribution")


class DistributionTests(unittest.TestCase):
    def test_bin_boundaries_and_half_shift(self):
        self.assertEqual(module.bin_label(0), "[0,10)")
        self.assertEqual(module.bin_label(9.999), "[0,10)")
        self.assertEqual(module.bin_label(10), "[10,20)")
        self.assertEqual(module.bin_label(120), "[120,+inf)")
        self.assertEqual(module.bin_label(7, width=10, shift=5, overflow=None), "[5,15)")
        self.assertEqual(module.bin_label(2, width=10, shift=5, overflow=None), "[0,5)")

    def test_formula_matches_canonical_api(self):
        try:
            metric = module.DimsimDistance()
        except Exception as error:
            self.skipTest(f"pinned DIMSIM unavailable: {error}")
        for left, right in (("ji1", "qi1"), ("ma1", "ma4"), ("qu4", "ju4"), ("mai4", "ai4")):
            with self.subTest(pair=(left, right)):
                self.assertAlmostEqual(module.reconstruct_distance(left, right)["distance"], metric(left, right))

    def test_cache_rejects_code_or_bundle_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle.json"; bundle.write_text("bundle", encoding="utf-8")
            code = root / "code.py"; code.write_text("code", encoding="utf-8")
            (root / "dimsim.py").write_text("adapter", encoding="utf-8")
            columns = {"distance": np.asarray([1.0]), "left_pinyin": np.asarray(["ma1"]), "right_pinyin": np.asarray(["ma2"])}
            summary = {"validation": {"manifest_sha256": module.sha256_file(bundle),
                                      "metric_provenance": {"package_file_sha256": {"x.py": "0" * 64}}}}
            manifest = module.save_cache(root / "cache", columns, summary, {}, code_path=code)
            loaded, _, _ = module.load_cache(root / "cache", code_path=code, bundle_manifest=bundle)
            self.assertEqual(loaded["distance"].tolist(), [1.0])
            code.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "code hash mismatch"):
                module.load_cache(root / "cache", code_path=code, bundle_manifest=bundle)
            self.assertTrue(json.loads(manifest.read_text())["artifacts"])


if __name__ == "__main__":
    unittest.main()
