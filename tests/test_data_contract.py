"""Characterize historical pinyin and the shared prepared contract."""
import importlib
import sys
import types
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "phonetics"
package = types.ModuleType("phonetics")
package.__path__ = [str(SOURCE)]
sys.modules.setdefault("phonetics", package)
pinyin = importlib.import_module("phonetics.pinyin")
contract = importlib.import_module("phonetics.data_contract")


class DataContractTests(unittest.TestCase):
    def test_historical_components_and_no_spelling_guess(self):
        self.assertEqual(pinyin.canonical_labels(["de", "ma1"]), ["de5", "ma1"])
        for token, components in {"ai4": ("", "ai", "4"), "qu2": ("q", "v", "2"),
                                  "gu1": ("g", "u", "1"), "yuan2": ("", "van", "2")}.items():
            indexes = pinyin.decompose(token)
            self.assertEqual(tuple(values[index] for values, index in
                                   zip((pinyin.INITIALS, pinyin.FINALS, pinyin.TONES), indexes)), components)
        for invalid in ("MA1", "nü3", "ma0", "bad9"):
            with self.assertRaises(ValueError):
                pinyin.decompose(invalid)
        with self.assertRaises(ValueError):
            pinyin.canonical_labels(["de"], tone_convention="explicit")

    def test_filter_order_and_zero_initial(self):
        wubi = {"爱": "ep", "妈": "vg"}
        self.assertIsNone(contract.pair_reason("爱", ["ai4"], wubi, {"ai4": 1}))
        cases = [("", [], "nonempty"), ("A", ["ma1"], "characters"),
                 ("妈", ["ma1", "ma1"], "alignment"), ("妈", ["bad9"], "pinyin"),
                 ("爸", ["ba1"], "wubi"), ("妈", ["ma1"], "acoustic")]
        for text, labels, expected in cases:
            self.assertEqual(contract.pair_reason(text, labels, wubi, {"ai4": 1}), expected)

    def test_stable_id_matches_legacy_serialization(self):
        self.assertEqual(contract.sample_id("妈", ["ma1"]),
                         "64066df4ebfea74f4bdac84a0e4b7504436b848387fc92c916883f543539b960")

    def test_wubi_preserves_digits_and_rejects_malformed(self):
        self.assertEqual(contract.validate_wubi({"妈": "vg1"}), {"妈": "vg1"})
        for value in ({}, {"妈妈": "vg"}, {"妈": ""}, {"妈": "VG"}):
            with self.assertRaises(ValueError):
                contract.validate_wubi(value)

    def test_acoustic_vectors_and_inventory_are_strict(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable in this interpreter")
        self.assertEqual(contract.validate_acoustic({"ma1": [1, 2]}, expected_keys=["ma1"]), 2)
        for value in ({"ma1": [0, 0]}, {"ma1": [np.nan, 1]},
                      {"ma1": [1], "ma2": [1, 2]}, {"ma1": [[1, 2]]}):
            with self.assertRaises(ValueError):
                contract.validate_acoustic(value)
        with self.assertRaises(ValueError):
            contract.validate_acoustic({"ma1": [1]}, expected_keys=["ma2"])


if __name__ == "__main__":
    unittest.main()
