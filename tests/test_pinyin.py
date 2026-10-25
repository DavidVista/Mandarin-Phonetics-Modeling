"""Project TONE3 syntax, dictionary eligibility, and strict component contract."""
import importlib
import sys
import types
import unittest
from pathlib import Path
from pypinyin.constants import PINYIN_DICT
from pypinyin.contrib.tone_convert import to_normal, to_initials, to_finals

SOURCE = Path(__file__).resolve().parents[1] / "src" / "phonetics"
package = types.ModuleType("phonetics")
package.__path__ = [str(SOURCE)]
sys.modules.setdefault("phonetics", package)
pinyin = importlib.import_module("phonetics.pinyin")


class PinyinTests(unittest.TestCase):
    def parts(self, token):
        value = pinyin.parse_components(token)
        return value.initial, value.final, value.tone

    def test_allowed_normalization_and_idempotence(self):
        for raw, expected in {"zhong1":"zhong1", "ma":"ma5", "ma5":"ma5", "lv4":"lv4",
                              "lu:4":"lv4", "lu:":"lv5", "lu:e4":"lve4", "liu2":"liu2"}.items():
            with self.subTest(raw=raw):
                self.assertEqual(pinyin.canonical_label(raw), expected)
                self.assertEqual(pinyin.canonical_label(expected), expected)
                self.assertEqual(pinyin.decompose(raw), pinyin.decompose(expected))
        self.assertEqual(self.parts("lu:"), ("l", "v", "5"))
        with self.assertRaises(ValueError): pinyin.canonical_label("ma", tone_convention="explicit")

    def test_rejects_every_out_of_contract_format(self):
        for raw in ("mā", "zhōng", "zho1ng", "lü4", "Zhong1", " ma1", "ma1 ", "ma1 ma2",
                    "ma0", "ma55", "ma6", "mā1", "ê1", "", None, 3):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                pinyin.canonical_label(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                pinyin.parse_components(raw)

    def test_dictionary_before_components(self):
        for raw in ("shong1", "biong1", "abc1"):
            with self.assertRaisesRegex(ValueError, "absent from pypinyin dictionary"):
                pinyin.canonical_label(raw)
        for base in ("m", "n", "ng", "hm", "hng"):
            self.assertIn(base, pinyin.VALID_SYLLABLES)
            with self.assertRaisesRegex(ValueError, "non-empty final required"):
                pinyin.canonical_label(base)

    def test_strict_examples(self):
        expected={"ju1":("j","v","1"), "qun2":("q","vn","2"), "xuan3":("x","van","3"),
                  "gu1":("g","u","1"), "kuan3":("k","uan","3"), "hu4":("h","u","4"),
                  "you3":("","iou","3"), "wei4":("","uei","4"), "wen2":("","uen","2"),
                  "weng1":("","ueng","1"), "lu:e4":("l","ve","4"), "liu2":("l","iou","2")}
        for raw, parts in expected.items():
            with self.subTest(raw=raw): self.assertEqual(self.parts(raw), parts)

    def test_inventory_derived_from_dictionary_and_roundtrips(self):
        expected={to_normal(reading, v_to_u=False).lower() for readings in PINYIN_DICT.values() for reading in readings.split(',')}
        self.assertEqual(pinyin.VALID_SYLLABLES, expected)
        components={(to_initials(base, strict=True), to_finals(base, strict=True, v_to_u=False))
                    for base in expected if base.isascii() and base.isalpha() and base.islower()}
        self.assertEqual(pinyin.INITIALS, sorted({i for i,f in components if f}))
        self.assertEqual(pinyin.FINALS, sorted({f for _,f in components if f}))
        self.assertIn("ueng", pinyin.FINALS)
        for base in expected:
            if base.isascii() and base.isalpha() and to_finals(base, strict=True):
                label=base+'5'; parsed=pinyin.parse_components(label); ids=pinyin.decompose(label)
                self.assertEqual(tuple(values[index] for values,index in zip((pinyin.INITIALS,pinyin.FINALS,pinyin.TONES),ids)),
                                 (parsed.initial,parsed.final,parsed.tone))

    def test_stored_labels_are_not_repaired(self):
        for raw in ("ma", "lu:4", "ma0", "MA1"):
            with self.assertRaises(ValueError): pinyin.validate_canonical_label(raw)
        self.assertEqual(pinyin.validate_canonical_label("ma5"), "ma5")


if __name__ == "__main__": unittest.main()
