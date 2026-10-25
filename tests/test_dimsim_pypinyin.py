import importlib
import importlib.metadata
import itertools
import math
import sys
import types
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "phonetics"
phonetics_package = types.ModuleType("phonetics"); phonetics_package.__path__ = [str(SOURCE)]
evaluation_package = types.ModuleType("phonetics.evaluation"); evaluation_package.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics_package); sys.modules.setdefault("phonetics.evaluation", evaluation_package)
pinyin_module = importlib.import_module("phonetics.pinyin")
public_module = importlib.import_module("phonetics.evaluation.dimsim")
module = importlib.import_module("phonetics.evaluation.dimsim_pypinyin")


class PypinyinDimsimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.public = public_module.DimsimDistance()
            cls.metric = module.PypinyinDimsimDistance()
            from dimsim.utils.pinyin import Pinyin
            cls.AuthorPinyin = Pinyin
        except importlib.metadata.PackageNotFoundError as error:
            raise unittest.SkipTest(f"pinned DIMSIM dependencies unavailable: {error}")

    def test_reproduction_scorer_matches_public_a_in_both_directions(self):
        labels = [
            "ba1", "ba4", "pa2", "ma3", "fa4", "da1", "ta2", "na3", "la4",
            "ga1", "ka2", "ha3", "ji1", "qi2", "xi3", "ju4", "qu1", "gu2",
            "xu3", "lv4", "nv3", "you3", "wei2", "wen4", "hui3", "fei3",
            "hua1", "fa1", "a1",
        ]
        for left, right in itertools.product(labels, repeat=2):
            with self.subTest(left=left, right=right):
                reproduced = module.score_original_components(
                    self.AuthorPinyin(left), self.AuthorPinyin(right)
                )
                self.assertEqual(reproduced, self.public(left, right))
                self.assertTrue(math.isfinite(reproduced))

    def test_strict_mapping_examples_are_exact(self):
        expected = {
            "ju1": ("j", "v", "j", "u:", "ju:"),
            "qu1": ("q", "v", "q", "u:", "qu:"),
            "gu1": ("g", "u", "g", "u", "gu"),
            "xu1": ("x", "v", "x", "u:", "xu:"),
            "yuan1": ("", "van", "", "u:an", "u:an"),
            "yun1": ("", "vn", "", "u:n", "u:n"),
            "you1": ("", "iou", "", "iou", "iou"),
            "wei1": ("", "uei", "", "uei", "uei"),
            "wen1": ("", "uen", "", "uen", "uen"),
            "hui1": ("h", "uei", "h", "uei", "huei"),
            "a1": ("", "a", "", "a", "a"),
        }
        for token, values in expected.items():
            with self.subTest(token=token):
                row = module.component_mapping(token)
                self.assertEqual(
                    (row["strict_initial"], row["strict_final"], row["dimsim_consonant"],
                     row["dimsim_vowel"], row["dimsim_name"]), values,
                )

    def test_complete_strict_inventory_is_supported(self):
        seen_initials, seen_finals = set(), set()
        for base in pinyin_module._INPUT_SYLLABLES:
            try:
                raw = pinyin_module.parse_components(base + "1")
            except ValueError:  # Dictionary interjections such as hm have no final.
                continue
            adapted = module.adapted_components(base + "1")
            seen_initials.add(raw.initial); seen_finals.add(raw.final)
            self.assertIn(adapted.consonant, module.consonantMap_TwoDCode)
            self.assertIn(adapted.vowel, module.vowelMap_TwoDCode)
        self.assertEqual(seen_initials, set(pinyin_module.INITIALS))
        self.assertEqual(seen_finals, set(pinyin_module.FINALS))
        self.assertEqual((len(seen_initials), len(seen_finals)), (22, 36))

    def test_identity_policy_and_hardcode_behavior_are_not_repaired(self):
        first = module.adapted_components("ba1")
        second = module.adapted_components("bo1")
        self.assertIs(first.consonant, second.consonant)
        # Strict hui -> huei changes the lookup key; author objects also fail the
        # hardcode identity comparison, so neither route silently fixes it.
        self.assertEqual(module.component_mapping("hui3")["dimsim_name"], "huei")
        self.assertEqual(self.public("hui3", "fei3"), 1.9047619047619047)
        self.assertEqual(self.public("hua1", "fa1"), 1.9047619047619047)

    def test_b_changes_only_decomposition_before_original_scoring(self):
        self.assertEqual(self.metric("ju1", "qu1"), 1.0 / 2.1)
        self.assertNotEqual(self.metric("ju1", "qu1"), self.public("ju1", "qu1"))
        for pair in (("ba1", "pa1"), ("ma1", "ma4"), ("hua1", "fa1")):
            with self.subTest(pair=pair):
                self.assertEqual(self.metric(*pair), self.public(*pair))

    def test_calls_do_not_mutate_author_maps(self):
        from dimsim.utils import maps
        snapshots = {
            name: dict(getattr(maps, name))
            for name in ("consonantMap_TwoDCode", "vowelMap_TwoDCode", "hardcodeMap")
        }
        for pair in (("ju1", "qu1"), ("yuan2", "yun4"), ("hui3", "fei3")):
            self.metric(*pair)
        for name, snapshot in snapshots.items():
            self.assertEqual(getattr(maps, name), snapshot)

    def test_errors_and_provenance_are_explicit(self):
        for invalid in ("", "MA1", "notpinyin1", "ma0", "ma6", "m1", "n1", "ng1", "hm1", "hng1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.metric(invalid, "ma1")
        with self.assertRaisesRegex(ValueError, "Pinyin components"):
            module.score_original_components(object(), object())
        provenance = self.metric.provenance
        self.assertFalse(provenance["canonical_weights_modified"])
        self.assertTrue(provenance["adapter_file_sha256"])
        self.assertIn("utils/utils.py", provenance["package_file_sha256"])
        self.assertIn("no contraction", provenance["spelling_policy"])
        self.assertEqual(provenance["pinyin_contract_version"], pinyin_module.PINYIN_CONTRACT_VERSION)
        self.assertEqual(provenance["parser_provenance"], pinyin_module.PARSER_PROVENANCE)


if __name__ == "__main__":
    unittest.main()
