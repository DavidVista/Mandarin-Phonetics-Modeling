"""Population, inventory and provenance fixtures for the new data path."""
import csv
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from phonetics.artifacts import sha256_file, write_jsonl
from phonetics.bundles import seal_bundle
from phonetics.corpus import prepare_character_corpus
from phonetics.acoustic import assemble_character_data, prepare_acoustic_targets, validate_character_data
from phonetics.sound_inventory import inspect_audio
from phonetics.pinyin import PARSER_PROVENANCE, PINYIN_CONTRACT_VERSION

MODEL = {"id": "fixture", "revision": "a" * 40, "layer": -1, "pooling": "mean_all",
         "sample_rate": 16000, "mono": True, "padding": "none", "dtype": "float32", "device": "cpu"}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory = self.root / "inventory"
        self.inventory.mkdir()
        self.rows = []
        for label in ("ma1", "ni3"):
            p = self.inventory / f"{label}.wav"
            with wave.open(str(p), "wb") as h:
                h.setnchannels(1)
                h.setsampwidth(2)
                h.setframerate(16000)
                h.writeframes(b"\0\0" * 160)
            self.rows.append({"label": label, "relative_path": p.name,
                              "sha256": sha256_file(p), **inspect_audio(p)})
        write_jsonl(self.inventory / "inventory.jsonl", self.rows)
        seal_bundle(self.inventory, "sound_inventory", {"fixture": True})

    def corpus(self):
        p = self.root / "source.csv"
        with p.open("w", encoding="utf-8", newline="") as h:
            w = csv.writer(h)
            w.writerow(["text", "pinyin"])
            w.writerows([("你", "ni3"), ("你", "ni3"), ("妈", "ma1"),
                         ("吗", "ma1"), ("吗", "ma5"), ("好", "hao3"), ("x", "ma1")])
        target = self.root / "corpus"
        prepare_character_corpus(p, self.inventory, target, expected_sha256=sha256_file(p))
        return target

    def test_conflicts_precede_coverage_and_duplicates_preserve_sources(self):
        corpus = self.corpus()
        rows = [json.loads(line) for line in (corpus / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({r["text"] for r in rows}, {"你", "妈"})
        self.assertEqual(len(next(r for r in rows if r["text"] == "你")["source_ids"]), 2)
        stats = json.loads((corpus / "statistics.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["input"], stats["selected"] + stats["duplicates"] + stats["rejected"])
        self.assertEqual(stats["reasons"], {"characters": 1, "conflicting_pronunciation": 2, "sound_coverage": 1})

    def test_prepared_preserves_population_and_reuse_checks_corruption(self):
        corpus = self.corpus()
        def backend(row, config):
            return [1.0, 2.0] if row["label"] == "ma1" else [2.0, 1.0]
        acoustic = prepare_acoustic_targets(self.inventory, self.root / "acoustic", model=MODEL,
                                           cache_root=self.root / "cache", backend=backend)
        prepared = assemble_character_data(corpus, acoustic, self.root / "prepared")
        self.assertEqual((prepared / "corpus.jsonl").read_bytes(), (corpus / "corpus.jsonl").read_bytes())
        self.assertEqual(assemble_character_data(corpus, acoustic, prepared), prepared)
        (prepared / "targets.npy").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            assemble_character_data(corpus, acoustic, prepared)

    def test_incomplete_inference_does_not_publish(self):
        def backend(row, config):
            if row["label"] == "ni3":
                raise RuntimeError("fixture inference failure")
            return [1.0, 2.0]
        with self.assertRaisesRegex(RuntimeError, "fixture inference"):
            prepare_acoustic_targets(self.inventory, self.root / "broken", model=MODEL,
                                     cache_root=self.root / "cache", backend=backend)
        self.assertFalse((self.root / "broken").exists())

    def test_corpus_import_does_not_import_ids_or_torch(self):
        source = str(Path(__file__).parents[1] / "src")
        code = ("import sys; sys.path.insert(0, " + repr(source) + "); import phonetics.corpus; "
                "assert 'phonetics.ids' not in sys.modules; assert 'torch' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_resealed_prepared_cannot_change_upstream_content(self):
        corpus = self.corpus()
        acoustic = prepare_acoustic_targets(self.inventory, self.root / "acoustic", model=MODEL,
                                           cache_root=self.root / "cache", backend=lambda r, c: [1., 2.])
        for name in ("corpus.jsonl", "targets.npy"):
            with self.subTest(name=name):
                destination = self.root / ("prepared-" + name)
                assemble_character_data(corpus, acoustic, destination)
                manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
                (destination / name).write_bytes(b"")
                seal_bundle(destination, "character_prepared", manifest["identity"])
                with self.assertRaisesRegex(ValueError, "upstream artifact"):
                    validate_character_data(destination)

    def test_legacy_import_authenticates_pickle_before_loading(self):
        legacy = self.root / "legacy"
        (legacy / "reports").mkdir(parents=True)
        write_jsonl(legacy / "sound_inventory.jsonl", self.rows)
        (legacy / "pinyin_embeddings.pkl").write_bytes(pickle.dumps({"ma1": [1., 2.], "ni3": [2., 1.]}))
        inference = {"backend": "transformers", "computed": 2, "cached": 0, "errors": 0}
        (legacy / "reports/inference.json").write_text(json.dumps(inference), encoding="utf-8")
        (legacy / "config.json").write_text(json.dumps({"bundle_id": "fixture", "model": MODEL}), encoding="utf-8")
        manifest = {"bundle_type": "acoustic_embeddings", "schema_version": 1,
                    "bundle_id": "fixture", "model": {**MODEL, "backend": "transformers"},
                    "inference": inference, "pinyin_contract": PINYIN_CONTRACT_VERSION,
                    "parser_provenance": PARSER_PROVENANCE, "count": 2, "dimension": 2,
                    "inventory_sha256": sha256_file(legacy / "sound_inventory.jsonl"),
                    "artifacts": {n: sha256_file(legacy / n) for n in
                                  ("sound_inventory.jsonl", "config.json", "reports/inference.json")}}
        def attempt():
            p = legacy / "manifest.json"
            p.write_text(json.dumps(manifest), encoding="utf-8")
            return prepare_acoustic_targets(self.inventory, self.root / "imported", model=MODEL,
                                            existing_bundle=legacy, existing_manifest_sha256=sha256_file(p))
        with patch("pickle.load", side_effect=AssertionError("unauthenticated pickle loaded")):
            with self.assertRaisesRegex(ValueError, "required authenticated"):
                attempt()
        manifest["artifacts"]["pinyin_embeddings.pkl"] = sha256_file(legacy / "pinyin_embeddings.pkl")
        (legacy / "pinyin_embeddings.pkl").write_bytes(b"changed pickle")
        with patch("pickle.load", side_effect=AssertionError("unauthenticated pickle loaded")):
            with self.assertRaisesRegex(ValueError, "checksum"):
                attempt()
        (legacy / "pinyin_embeddings.pkl").write_bytes(pickle.dumps({"ma1": [1., 2.], "ni3": [2., 1.]}))
        manifest["artifacts"]["pinyin_embeddings.pkl"] = sha256_file(legacy / "pinyin_embeddings.pkl")
        self.assertEqual(attempt(), self.root / "imported")


if __name__ == "__main__":
    unittest.main()
