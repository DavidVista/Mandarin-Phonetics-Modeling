import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from phonetics.artifacts import sha256_file, write_json
from phonetics.experiment import (_split_ids, default_experiment_config, load_experiment,
                                 prepare_experiment, validate_experiment)
from phonetics.preparation import prepare_acoustic_bundle, prepare_dataset, validate_prepared


class StubBackend:
    cache_id = "experiment-fixture-v1"

    def __call__(self, entry, config):
        index = {"ma1": 1., "ma3": 2., "ba4": 3.}[entry["label"]]
        return np.asarray([index, 1.], dtype=np.float32)


class ExperimentBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        self.source = self.root / "prepared"
        chars = list("甲乙丙丁戊己庚辛壬癸子丑")
        self.dictionary = self.root / "dictionary.jsonl"
        self.dictionary.write_text("".join(json.dumps({"character": c, "decomposition": c}, ensure_ascii=False) + "\n" for c in chars), encoding="utf-8")
        for name, content in (("COPYING", "fixture COPYING"), ("LGPL", "fixture LGPL")):
            (self.root / name).write_text(content, encoding="utf-8")
        self.ids = {"path": str(self.dictionary), "sha256": sha256_file(self.dictionary),
                    "snapshot_commit": "618dbab8a8ddefb958763c8b4afbaa741a4460de",
                    "license": "LGPL-3.0-or-later", "attribution": "fixture source attribution"}
        sounds = self.root / "sounds"
        sounds.mkdir()
        for label in ("ma1", "ma3", "ba4"):
            with wave.open(str(sounds / f"{label}.wav"), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000); f.writeframes(b"\0\0" * 24)
        sound_csv = self.root / "sounds.csv"
        sound_csv.write_text("path,pinyin\nma1.wav,ma1\nma3.wav,ma3\nba4.wav,ba4\n", encoding="utf-8")
        model = {"id":"fixture", "revision":"r1", "layer":-1, "pooling":"mean_all", "sample_rate":8000,
                 "mono":True, "padding":"none", "dtype":"float32", "device":"cpu"}
        acoustic = prepare_acoustic_bundle({"bundle_id":"audio", "output_root":str(self.root / "audio-bundles"),
            "cache_root":str(self.root / "cache"), "sounds":{"path":str(sounds), "manifest":str(sound_csv),
            "sha256":sha256_file(sound_csv), "tone_convention":"explicit"}, "model":model}, embedding_backend=StubBackend())
        corpus = self.root / "corpus.csv"
        with corpus.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n"); writer.writerow(["text", "pinyin", "source_id"])
            for i, char in enumerate(chars): writer.writerow([char, ("ma1", "ma3", "ba4")[i % 3], f"source-{i}"])
        prepared_cfg = {"dataset_id":"fixture", "output_root":str(self.root / "prepared-out"),
            "corpus":{"path":str(corpus), "sha256":sha256_file(corpus), "tone_convention":"explicit"},
            "ids":self.ids, "sampling":{"size":None, "strategy":"uniform", "seed":5},
            "acoustic_bundle":{"path":str(acoustic), "manifest_sha256":sha256_file(acoustic / "manifest.json")}}
        self.prepared = prepare_dataset(prepared_cfg)
        validate_prepared(self.prepared)
        pin = sha256_file(self.prepared / "manifest.json")
        cfg = default_experiment_config(2); cfg["prepared_manifest_sha256"] = pin
        self.bundle = prepare_experiment("fixture-exp", prepared_root=self.prepared, output_root=self.root / "experiments", config=cfg)
        self.pin = pin

    def tearDown(self):
        self.tmp.cleanup()

    def _copy_bundle(self, name):
        dest = self.root / name
        shutil.copytree(self.bundle, dest)
        return dest

    def _recheck_artifact(self, root, relative):
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][relative] = sha256_file(root / relative)
        if relative.startswith("provenance/"):
            manifest["provenance"][relative] = sha256_file(root / relative)
        write_json(manifest_path, manifest)

    def _rewrite_members(self, root, rows):
        (root / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        splits = _split_ids([r["sample_id"] for r in rows], config["split_seed"], config["split_ratios"])
        write_json(root / "splits.json", splits)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest["artifacts"]["samples.jsonl"] = sha256_file(root / "samples.jsonl")
        manifest["artifacts"]["splits.json"] = sha256_file(root / "splits.json")
        manifest["counts"]["samples"] = len(rows)
        manifest["counts"]["positions"] = sum(r["sentence_length"] for r in rows)
        write_json(root / "manifest.json", manifest)

    def test_writer_loader_preserve_data_and_portability(self):
        loaded = load_experiment(self.bundle)
        self.assertEqual(len(loaded["samples"]), 12)
        self.assertEqual(loaded["config"]["training_seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(loaded["config"]["training"]["weight_decay"], 0.0)
        self.assertEqual(loaded["vocab"]["syllables"], sorted(["ma1", "ma3", "ba4"]))
        self.assertEqual(loaded["acoustic_targets"].shape, (3, 2))
        self.assertFalse((self.bundle / "initialization").exists())
        self.assertEqual(set(loaded["samples"]), set(sum(loaded["splits"].values(), [])))
        prepared_rows = {r["sample_id"]: r for r in (json.loads(x) for x in (self.prepared / "corpus.jsonl").read_text(encoding="utf-8").splitlines())}
        for sid, row in loaded["samples"].items():
            for key in ("component_lengths", "ids_sequences", "labels", "positions", "sample_id", "sentence_length", "source_ids", "text"):
                self.assertEqual(row[key], prepared_rows[sid][key])
            self.assertEqual(row["component_labels"], row["syllable_ids"] and [list(__import__("phonetics.pinyin", fromlist=["decompose"]).decompose(label)) for label in row["labels"]])
            self.assertEqual(row["acoustic_ids"], row["syllable_ids"])
        portable = self._copy_bundle("portable")
        shutil.rmtree(self.prepared)
        self.assertEqual(validate_experiment(portable)["counts"]["samples"], 12)

    def test_config_input_pin_validation_and_overwrite(self):
        with self.assertRaises(FileExistsError):
            prepare_experiment("fixture-exp", prepared_root=self.prepared, output_root=self.root / "experiments",
                               config=default_experiment_config(2) | {"prepared_manifest_sha256": self.pin})
        for mutate in (
            lambda c: c.update(split_seed=True),
            lambda c: c.update(split_ratios=[.8, float("nan"), .1]),
            lambda c: c.update(training_seeds=[0, 1, 2, 3, 3]),
            lambda c: c.update(prepared_manifest_sha256="0" * 64),
            lambda c: c["training"].update(weight_decay=.001),
        ):
            cfg = default_experiment_config(2); cfg["prepared_manifest_sha256"] = self.pin; mutate(cfg)
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                prepare_experiment("bad", prepared_root=self.prepared, output_root=self.root / "bad-out", config=cfg)

        changed = self._copy_bundle("bad-config")
        saved = json.loads((changed / "config.json").read_text(encoding="utf-8"))
        saved["training"]["weight_decay"] = .001
        write_json(changed / "config.json", saved); self._recheck_artifact(changed, "config.json")
        with self.assertRaisesRegex(ValueError, "conflicts with the training protocol"):
            validate_experiment(changed)

    def test_rehashed_semantic_tampering_and_missing_required_file(self):
        broken = self._copy_bundle("bad-ids")
        mapping_path = broken / "ids/mapping.json"
        mapping = json.loads(mapping_path.read_text(encoding="utf-8")); mapping["characters"].pop(next(iter(mapping["characters"])))
        write_json(mapping_path, mapping); self._recheck_artifact(broken, "ids/mapping.json")
        with self.assertRaises(ValueError): validate_experiment(broken)

        broken = self._copy_bundle("bad-target")
        targets = np.load(broken / "acoustic_targets.npy"); targets[0] = 0
        np.save(broken / "acoustic_targets.npy", targets); self._recheck_artifact(broken, "acoustic_targets.npy")
        with self.assertRaisesRegex(ValueError, "acoustic target matrix"): validate_experiment(broken)

        broken = self._copy_bundle("missing-target")
        rows_path = broken / "samples.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        from phonetics.data_contract import sample_id
        rows[0]["labels"][0] = "ma2"
        rows[0]["sample_id"] = sample_id(rows[0]["text"], rows[0]["labels"])
        rows_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        self._recheck_artifact(broken, "samples.jsonl")
        with self.assertRaisesRegex(ValueError, "missing acoustic target"):
            validate_experiment(broken)

        broken = self._copy_bundle("bad-split")
        splits = json.loads((broken / "splits.json").read_text(encoding="utf-8")); splits["test"].pop()
        write_json(broken / "splits.json", splits); self._recheck_artifact(broken, "splits.json")
        with self.assertRaises(ValueError): validate_experiment(broken)

        broken = self._copy_bundle("missing-required")
        (broken / "ids/LGPL").unlink()
        with self.assertRaises(ValueError): validate_experiment(broken)

    def test_pinned_population_and_source_ids_survive_internal_rehashing(self):
        rows = [json.loads(line) for line in (self.bundle / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
        dropped = self._copy_bundle("dropped-row")
        self._rewrite_members(dropped, rows[1:])
        with self.assertRaisesRegex(ValueError, "pinned prepared corpus"):
            validate_experiment(dropped)

        invented = self._copy_bundle("invented-source")
        altered = json.loads(json.dumps(rows))
        altered[0]["source_ids"] = ["invented-source"]
        self._rewrite_members(invented, altered)
        with self.assertRaisesRegex(ValueError, "pinned prepared corpus"):
            validate_experiment(invented)

    def test_pinned_acoustics_and_seed_protocol_survive_internal_rehashing(self):
        altered = self._copy_bundle("changed-acoustic-value")
        matrix = np.load(altered / "acoustic_targets.npy")
        matrix[0, 0] += np.float32(.25)
        np.save(altered / "acoustic_targets.npy", matrix)
        self._recheck_artifact(altered, "acoustic_targets.npy")
        with self.assertRaisesRegex(ValueError, "reproduce pinned prepared targets"):
            validate_experiment(altered)

        for name, key, value in (("alternate-training-seeds", "training_seeds", [5, 6, 7, 8, 9]),
                                 ("alternate-split-seed", "split_seed", 5)):
            changed = self._copy_bundle(name)
            config = json.loads((changed / "config.json").read_text(encoding="utf-8"))
            config[key] = value
            write_json(changed / "config.json", config)
            self._recheck_artifact(changed, "config.json")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "fixed protocol"):
                validate_experiment(changed)

        changed_provenance = self._copy_bundle("rehashed-pinned-source")
        corpus_copy = changed_provenance / "provenance/prepared_corpus.jsonl"
        corpus_copy.write_text(corpus_copy.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self._recheck_artifact(changed_provenance, "provenance/prepared_corpus.jsonl")
        with self.assertRaisesRegex(ValueError, "copied upstream provenance"):
            validate_experiment(changed_provenance)

    def test_public_dataset_api_imports_without_torch(self):
        env = dict(os.environ); env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
        command = "import sys; from phonetics import prepare_experiment, load_experiment, validate_experiment; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", command], check=True, env=env, capture_output=True, text=True)

    def test_deterministic_splits_and_safe_ids(self):
        ids = [f"sample-{i:02}" for i in range(20)]
        self.assertEqual(_split_ids(ids, 0, [.8, .1, .1]), _split_ids(list(reversed(ids)), 0, [.8, .1, .1]))
        with self.assertRaises(ValueError): prepare_experiment("../escape", prepared_root=self.prepared, output_root=self.root / "experiments")
        with self.assertRaises(ValueError): prepare_experiment(".", prepared_root=self.prepared, output_root=self.root / "experiments")
        config = default_experiment_config(2); config["prepared_manifest_sha256"] = self.pin
        output = prepare_experiment("default-destination", prepared_root=self.prepared, config=config)
        self.assertEqual(output, self.root / "experiments" / "default-destination")


if __name__ == "__main__":
    unittest.main()
