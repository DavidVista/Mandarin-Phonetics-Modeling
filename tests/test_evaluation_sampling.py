import hashlib
import importlib
import json
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "phonetics"
# The existing phonetics package initializer imports the scientific training
# stack. Load this dependency-free subpackage directly for stdlib-only checks.
phonetics_package = types.ModuleType("phonetics")
phonetics_package.__path__ = [str(SOURCE)]
evaluation_package = types.ModuleType("phonetics.evaluation")
evaluation_package.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics_package)
sys.modules.setdefault("phonetics.evaluation", evaluation_package)

bundle_module = importlib.import_module("phonetics.evaluation.bundle")
sampling_module = importlib.import_module("phonetics.evaluation.sampling")
load_evaluation_bundle = bundle_module.load_evaluation_bundle
prepare_evaluation_bundle = bundle_module.prepare_evaluation_bundle
pair_count = sampling_module.pair_count
pair_from_rank = sampling_module.pair_from_rank
sample_pair_indices = sampling_module.sample_pair_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PairSamplingTests(unittest.TestCase):
    def test_rank_boundaries_and_complete_identity(self):
        self.assertEqual(pair_count(4), 6)
        pairs = [pair_from_rank(4, rank) for rank in range(6)]
        self.assertEqual(pairs, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        with self.assertRaises(IndexError):
            pair_from_rank(4, 6)

    def test_sampling_is_deterministic_unique_and_approximately_uniform(self):
        self.assertEqual(sample_pair_indices(10, 12, 7), sample_pair_indices(10, 12, 7))
        self.assertEqual(len(set(sample_pair_indices(10, 12, 7))), 12)
        counts = Counter(pair for seed in range(900) for pair in sample_pair_indices(4, 1, seed))
        self.assertEqual(set(counts), set(pair_from_rank(4, rank) for rank in range(6)))
        self.assertLess(max(counts.values()) - min(counts.values()), 55)


class EvaluationBundleTests(unittest.TestCase):
    def make_experiment(self, root):
        experiment = root / "experiment"
        experiment.mkdir()
        vocab = {"acoustic_tokens": ["ma1", "ma2", "ba1"], "labels": {"initials": ["m", "b"], "finals": ["a"], "tones": ["1", "2"]}}
        samples = [
            {"sample_id": "a", "text": "妈妈", "acoustic_ids": [0, 1], "component_labels": [[0, 0, 0], [0, 0, 1]]},
            {"sample_id": "b", "text": "爸", "acoustic_ids": [2], "component_labels": [[1, 0, 0]]},
        ]
        (experiment / "samples.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in samples), encoding="utf-8")
        (experiment / "splits.json").write_text(json.dumps({"train": [], "validation": [], "test": ["b", "a"]}), encoding="utf-8")
        (experiment / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (experiment / "config.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        artifacts = {name: sha(experiment / name) for name in ("samples.jsonl", "splits.json", "vocab.json", "config.json")}
        (experiment / "manifest.json").write_text(json.dumps({"experiment_id": "synthetic", "artifacts": artifacts}), encoding="utf-8")
        return experiment

    def test_bundle_ids_groups_pairs_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            experiment = self.make_experiment(root)
            bundle_path = prepare_evaluation_bundle(
                "1.0.0", experiment_path=experiment, output_root=root / "evaluations",
                contextual_pair_count=3, character_pair_count=3, max_all_pairs=10,
                distance_fn=lambda left, right: 0.0 if left == right else 1.0,
                metric_provenance={"metric_name": "synthetic"},
                code_provenance={"test": "0" * 64},
            )
            bundle = load_evaluation_bundle(bundle_path)
            self.assertEqual([x["occurrence_id"] for x in bundle["occurrences"]], ["a:0", "a:1", "b:0"])
            self.assertEqual(len(bundle["pairs"]["contextual"]), 3)
            self.assertEqual(bundle["coverage"]["levels"]["contextual"]["unordered_pair_population"]["eligible"], 3)
            self.assertEqual(len({x["pair_id"] for x in bundle["pairs"]["contextual"]}), 3)
            syllable = [x for x in bundle["groups"] if x["group_kind"] == "syllable"]
            self.assertEqual(sorted(x["occurrence_count"] for x in syllable), [1, 1, 1])
            self.assertTrue(all(x["distance_status"] == "ok" for x in bundle["pairs"]["contextual"]))
            with self.assertRaises(FileExistsError):
                prepare_evaluation_bundle(
                    "1.0.0", experiment_path=experiment, output_root=root / "evaluations", contextual_pair_count=1,
                    character_pair_count=1, distance_fn=lambda _left, _right: 0.0,
                    metric_provenance={"metric_name": "synthetic"},
                    code_provenance={"test": "0" * 64},
                )
            with (bundle_path / "occurrences.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_evaluation_bundle(bundle_path)

    def test_invalid_distance_is_retained(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            bundle_path = prepare_evaluation_bundle(
                "1.0.0", experiment_path=self.make_experiment(root), output_root=root / "out",
                contextual_pair_count=1, character_pair_count=1,
                distance_fn=lambda _left, _right: (_ for _ in ()).throw(KeyError("unsupported")),
                metric_provenance={"metric_name": "synthetic"},
                code_provenance={"test": "0" * 64},
            )
            row = load_evaluation_bundle(bundle_path)["pairs"]["contextual"][0]
            self.assertIsNone(row["dimsim_distance"])
            self.assertEqual(row["distance_status"], "invalid:KeyError")

    def make_zero_initial_experiment(self, root):
        experiment = root / "experiment"
        experiment.mkdir()
        vocab = {
            "acoustic_tokens": ["a1", "ma1", "ba1", "a2", "ma2"],
            "labels": {"initials": ["", "m", "b"], "finals": ["a"], "tones": ["1", "2"]},
        }
        samples = [
            {"sample_id": "a", "text": "阿妈", "acoustic_ids": [0, 1], "component_labels": [[0, 0, 0], [1, 0, 0]]},
            {"sample_id": "b", "text": "爸啊吗", "acoustic_ids": [2, 3, 4], "component_labels": [[2, 0, 0], [0, 0, 1], [1, 0, 1]]},
        ]
        (experiment / "samples.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in samples), encoding="utf-8")
        (experiment / "splits.json").write_text(json.dumps({"train": [], "validation": [], "test": ["a", "b"]}), encoding="utf-8")
        (experiment / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (experiment / "config.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        artifacts = {name: sha(experiment / name) for name in ("samples.jsonl", "splits.json", "vocab.json", "config.json")}
        (experiment / "manifest.json").write_text(json.dumps({"experiment_id": "synthetic", "artifacts": artifacts}), encoding="utf-8")
        return experiment

    def test_nonzero_policy_preserves_context_and_samples_only_eligible_population(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            path = prepare_evaluation_bundle(
                "1.0.0", experiment_path=self.make_zero_initial_experiment(root), output_root=root / "out",
                contextual_pair_count=3, character_pair_count=3, syllable_pair_count=None,
                distance_fn=lambda left, right: 0.0 if left == right else 1.0,
                metric_provenance={"metric_name": "synthetic"}, code_provenance={"test": "0" * 64},
            )
            bundle = load_evaluation_bundle(path)
            self.assertEqual([row["sample_id"] for row in bundle["sentences"]], ["a", "b"])
            self.assertEqual(len(bundle["occurrences"]), 5)
            self.assertEqual(len(bundle["groups"]), 10)
            for level, rows in bundle["pairs"].items():
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(not row["left_zero_initial"] and not row["right_zero_initial"] for row in rows))
                coverage = bundle["coverage"]["levels"][level]
                self.assertEqual(coverage["objects"], {
                    "original": 5, "eligible": 3, "excluded": 2,
                    "excluded_by_reason": {"zero_initial": 2},
                })
                self.assertEqual(coverage["unordered_pair_population"], {
                    "original": 10, "eligible": 3, "excluded": 7,
                    "excluded_by_reason": {"zero_initial": 7},
                })
                self.assertEqual(coverage["selection_count"], 3)
                self.assertEqual({frozenset((row["left_pinyin"], row["right_pinyin"])) for row in rows}, {
                    frozenset(("ba1", "ma1")), frozenset(("ba1", "ma2")), frozenset(("ma1", "ma2")),
                })

    def test_loader_rejects_old_schema_policy_and_inconsistent_zero_flags(self):
        def build(root):
            return prepare_evaluation_bundle(
                "1.0.0", experiment_path=self.make_experiment(root), output_root=root / "out",
                contextual_pair_count=1, character_pair_count=1,
                distance_fn=lambda _left, _right: 0.0,
                metric_provenance={"metric_name": "synthetic"}, code_provenance={"test": "0" * 64},
            )

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = build(Path(temporary)); manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["schema_version"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                load_evaluation_bundle(path)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = build(Path(temporary)); manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")); manifest["pairing_policy"] = "all_pairs"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "policy"):
                load_evaluation_bundle(path)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path = build(Path(temporary)); pair_path = path / "pairs" / "contextual.jsonl"
            rows = [json.loads(line) for line in pair_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["left_zero_initial"] = True
            pair_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest_path = path / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["pairs/contextual.jsonl"] = sha(pair_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent zero-initial flags"):
                load_evaluation_bundle(path)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            path = prepare_evaluation_bundle(
                "1.0.0", experiment_path=self.make_zero_initial_experiment(root), output_root=root / "out",
                contextual_pair_count=1, character_pair_count=1,
                distance_fn=lambda _left, _right: 0.0,
                metric_provenance={"metric_name": "synthetic"}, code_provenance={"test": "0" * 64},
            )
            occurrences = {row["occurrence_id"]: row for row in map(json.loads, (path / "occurrences.jsonl").read_text(encoding="utf-8").splitlines())}
            zero_id = next(object_id for object_id, row in occurrences.items() if row["initial"] == "")
            nonzero_id = next(object_id for object_id, row in occurrences.items() if row["initial"] != "")
            left_id, right_id = sorted((zero_id, nonzero_id)); left, right = occurrences[left_id], occurrences[right_id]
            pair_path = path / "pairs" / "contextual.jsonl"
            row = json.loads(pair_path.read_text(encoding="utf-8").splitlines()[0])
            row.update({
                "left_id": left_id, "right_id": right_id, "left_pinyin": left["pinyin"], "right_pinyin": right["pinyin"],
                "pair_id": hashlib.sha256(json.dumps([left_id, right_id], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
                "left_zero_initial": left["initial"] == "", "right_zero_initial": right["initial"] == "",
                "no_zero_initial_diagnostic": False,
            })
            pair_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest_path = path / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["pairs/contextual.jsonl"] = sha(pair_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "violates nonzero_nonzero policy"):
                load_evaluation_bundle(path)

    def test_evaluation_id_must_be_dotted_version(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "X.Y.Z"):
                prepare_evaluation_bundle(
                    "eval", experiment_path=self.make_experiment(root), output_root=root / "out",
                    distance_fn=lambda _left, _right: 0.0,
                    metric_provenance={"metric_name": "synthetic"}, code_provenance={"test": "0" * 64},
                )

    def test_non_string_initial_labels_and_loaded_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary); experiment = self.make_experiment(root)
            vocab_path = experiment / "vocab.json"; vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            vocab["labels"]["initials"][0] = None; vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
            manifest_path = experiment / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["vocab.json"] = sha(vocab_path); manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "component label vocabulary"):
                prepare_evaluation_bundle(
                    "1.0.0", experiment_path=experiment, output_root=root / "out",
                    distance_fn=lambda _left, _right: 0.0,
                    metric_provenance={"metric_name": "synthetic"}, code_provenance={"test": "0" * 64},
                )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            path = prepare_evaluation_bundle(
                "1.0.0", experiment_path=self.make_experiment(root), output_root=root / "out",
                contextual_pair_count=1, character_pair_count=1, distance_fn=lambda _left, _right: 0.0,
                metric_provenance={"metric_name": "synthetic"}, code_provenance={"test": "0" * 64},
            )
            occurrence_path = path / "occurrences.jsonl"
            rows = [json.loads(line) for line in occurrence_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["initial"] = None
            occurrence_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest_path = path / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["occurrences.jsonl"] = sha(occurrence_path); manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "initial metadata must be strings"):
                load_evaluation_bundle(path)


if __name__ == "__main__":
    unittest.main()
