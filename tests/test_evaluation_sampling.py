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
                "eval", experiment_path=experiment, output_root=root / "evaluations",
                contextual_pair_count=3, character_pair_count=3, max_all_pairs=10,
                distance_fn=lambda left, right: 0.0 if left == right else 1.0,
                metric_provenance={"metric_name": "synthetic"},
                code_provenance={"test": "0" * 64},
            )
            bundle = load_evaluation_bundle(bundle_path)
            self.assertEqual([x["occurrence_id"] for x in bundle["occurrences"]], ["a:0", "a:1", "b:0"])
            self.assertEqual(len(bundle["pairs"]["contextual"]), 3)
            self.assertEqual(bundle["coverage"]["levels"]["contextual"]["eligible_pair_population"], 3)
            self.assertEqual(len({x["pair_id"] for x in bundle["pairs"]["contextual"]}), 3)
            syllable = [x for x in bundle["groups"] if x["group_kind"] == "syllable"]
            self.assertEqual(sorted(x["occurrence_count"] for x in syllable), [1, 1, 1])
            self.assertTrue(all(x["distance_status"] == "ok" for x in bundle["pairs"]["contextual"]))
            with self.assertRaises(FileExistsError):
                prepare_evaluation_bundle(
                    "eval", experiment_path=experiment, output_root=root / "evaluations", contextual_pair_count=1,
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
                "eval", experiment_path=self.make_experiment(root), output_root=root / "out",
                contextual_pair_count=1, character_pair_count=1,
                distance_fn=lambda _left, _right: (_ for _ in ()).throw(KeyError("unsupported")),
                metric_provenance={"metric_name": "synthetic"},
                code_provenance={"test": "0" * 64},
            )
            row = load_evaluation_bundle(bundle_path)["pairs"]["contextual"][0]
            self.assertIsNone(row["dimsim_distance"])
            self.assertEqual(row["distance_status"], "invalid:KeyError")


if __name__ == "__main__":
    unittest.main()
