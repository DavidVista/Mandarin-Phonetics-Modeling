import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy  # noqa: F401
    import matplotlib  # noqa: F401
    import torch
except ImportError as error:
    raise unittest.SkipTest(f"scientific evaluation dependencies unavailable: {error}")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phonetics.evaluation.bundle import prepare_evaluation_bundle
from phonetics.evaluation.runner import run_evaluation
from phonetics.evaluation.encoders import extract_bilstm


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunnerTests(unittest.TestCase):
    def test_random_baseline_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary); experiment = root / "experiment"; experiment.mkdir()
            vocab = {"acoustic_tokens": ["ma1", "ma2", "ba1"],
                     "labels": {"initials": ["m", "b"], "finals": ["a"], "tones": ["1", "2"]}}
            samples = [
                {"sample_id": "a", "text": "妈妈", "acoustic_ids": [0, 1], "component_labels": [[0, 0, 0], [0, 0, 1]]},
                {"sample_id": "b", "text": "爸", "acoustic_ids": [2], "component_labels": [[1, 0, 0]]},
            ]
            (experiment / "samples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
            (experiment / "splits.json").write_text(json.dumps({"train": [], "validation": [], "test": ["a", "b"]}), encoding="utf-8")
            (experiment / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
            (experiment / "config.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            artifacts = {name: sha(experiment / name) for name in ("samples.jsonl", "splits.json", "vocab.json", "config.json")}
            (experiment / "manifest.json").write_text(json.dumps({"experiment_id": "synthetic", "artifacts": artifacts}), encoding="utf-8")
            bundle = prepare_evaluation_bundle(
                "eval", experiment_path=experiment, output_root=root / "bundles", contextual_pair_count=3,
                character_pair_count=3, max_all_pairs=10, distance_fn=lambda a, b: 0.0 if a == b else 1.0,
                metric_provenance={"metric_name": "synthetic"},
                code_provenance={"test": "0" * 64},
            )
            run = run_evaluation(bundle, output_root=root / "runs", run_id="random-1", adapter="random",
                                 random_seed=7, random_dimension=4, code_provenance={"test": "0" * 64})
            self.assertTrue((run / "actual_config.json").is_file())
            self.assertTrue((run / "analysis" / "metrics.json").is_file())
            rows = (run / "analysis" / "similarities" / "contextual.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)
            with self.assertRaises(FileExistsError):
                run_evaluation(bundle, output_root=root / "runs", run_id="random-1", adapter="random",
                               random_seed=7, random_dimension=4, code_provenance={"test": "0" * 64})
            wrong_checkpoint = root / "wrong.pt"
            torch.save({"state_dict": {}, "manifest_sha256": "wrong", "config_sha256": "wrong", "vocab_sha256": "wrong"}, wrong_checkpoint)
            with self.assertRaisesRegex(ValueError, "checkpoint provenance"):
                extract_bilstm(bundle, experiment_path=experiment, checkpoint_path=wrong_checkpoint,
                               output_dir=root / "bad-representations")


if __name__ == "__main__":
    unittest.main()
