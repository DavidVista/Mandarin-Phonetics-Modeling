"""Small deterministic checks for the contextual encoder extraction contract."""

import sys
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Torch is unavailable; synthetic encoder extraction checks are not executable")
class ContextualEncoderTests(unittest.TestCase):
    def test_eval_outputs_are_deterministic_and_padding_invariant(self):
        from phonetics.model import build_encoder
        from phonetics.data import collate_experiment
        from phonetics.pinyin import decompose

        vocab = {"ids": {"tokens": ["[PAD]", "甲", "乙"]},
                 "labels": {"initials": [""], "finals": ["a"], "tones": ["1"]},
                 "syllables": ["ma1", "ba4"]}
        model = build_encoder({"architecture": {"embedding_dim": 64, "context_dim": 128,
                                                  "acoustic_dim": 3}}, vocab,
                              model_id="components", seed=11).eval()
        def sample(sid, text, sequences):
            return {"sample_id": sid, "text": text, "positions": list(range(len(text))),
                    "sentence_length": len(text), "ids_sequences": sequences,
                    "component_lengths": [len(x) for x in sequences],
                    "component_labels": [list(decompose("ma1")) for _ in text],
                    "syllable_ids": [0] * len(text), "acoustic_ids": [0] * len(text)}
        short = sample("s", "甲", [[1]])
        long = sample("l", "乙甲", [[2], [1]])
        with torch.no_grad():
            alone, mask_alone = model.encode(collate_experiment([short]))
            mixed, mask_mixed = model.encode(collate_experiment([short, long]))
            repeated, _ = model.encode(collate_experiment([short]))
        self.assertFalse(model.training)
        torch.testing.assert_close(alone[0, 0], mixed[0, 0])
        torch.testing.assert_close(alone, repeated)
        self.assertEqual(mask_alone.tolist(), [[True]])
        self.assertEqual(mask_mixed.tolist(), [[True, False], [True, True]])
        self.assertEqual(tuple(alone.shape[-1:]), (128,))

    def test_extractor_loads_synthetic_checkpoint_and_persists_raw_prehead_h(self):
        from phonetics.model import build_encoder
        from phonetics.artifacts import sha256_file
        from phonetics.evaluation import encoders, representations
        vocab = {"ids": {"tokens": ["[PAD]", "甲"]},
                 "labels": {"initials": [""], "finals": ["a"], "tones": ["1"]},
                 "syllables": ["ma1"]}
        config = {"architecture": {"embedding_dim": 64, "context_dim": 128, "acoustic_dim": 3}}
        def sample(sid, text):
            return {"sample_id": sid, "text": text, "positions": list(range(len(text))),
                    "sentence_length": len(text), "ids_sequences": [[1] for _ in text],
                    "component_lengths": [1 for _ in text], "component_labels": [[0, 0, 0] for _ in text],
                    "syllable_ids": [0 for _ in text], "acoustic_ids": [0 for _ in text]}
        samples = {"s": sample("s", "甲"), "l": sample("l", "甲甲")}
        occurrences = [{"occurrence_id": "o1", "sample_id": "s", "character_position": 0, "character": "甲"},
                       {"occurrence_id": "o2", "sample_id": "l", "character_position": 0, "character": "甲"},
                       {"occurrence_id": "o3", "sample_id": "l", "character_position": 1, "character": "甲"}]
        evaluation_bundle = {"manifest": {"source_experiment": {"hashes": {}}},
                             "occurrences": occurrences, "sentences": [{"sample_id": "s"}, {"sample_id": "l"}],
                             "groups": []}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); experiment = root / "experiment"; experiment.mkdir()
            bundle_path = root / "bundle"; bundle_path.mkdir()
            (bundle_path / "manifest.json").write_text("{}\n", encoding="utf-8")
            for name in ("manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json"):
                (experiment / name).write_text("{}\n", encoding="utf-8")
            source_hashes = {name: sha256_file(experiment / name) for name in
                             ("manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json")}
            evaluation_bundle["manifest"]["source_experiment"]["hashes"] = source_hashes
            experiment_manifest = {"artifacts": {"acoustic_targets.npy": "a" * 64}}
            hashes = encoders._checkpoint_hash_contract(source_hashes, experiment_manifest)
            model = build_encoder(config, vocab, "components", seed=3)
            checkpoint = {"model_id": "components", "seed": 3, "state_dict": model.state_dict(),
                          "experiment_hashes": hashes, "config_sha256": source_hashes["config.json"],
                          "vocab_sha256": source_hashes["vocab.json"]}
            checkpoint_path = root / "checkpoint.pt"; torch.save(checkpoint, checkpoint_path)
            loaded_experiment = {"config": config, "vocab": vocab, "manifest": experiment_manifest,
                                 "samples": samples}
            with patch.object(encoders, "load_evaluation_bundle", return_value=evaluation_bundle), \
                 patch.object(encoders, "load_experiment", return_value=loaded_experiment):
                result = encoders.extract_bilstm(bundle_path, experiment_path=experiment,
                    checkpoint_path=checkpoint_path, output_dir=root / "representations",
                    batch_size=2, contextual_only=True)
            loaded = representations.load_contextual_representations(result)
            self.assertEqual(loaded["occurrence_ids"], ["o1", "o2", "o3"])
            self.assertEqual(loaded["hidden"].shape, (3, 128))
            self.assertEqual(loaded["manifest"]["provenance"]["seed"], 3)
            self.assertTrue(loaded["manifest"]["evaluation_mode"])
            self.assertEqual(loaded["hidden"].dtype, np.dtype("float32"))
            direct_model = build_encoder(config, vocab, "components", seed=3).eval()
            direct_model.load_state_dict(checkpoint["state_dict"], strict=True)
            with torch.no_grad():
                direct, direct_mask = direct_model.encode(__import__("phonetics.data", fromlist=["collate_experiment"])
                                                          .collate_experiment([samples["s"], samples["l"]]))
            self.assertEqual(direct_mask.tolist(), [[True, False], [True, True]])
            np.testing.assert_array_equal(loaded["hidden"],
                np.stack([direct[0, 0].numpy(), direct[1, 0].numpy(), direct[1, 1].numpy()]))
            with patch.object(encoders, "load_evaluation_bundle", return_value=evaluation_bundle), \
                 patch.object(encoders, "load_experiment", return_value=loaded_experiment):
                with self.assertRaisesRegex(ValueError, "expected_model_id"):
                    encoders.extract_bilstm(bundle_path, experiment_path=experiment,
                        checkpoint_path=checkpoint_path, output_dir=root / "mismatch-model",
                        contextual_only=True, expected_model_id="acoustic")
                with self.assertRaisesRegex(ValueError, "expected_seed"):
                    encoders.extract_bilstm(bundle_path, experiment_path=experiment,
                        checkpoint_path=checkpoint_path, output_dir=root / "mismatch-seed",
                        contextual_only=True, expected_seed=4)
                bad_checkpoint = dict(checkpoint)
                bad_checkpoint["experiment_hashes"] = {**hashes, "config.json": "0" * 64}
                bad_checkpoint_path = root / "bad-checkpoint.pt"; torch.save(bad_checkpoint, bad_checkpoint_path)
                with self.assertRaisesRegex(ValueError, "provenance"):
                    encoders.extract_bilstm(bundle_path, experiment_path=experiment,
                        checkpoint_path=bad_checkpoint_path, output_dir=root / "mismatch-provenance",
                        contextual_only=True)
            misaligned_bundle = dict(evaluation_bundle)
            misaligned_bundle["occurrences"] = occurrences[:-1]
            with patch.object(encoders, "load_evaluation_bundle", return_value=misaligned_bundle), \
                 patch.object(encoders, "load_experiment", return_value=loaded_experiment):
                with self.assertRaisesRegex(ValueError, "cover source sentence"):
                    encoders.extract_bilstm(bundle_path, experiment_path=experiment,
                        checkpoint_path=checkpoint_path, output_dir=root / "misaligned",
                        contextual_only=True)
            class FakeEncoder:
                training = False
                context_dim = 128
                def load_state_dict(self, *_args, **_kwargs): return self
                def to(self, *_args, **_kwargs): return self
                def eval(self): return self
            class WrongShape(FakeEncoder):
                def encode(self, _batch): return torch.zeros((1, 1, 128)), torch.ones((1, 1), dtype=torch.bool)
            class WrongMask(FakeEncoder):
                def encode(self, batch):
                    mask = batch["char_mask"]
                    return torch.ones((*mask.shape, 128)), torch.zeros_like(mask)
            for bad_model, message in ((WrongShape(), "output shape"), (WrongMask(), "output mask")):
                with patch.object(encoders, "load_evaluation_bundle", return_value=evaluation_bundle), \
                     patch.object(encoders, "load_experiment", return_value=loaded_experiment), \
                     patch.object(encoders, "build_encoder", return_value=bad_model):
                    with self.assertRaisesRegex(ValueError, message):
                        encoders.extract_bilstm(bundle_path, experiment_path=experiment,
                            checkpoint_path=checkpoint_path, output_dir=root / f"bad-{message.replace(' ', '-')}",
                            contextual_only=True)


if __name__ == "__main__":
    unittest.main()
