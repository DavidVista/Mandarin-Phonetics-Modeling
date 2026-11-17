"""CPU-only checks for the BERT encoder configuration and token alignment."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from phonetics.architecture import BERT_FILES, BERT_MODEL_ID, BERT_REVISION
from phonetics.bert import BertInputs
from phonetics.bert_snapshot import load_bert_inputs
from phonetics.experiment import _validate_config, default_experiment_config


class BertInputTests(unittest.TestCase):
    def test_downloads_pinned_files_and_reuses_verified_snapshot(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            files = {"config.json": b"config", "model.safetensors": b"weights"}
            spec = {"model_id": BERT_MODEL_ID, "revision": BERT_REVISION,
                    "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
            requests = []

            def download(*, repo_id, revision, allow_patterns, local_dir):
                requests.append((repo_id, revision, allow_patterns))
                for name in allow_patterns:
                    (local_dir / name).write_bytes(files[name])

            with (patch("phonetics.bert_snapshot.validate_bert_snapshot"),
                  patch("phonetics.bert_snapshot.BertInputs",
                        side_effect=lambda *a, **k: Path(os.environ["PHONETICS_BERT_SNAPSHOT"])),
                  patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}),
                  patch.dict(os.environ, {"PHONETICS_BERT_SNAPSHOT": "", "PHONETICS_BERT_CACHE": directory})):
                tokenizer = load_bert_inputs(spec)
                self.assertFalse((tokenizer / "model.safetensors").exists())
                full = load_bert_inputs(spec, pretrained=True)
                self.assertEqual(load_bert_inputs(spec), full)
                self.assertEqual(load_bert_inputs(spec, pretrained=True), full)
                self.assertEqual(os.environ["PHONETICS_BERT_SNAPSHOT"], "")
                os.environ["PHONETICS_BERT_SNAPSHOT"] = str(tokenizer)
                self.assertEqual(load_bert_inputs(spec), tokenizer)
                self.assertEqual(requests, [(BERT_MODEL_ID, BERT_REVISION, ["config.json"]),
                                            (BERT_MODEL_ID, BERT_REVISION, list(files))])
                os.environ["PHONETICS_BERT_SNAPSHOT"] = ""
                files["config.json"] = b"modified"
                os.environ["PHONETICS_BERT_CACHE"] = str(Path(directory) / "corrupt")
                with self.assertRaisesRegex(ValueError, "checksum mismatch: config.json"):
                    load_bert_inputs(spec)

    def test_accepts_pinned_encoder_and_rejects_missing_tokenizer_hash(self):
        config = default_experiment_config()
        encoder = {"type": "bert", "model_id": BERT_MODEL_ID,
                   "revision": BERT_REVISION, "files": dict(BERT_FILES)}
        config["architecture"]["encoder"] = encoder
        config["training"].update(optimizer="AdamW", learning_rate=2e-5,
                                  scheduler={"type": "linear", "warmup_ratio": 0.05})
        self.assertEqual(_validate_config(config, 1024), config)
        encoder["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "supported revision"):
            _validate_config(config, 1024)
        with self.assertRaisesRegex(ValueError, "supported revision"):
            BertInputs(encoder)
        encoder["revision"] = BERT_REVISION
        encoder["files"]["tokenizer.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            _validate_config(config, 1024)
        del encoder["files"]["tokenizer.json"]
        with self.assertRaisesRegex(ValueError, "SHA256"):
            _validate_config(config, 1024)

    def test_aligns_repeated_characters_and_unknown_and_rejects_invalid_rows(self):
        class Encoding(dict):
            def word_ids(self, batch_index):
                return self.mapping

        class Tokenizer:
            pad_token_id, unk_token_id, cls_token_id, sep_token_id = 0, 1, 2, 3

            def __call__(self, texts, **settings):
                self.texts = texts
                self.settings = settings
                return encoded

        inputs = BertInputs.__new__(BertInputs)
        inputs.config = SimpleNamespace(max_position_embeddings=5)
        inputs.tokenizer = Tokenizer()
        encoded = Encoding(input_ids=[[2, 4, 4, 1, 3]])
        encoded.mapping = [None, 0, 1, 2, None]
        samples = [{"sample_id": "example", "text": "妈妈𠀀"}]
        self.assertEqual(inputs.encode(samples), [([2, 4, 4, 1, 3], [1, 2, 3], [False, False, True])])
        self.assertEqual(inputs.tokenizer.texts, [list("妈妈𠀀")])
        self.assertFalse(inputs.tokenizer.settings["truncation"])
        inputs.config.max_position_embeddings = 4
        with self.assertRaisesRegex(ValueError, "position limit exceeded by example"):
            inputs.encode(samples)
        for mapping in ([None, 0, 0, 2, None], [None, 0, None, 2, None]):
            encoded.mapping = mapping
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, "alignment failed for example"):
                inputs.encode(samples)
