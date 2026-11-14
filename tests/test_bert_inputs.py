"""CPU-only checks for the BERT encoder configuration and token alignment."""
from types import SimpleNamespace
import unittest

from phonetics.architecture import BERT_FILES, BERT_MODEL_ID, BERT_REVISION
from phonetics.bert import BertInputs
from phonetics.experiment import _validate_config, default_experiment_config


class BertInputTests(unittest.TestCase):
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
