"""Pinned Chinese BERT inputs and character alignment."""
from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path

from .architecture import BERT_REQUIREMENTS, validate_bert_snapshot
from .artifacts import safe_artifact, sha256_file


def is_bert(config):
    return config["architecture"].get("encoder", {}).get("type") == "bert"


class BertInputs:
    def __init__(self, spec, *, pretrained=False):
        validate_bert_snapshot(spec)
        for requirement in BERT_REQUIREMENTS:
            name, expected = requirement.split("==")
            if version(name) != expected:
                raise RuntimeError(f"Required {requirement}")
        from transformers import BertConfig, BertTokenizerFast

        location = os.environ.get("PHONETICS_BERT_SNAPSHOT")
        if not location:
            raise ValueError("set PHONETICS_BERT_SNAPSHOT to the pinned BERT snapshot directory")
        self.root = Path(location).resolve(strict=True)
        for name, digest in spec["files"].items():
            if name == "model.safetensors" and not pretrained:
                continue
            if sha256_file(safe_artifact(self.root, name)) != digest:
                raise ValueError(f"BERT snapshot checksum mismatch: {name}")
        self.config = BertConfig.from_json_file(str(self.root / "config.json"))
        settings = json.loads((self.root / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.tokenizer = BertTokenizerFast(
            vocab_file=str(self.root / "vocab.txt"), tokenizer_file=str(self.root / "tokenizer.json"),
            **{key: settings[key] for key in ("do_lower_case", "tokenize_chinese_chars", "strip_accents",
                                            "model_max_length") if key in settings},
        )
        if (not self.tokenizer.is_fast or len(self.tokenizer) != self.config.vocab_size
                or self.config.hidden_size != 768 or self.config.model_type != "bert"
                or self.tokenizer.pad_token_id != self.config.pad_token_id):
            raise ValueError("incompatible Chinese BERT config or tokenizer")
        self.tokenizer.padding_side = "right"

    def encode(self, samples):
        texts = [sample["text"] for sample in samples]
        encoded = self.tokenizer([list(text) for text in texts], is_split_into_words=True,
                                 add_special_tokens=True, padding=False, truncation=False,
                                 return_attention_mask=False, return_token_type_ids=False)
        rows = []
        for index, sample in enumerate(samples):
            ids = encoded["input_ids"][index]
            mapping = encoded.word_ids(batch_index=index)
            positions = [i for i, word in enumerate(mapping) if word is not None]
            if ([mapping[i] for i in positions] != list(range(len(texts[index])))
                    or len(ids) != len(positions) + 2 or ids[0] != self.tokenizer.cls_token_id
                    or ids[-1] != self.tokenizer.sep_token_id
                    or any(ids[i] in (self.tokenizer.pad_token_id, self.tokenizer.cls_token_id,
                                      self.tokenizer.sep_token_id) for i in positions)):
                raise ValueError(f"BERT character alignment failed for {sample['sample_id']}")
            if len(ids) > self.config.max_position_embeddings:
                raise ValueError(f"BERT position limit exceeded by {sample['sample_id']}: {len(ids)} tokens")
            rows.append((ids, positions, [ids[i] == self.tokenizer.unk_token_id for i in positions]))
        return rows

    def prepare(self, samples, *, split):
        rows = {}
        for start in range(0, len(samples), 256):
            batch = samples[start:start + 256]
            try:
                encoded = self.encode(batch)
            except ValueError as error:
                raise ValueError(f"{split}: {error}") from error
            rows.update((sample["sample_id"], row) for sample, row in zip(batch, encoded))
        return rows

    def batch(self, rows, char_mask):
        import torch

        size = len(rows)
        width = max(len(ids) for ids, _, _ in rows)
        ids = torch.full((size, width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        positions = torch.zeros(char_mask.shape, dtype=torch.long)
        for index, (tokens, aligned, _) in enumerate(rows):
            ids[index, :len(tokens)] = torch.as_tensor(tokens)
            attention[index, :len(tokens)] = 1
            positions[index, :len(aligned)] = torch.as_tensor(aligned)
        return {"bert_input_ids": ids, "bert_attention_mask": attention, "bert_positions": positions,
                "char_mask": char_mask}
