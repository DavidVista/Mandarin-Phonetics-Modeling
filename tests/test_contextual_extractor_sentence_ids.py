"""Exercise the real extraction function with lightweight tensor/model doubles."""
import ast
import copy
import hashlib
import platform
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "src/phonetics/evaluation/encoders.py"


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.ndim = self.values.ndim
        self.dtype = self.values.dtype

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class ContextualExtractorSentenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        names = ("manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json")
        for name in names:
            (self.root / name).write_text("{}", encoding="utf-8")
        hashes = {name: self.sha(self.root / name) for name in names}
        (self.root / "best.pt").write_bytes(b"synthetic checkpoint")
        self.bundle = {
            "manifest": {"schema_version": 3, "source_experiment": {"hashes": hashes}},
            "sentences": [{"sentence_id": "sample-a", "text": "你好"}],
            "occurrences": [
                {"sample_id": "sample-a", "occurrence_id": f"sample-a:{i}",
                 "character_position": i, "character": char}
                for i, char in enumerate("你好")
            ],
        }
        mask = Tensor([[True, True]])
        hidden = Tensor(np.arange(6, dtype=np.float32).reshape(1, 2, 3) + 1)

        class Model:
            context_dim = 3
            training = False

            def load_state_dict(self, state, strict):
                pass

            def to(self, device):
                return self

            def eval(self):
                return self

            def encode(self, batch):
                return hidden, mask

        checkpoint = {
            "model_id": "syllable", "seed": 0, "state_dict": {},
            "experiment_hashes": dict(hashes, **{"acoustic_targets.npy": "a" * 64}),
            "config_sha256": hashes["config.json"], "vocab_sha256": hashes["vocab.json"],
        }
        torch = types.SimpleNamespace(
            load=lambda *a, **k: checkpoint, device=lambda value: value,
            no_grad=nullcontext, is_tensor=lambda value: isinstance(value, Tensor),
            equal=lambda a, b: np.array_equal(a.values, b.values), bool=np.dtype(bool), __version__="synthetic",
        )
        self.saved = {}

        def save(output_dir, **values):
            self.saved.update(values)
            return Path(output_dir)

        self.ns = {
            "np": np, "Path": Path, "hashlib": hashlib, "platform": platform, "torch": torch,
            "load_evaluation_bundle": lambda path: self.bundle,
            "load_experiment": lambda path: {
                "manifest": {"artifacts": {"acoustic_targets.npy": "a" * 64}},
                "config": {}, "vocab": {}, "samples": {"sample-a": {"text": "你好"}},
            },
            "canonical_model_id": lambda value: value, "parse_model_id": lambda value: (value, None),
            "build_encoder": lambda *a, **k: Model(),
            "collate_experiment": lambda samples: {"char_mask": mask},
            "validate_matrix": lambda ids, matrix: matrix,
            "save_contextual_representations": save,
        }
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in {"_sha256", "_checkpoint_hash_contract", "extract_bilstm"}]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), self.ns)

    def extract(self):
        return self.ns["extract_bilstm"](
            self.root, experiment_path=self.root, checkpoint_path=self.root / "best.pt",
            output_dir=self.root / "export", contextual_only=True, source_sha256="b" * 64,
        )

    def test_schema_three_sentence_ids_reach_aligned_export(self):
        self.extract()
        self.assertEqual(self.saved["occurrence_ids"], ["sample-a:0", "sample-a:1"])
        np.testing.assert_array_equal(self.saved["hidden"], np.arange(1, 7, dtype=np.float32).reshape(2, 3))

    def test_wrong_character_position_is_rejected(self):
        self.bundle = copy.deepcopy(self.bundle)
        self.bundle["occurrences"][1]["character_position"] = 2
        with self.assertRaisesRegex(ValueError, "exactly cover source sentence"):
            self.extract()
        self.assertEqual(self.saved, {})
