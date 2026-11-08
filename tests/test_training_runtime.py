"""Static objective and diagnostic contracts; no Torch import or execution."""
import ast
import math
import unittest
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/"src/phonetics"

class TrainingStaticContractTests(unittest.TestCase):
    def test_runtime_sources_parse_and_remove_legacy_objective_protocol(self):
        for name in ("model.py", "data.py", "metrics.py", "training.py", "reproducibility.py", "evaluation/encoders.py"):
            source = (SOURCE / name).read_text(encoding="utf-8")
            ast.parse(source)
        model = (SOURCE / "model.py").read_text(encoding="utf-8")
        training = (SOURCE / "training.py").read_text(encoding="utf-8")
        metrics = (SOURCE / "metrics.py").read_text(encoding="utf-8")
        self.assertIn("parse_model_id", model)
        self.assertIn("weight_decay=0.0", training)
        self.assertIn("torch.optim.Adam(", training)
        self.assertNotIn("AdamW", training)
        self.assertNotIn("mse_only", metrics)
        self.assertNotIn("gradient clipping", training.lower())

    def test_if_pair_inventory_matches_only_canonical_strict_pairs(self):
        import sys
        sys.path.insert(0, str(PROJECT / "src"))
        from phonetics.pinyin import VALID_SYLLABLES, canonical_label, decompose
        from phonetics.training_data import valid_initial_final_pairs
        expected = set()
        for base in VALID_SYLLABLES:
            try:
                canonical_label(base + "1", tone_convention="explicit")
                initial, final, _ = decompose(base + "1")
                expected.add((initial, final))
            except (ValueError, TypeError, IndexError):
                continue
        self.assertEqual(valid_initial_final_pairs(), frozenset(expected))

    def test_combined_objective_fixture_dimensions_cover_canonical_ma1_targets(self):
        import sys
        sys.path.insert(0, str(PROJECT / "src"))
        from phonetics.pinyin import decompose
        target = decompose("ma1")
        class_sizes = (22, 36, 5)
        self.assertEqual(target, (11, 0, 0))
        self.assertTrue(all(0 <= label < size for label, size in zip(target, class_sizes)))

    def test_eval_hash_contract_completes_five_source_hashes_with_acoustic_pin(self):
        source = (SOURCE / "evaluation" / "encoders.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_checkpoint_hash_contract")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "hash-contract", "exec"), namespace)
        hashes = {name: str(index) * 64 for index, name in enumerate(
            ("manifest.json", "config.json", "vocab.json", "samples.jsonl", "splits.json"), 1)}
        acoustic = "a" * 64
        self.assertEqual(namespace["_checkpoint_hash_contract"](hashes, {"artifacts": {"acoustic_targets.npy": acoustic}}),
                         {**hashes, "acoustic_targets.npy": acoustic})

    def test_nonfinite_failure_helper_preserves_current_epoch_gradient_history(self):
        source = (SOURCE / "training.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_failure_diagnostic")
        summary = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_gradient_summary")
        namespace = {"statistics": __import__("statistics"), "math": math}
        exec(compile(ast.Module(body=[helper, summary], type_ignores=[]), "failure-diagnostic", "exec"), namespace)
        diagnostic = namespace["_failure_diagnostic"]({"event": "nonfinite_loss"}, [2.0], 0)
        self.assertEqual(diagnostic["gradient_norms"], [2.0])
        self.assertEqual(namespace["_gradient_summary"](diagnostic["gradient_norms"], diagnostic["nonfinite_count"]),
                         {"mean": 2.0, "std": 0.0, "max": 2.0, "nonfinite_count": 0})
        epoch_source = ast.get_source_segment(source, next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_epoch"))
        self.assertGreaterEqual(epoch_source.count("_failure_diagnostic(record, gradient_norms, nonfinite_gradients)"), 2)
