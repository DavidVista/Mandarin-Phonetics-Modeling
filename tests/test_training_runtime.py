import ast
import json
import math
import importlib.util
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "src" / "phonetics"

try:
    import torch
except ImportError:
    torch = None


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

    def test_readable_notebook_definitions_and_provenance(self):
        builder_path = PROJECT.parent / "meta" / "tools" / "build_training_notebooks.py"
        spec = importlib.util.spec_from_file_location("readable_builder", builder_path)
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        notebook = builder.notebook("components-acoustic", composite=False)
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        joined = "\n".join(cells.values())
        for forbidden in ("EMBEDDED_SOURCES", "exec(", "materialize_embedded_package", "<embedded>"):
            self.assertNotIn(forbidden, joined)
        trainer = cells["checkpoints-summaries"]
        self.assertIn('BUILD_METADATA["source_sha256"][name]', trainer)
        self.assertNotIn("__file__", trainer)
        self.assertIn("Path(OUTPUT_ROOT)", trainer)
        expected = {
            "batch-construction": {"collate_experiment", "create_loaders"},
            "character-encoder": {"IDSCharacterEncoder"},
            "context-readouts": {"SharedEncoder", "TrainingModel", "ClassifierHead"},
            "objectives-metrics": {"batch_objective", "finalize_epoch"},
            "reproducibility": {"build_encoder", "seed_everything"},
            "training-diagnostics": {"_epoch", "_gradient_norm"},
            "checkpoints-summaries": {"train_experiment"},
        }
        for cell_id, names in expected.items():
            definitions = {getattr(n, "name", None) for n in ast.parse(cells[cell_id]).body}
            self.assertTrue(names.issubset(definitions), cell_id)

    def test_all_eight_generated_training_notebooks_are_guarded_and_parse(self):
        families = ("syllable", "components", "acoustic", "components-acoustic")
        for family in families:
            for suffix in ("", "-composite"):
                path = PROJECT / "notebooks" / "Training" / family / f"{family}{suffix}.ipynb"
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        source = "".join(cell["source"])
                        ast.parse(source)
                settings = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["id"] == "settings")
                self.assertIn("RUN_REAL_TRAINING = False", settings)
                self.assertIn("719fc321a121c06a25c5247533564c15ce02f26e6bc82aa20c33fc567a227185", settings)


@unittest.skipIf(torch is None, "Torch is unavailable; model/collate runtime tests are not executable")
class TorchTrainingRuntimeTests(unittest.TestCase):
    def setUp(self):
        from phonetics.pinyin import decompose
        self.components = list(decompose("ma1"))
        self.vocab = {"ids": {"tokens": ["[PAD]", "x", "y"]},
                      "labels": {"initials": [""], "finals": ["a"], "tones": ["1"]},
                      "syllables": ["ma1", "ba4"]}
        self.architecture = {"embedding_dim": 64, "context_dim": 128, "acoustic_dim": 3}

    def sample(self, sid, text, sequences):
        labels = [self.components for _ in text]
        return {"sample_id": sid, "text": text, "positions": list(range(len(text))),
                "sentence_length": len(text), "ids_sequences": sequences,
                "component_lengths": [len(x) for x in sequences],
                "component_labels": labels, "syllable_ids": [0] * len(text),
                "acoustic_ids": [0] * len(text)}

    def test_collate_keeps_real_class_zero_and_excludes_sentence_padding(self):
        from phonetics.data import collate_experiment
        batch = collate_experiment([self.sample("short", "甲", [[1]])])
        mixed = collate_experiment([self.sample("short", "甲", [[1]]), self.sample("long", "乙丙", [[1, 2], [2]])])
        self.assertEqual(batch["component_labels"][0, 0, 0].item(), self.components[0])
        self.assertEqual(batch["syllable_ids"][0, 0].item(), 0)
        self.assertTrue(batch["char_mask"][0, 0].item())
        self.assertEqual(mixed["component_labels"][0, 1, 0].item(), -100)
        self.assertEqual(mixed["acoustic_ids"][0, 1].item(), -1)
        self.assertFalse(mixed["char_mask"][0, 1].item())

    def test_shared_initialization_heads_and_padding_invariance(self):
        from phonetics.model import build_encoder, shared_encoder_sha256
        from phonetics.data import collate_experiment
        config = {"architecture": self.architecture}
        components = build_encoder(config, self.vocab, "components", seed=4).eval()
        syllable = build_encoder(config, self.vocab, "syllable", seed=4).eval()
        acoustic = build_encoder(config, self.vocab, "acoustic", seed=4).eval()
        self.assertEqual(shared_encoder_sha256(components), shared_encoder_sha256(syllable))
        self.assertEqual(shared_encoder_sha256(components), shared_encoder_sha256(acoustic))
        self.assertEqual(set(acoustic.component_heads), set())
        self.assertIsNotNone(syllable.syllable_head)
        self.assertIsInstance(acoustic.acoustic_head, torch.nn.Linear)
        self.assertFalse(any(isinstance(m, torch.nn.Dropout) for m in components.encoder.modules()))
        first = self.sample("a", "甲", [[1]])
        longer = self.sample("b", "乙丙", [[2], [1, 2]])
        h_single, _ = components.encode(collate_experiment([first]))
        h_mixed, mask = components.encode(collate_experiment([first, longer]))
        torch.testing.assert_close(h_single[0, 0], h_mixed[0, 0])
        self.assertEqual(mask.tolist(), [[True, False], [True, True]])

    def test_syllable_objective_uses_real_positions_including_class_zero(self):
        from phonetics.metrics import batch_objective, finalize_epoch
        from phonetics.data import collate_experiment
        batch = collate_experiment([self.sample("short", "甲", [[1]])])
        output = {"hidden": torch.zeros((1, 1, 128)), "mask": batch["char_mask"],
                  "syllable_logits": torch.tensor([[[5.0, 0.0]]])}
        stats = batch_objective(output, batch, torch.ones((1, 3)), "syllable")
        self.assertEqual(stats["characters"], 1)
        self.assertAlmostEqual(stats["syllable_ce_sum"].item(), math.log1p(math.exp(-5)), places=6)
        epoch = finalize_epoch({"characters": 1, "objective_sum": 0.0, "syllable_ce_sum": 0.0, "syllable_correct": 1})
        self.assertEqual(epoch["syllable_accuracy"], 1.0)

    def test_combined_objective_averages_component_heads_and_adds_lambda_cosine(self):
        from phonetics.metrics import batch_objective
        from phonetics.data import collate_experiment
        batch = collate_experiment([self.sample("short", "甲", [[1]])])
        logits = {name: torch.zeros((1, 1, size)) for name, size in
                  (("initials", 22), ("finals", 36), ("tones", 5))}
        outputs = {"hidden": torch.zeros((1, 1, 128)), "mask": batch["char_mask"],
                   "component_logits": logits,
                   "acoustic_prediction": torch.tensor([[[1.0, 0.0, 0.0]]])}
        targets = torch.tensor([[0.0, 1.0, 0.0]])
        one = batch_objective(outputs, batch, targets, "components-acoustic-lambda-1")
        ten = batch_objective(outputs, batch, targets, "components-acoustic-lambda-10")
        expected_component_mean = (math.log(22) + math.log(36) + math.log(5)) / 3
        self.assertAlmostEqual(one["component_ce_sum"].item(), expected_component_mean, places=6)
        self.assertAlmostEqual(one["cosine_sum"].item(), 1.0, places=7)
        self.assertAlmostEqual(one["weighted_cosine_sum"].item(), 1.0, places=6)
        self.assertAlmostEqual(one["objective"].item(), expected_component_mean + 1, places=6)
        self.assertAlmostEqual(ten["weighted_cosine_sum"].item(), 10.0, places=6)
        self.assertAlmostEqual(ten["objective"].item(), expected_component_mean + 10, places=6)
        from phonetics.objectives import combined_model_id
        for weight in (0, .25, 2.5, -0.5):
            with self.subTest(weight=weight):
                actual = batch_objective(outputs, batch, targets, combined_model_id(weight))
                self.assertAlmostEqual(actual["weighted_cosine_sum"].item(), weight, places=6)
                self.assertAlmostEqual(actual["objective"].item(), expected_component_mean + weight, places=6)


    def test_nonfinite_gradient_norm_is_detected(self):
        from phonetics.training import _gradient_norm
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([float("inf")])
        # Exercise the actual reducer against a minimal module carrying the gradient.
        class Holder(torch.nn.Module):
            def __init__(self, param):
                super().__init__(); self.param = param
        value, finite = _gradient_norm(Holder(parameter))
        self.assertFalse(finite)
        self.assertTrue(math.isinf(value))

    def test_five_seed_training_writes_best_last_and_complete_manifests(self):
        from project.tests.test_experiment_bundle import ExperimentBundleTests
        from phonetics.artifacts import sha256_file
        from phonetics.training import train_experiment
        fixture = ExperimentBundleTests(methodName="test_writer_loader_preserve_data_and_portability")
        fixture.setUp()
        try:
            result = train_experiment(fixture.bundle, model_id="syllable", runs_root=fixture.root / "runs",
                                      device="cpu", run_id="tiny-fixture",
                                      expected_manifest_sha256=sha256_file(fixture.bundle / "manifest.json"))
            self.assertEqual(result["manifest"]["status"], "complete")
            self.assertEqual(len(result["manifest"]["seed_results"]), 5)
            for seed in range(5):
                folder = result["run_dir"] / f"seed-{seed}"
                self.assertTrue((folder / "best.pt").is_file())
                self.assertTrue((folder / "last.pt").is_file())
                self.assertEqual(json.loads((folder / "summary.json").read_text())["status"], "complete")
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
