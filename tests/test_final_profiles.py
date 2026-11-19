"""Full-run profile contracts and state-pinned pretrained evaluation handoff."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from phonetics.artifacts import sha256_file, write_json
from phonetics.bundles import seal_bundle
from phonetics.encoder_jobs import encoder_inputs
from phonetics.notebooks import generate_encoder_notebooks
from phonetics.pipeline import load_config

PROJECT = Path(__file__).resolve().parents[1]


class FinalProfileTests(unittest.TestCase):
    def test_profiles_share_scientific_protocol_and_separate_execution_outputs(self):
        for architecture, model_count, seed_count in (("bilstm", 7, 5), ("bert", 3, 3)):
            profiles = [load_config(PROJECT / "configs" / mode / f"experiment.{architecture}.json")
                        for mode in ("local", "notebook")]
            self.assertEqual(profiles[0]["experiment"], profiles[1]["experiment"])
            for config in profiles:
                experiment = config["experiment"]
                self.assertEqual(len(experiment["model_ids"]), model_count)
                self.assertIn("syllable", experiment["model_ids"])
                self.assertEqual(experiment["seed_count"], seed_count)
                self.assertEqual((experiment["sample_size"], experiment["test_size"]), (100000, 100000))
                self.assertEqual(config["acoustic"], {"existing_bundle": None, "manifest_sha256": None})
                self.assertTrue(config["encoder_geometry"]["diagnostics"])
                self.assertIsNone(config["evaluation"]["experiment_manifest_sha256"])
                self.assertEqual(set(config["encoder_geometry"]["training_runs"]), set(experiment["model_ids"]))
                self.assertNotIn("pretrained-bert", experiment["model_ids"])
                if architecture == "bilstm":
                    self.assertIn("syllable-acoustic-lambda-1", experiment["model_ids"])
                    self.assertEqual(experiment["training"]["epochs"], 20)
                    self.assertEqual(experiment["training"]["epochs_by_model"], {"components": 10, "acoustic": 60})
                else:
                    self.assertEqual(experiment["training"]["epochs"], 4)
                    self.assertEqual(config["encoder_geometry"]["representation_sources"],
                                     {"pretrained-bert": {"type": "pretrained_bert"}})
            for key in ("state", "generated", "runs", "reports"):
                self.assertNotEqual(profiles[0]["paths"][key], profiles[1]["paths"][key])

    def test_managed_pretrained_uses_completed_experiment_and_training_pins(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = load_config(PROJECT / "configs/notebook/experiment.bert.json")
            experiment = root / "experiment"
            experiment.mkdir()
            write_json(experiment / "config.json", config["experiment"])
            seal_bundle(experiment, "character_experiment", {
                "config": config["experiment"], "training_seeds": [0, 1, 2]})
            config["evaluation"].update(experiment_root=str(experiment), output_root=str(root / "evaluation"))
            evaluation = root / "evaluation" / config["evaluation"]["id"]
            evaluation.mkdir(parents=True)
            write_json(evaluation / "manifest.json", {"fixture": "evaluation"})
            state = {"completed": {}}
            def completed(key, path):
                state["completed"][key] = {"path": str(path), "manifest_sha256": sha256_file(path / "manifest.json")}
            completed("prepare-experiment", experiment)
            completed("prepare-evaluation", evaluation)
            runs = {}
            for model in config["experiment"]["model_ids"]:
                path = root / "runs" / model
                path.mkdir(parents=True)
                write_json(path / "manifest.json", {"fixture_model": model})
                runs[model] = str(path)
                completed("train:" + model, path)
            config["encoder_geometry"]["training_runs"] = runs
            config["paths"]["generated"] = str(root / "generated")
            refs, pins = encoder_inputs(config, state)
            self.assertEqual(refs["pretrained-bert"], refs["experiment"])
            self.assertEqual(pins["pretrained-bert"], state["completed"]["prepare-experiment"]["manifest_sha256"])
            self.assertNotIn("train:pretrained-bert", state["completed"])
            notebook, = generate_encoder_notebooks(config, state=state)
            document = json.loads(Path(notebook["notebook"]).read_text(encoding="utf-8"))
            self.assertEqual(document["metadata"]["input_manifest_sha256"], pins)
            self.assertIn("pretrained-bert", "".join(document["cells"][3]["source"]))
            incomplete = copy.deepcopy(state)
            del incomplete["completed"]["train:syllable"]
            with self.assertRaisesRegex(ValueError, "train:syllable must finish"):
                encoder_inputs(config, incomplete)
            changed = copy.deepcopy(state)
            changed["completed"]["prepare-experiment"]["manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                encoder_inputs(config, changed)
            write_json(evaluation / "manifest.json", {"fixture": "modified evaluation"})
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                encoder_inputs(config, state)


if __name__ == "__main__":
    unittest.main()
