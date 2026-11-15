"""Exercise the configured evaluation stage on a sealed character experiment."""
import contextlib
import io
import json
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import test_character_experiment as character_experiment
from phonetics.artifacts import read_jsonl, sha256_file, write_json, write_jsonl
from phonetics.bundles import seal_bundle
from phonetics.cli import main
from phonetics.evaluation.distribution import compute_retained_population
from phonetics.evaluation.geometry_bundle import prepare_geometry_bundle, validate_geometry_bundle
from phonetics.execution import _run_pipeline_workflow, load_state, run_pipeline
from phonetics.notebooks import generate_encoder_notebooks
from phonetics.pipeline import load_config, run_stage
from phonetics.workflows import run_workflow, validate_workflow_result, workflow_pins


class GeometryEvaluationTests(unittest.TestCase):
    def test_research_reports_accept_current_acoustic_and_evaluation_bundles(self):
        from phonetics.visualization.core import VisualizationError, require_arial

        try:
            require_arial()
        except VisualizationError as error:
            self.skipTest(str(error))
        fixture = character_experiment.CharacterExperimentTests()
        fixture.fixture_labels = ("ba1", "ba2", "ma1", "ma2", "gu1", "qu1", "ju1", "xu1")
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        experiment = fixture.build()
        evaluation = prepare_geometry_bundle({
            "id": "geometry", "experiment_root": str(experiment),
            "experiment_manifest_sha256": sha256_file(experiment / "manifest.json"),
            "output_root": str(fixture.root / "evaluations"), "excluded_pinyin": [],
            "exclusion_source": {"run_id": "fixture", "manifest_sha256": "a" * 64},
        })
        root = Path(__file__).parents[1]
        config = load_config(root / "configs/local/experiment.bilstm.json")
        config["paths"].update(state=str(fixture.root / "state.json"), reports=str(fixture.root / "reports"))
        reference = next(job["parameters"] for job in config["analysis"] if job["workflow"] == "reference-geometry")
        for key, model, dimension in (("base_root", reference["base_model"], 768),
                                      ("zh_cn_root", config["model"], 1024)):
            target = fixture.root / key
            target.mkdir()
            labels = sorted(fixture.fixture_labels)
            write_json(target / "labels.json", labels)
            write_jsonl(target / "inventory.jsonl", [{"label": label, "sha256": "a" * 64} for label in labels])
            write_json(target / "inference.json", {"fixture": True})
            vectors = np.random.default_rng(dimension).normal(size=(len(labels), dimension)).astype(np.float32)
            np.save(target / "targets.npy", vectors, allow_pickle=False)
            seal_bundle(target, "acoustic_targets", {"model": model, "backend": "transformers",
                                                      "inventory_manifest_sha256": "a" * 64})
            reference[key] = str(target)
        state = load_state(config)
        for key, path in (("prepare-experiment", experiment), ("prepare-evaluation", evaluation),
                          ("prepare-embeddings", fixture.root / "zh_cn_root"),
                          ("reference-acoustic:bilstm-reference-geometry", fixture.root / "base_root")):
            state["completed"][key] = {"path": str(path), "manifest_sha256": sha256_file(path / "manifest.json")}
        for workflow in config["analysis"]:
            name = workflow["workflow"]
            if name == "training-report":
                continue
            with self.subTest(workflow=name):
                params = workflow["parameters"]
                if "evaluation_root" in params:
                    params["evaluation_root"] = str(evaluation)
                if name == "dimsim-parser-alignment":
                    params["experiment_root"] = str(experiment)
                    params["bootstrap_replicates"] = 8
                pins = workflow_pins(workflow)
                self.assertIsNone(_run_pipeline_workflow(config, state, "analysis", workflow))
                result = Path(state["completed"]["analysis:" + workflow["id"]]["path"])
                self.assertEqual(validate_workflow_result(result)["identity"]["input_pins"], pins)
                self.assertIsNone(_run_pipeline_workflow(config, state, "analysis", workflow))
                key = next(iter(pins))
                with self.assertRaisesRegex(ValueError, "differ from completed pipeline stages"):
                    run_workflow(workflow, fixture.root / "reports", expected_input_pins={**pins, key: "0" * 64})

    def test_run_uses_unchanged_config_and_pins_reports_and_notebook(self):
        for mode in ("local", "notebook"):
            with self.subTest(mode=mode):
                self.check_configured_run(mode)

    def check_configured_run(self, mode):
        recipe = Path(__file__).parents[1] / "configs/experiment.mock.json"
        config = json.loads(recipe.read_text(encoding="utf-8"))
        fixture = character_experiment.CharacterExperimentTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        config["paths"] = {key: key for key in config["paths"]}
        config["paths"]["prepared"] = "mock-prepared"
        config["experiment"].update(sample_size=20, test_size=10, model_ids=["components"])
        config["analysis"][0]["parameters"] = {}
        config["evaluation"] = {
            "id": "geometry", "experiment_root": "experiments/mock", "experiment_manifest_sha256": None,
            "output_root": "evaluation", "excluded_pinyin": [],
            "exclusion_source": {"run_id": "fixture", "manifest_sha256": "a" * 64},
        }
        config["encoder_geometry"] = {
            "id": "encoder", "representations_root": "hidden-states", "device": "cpu",
            "batch_size": 8, "tolerance": 1e-12, "allow_extraction": True, "execution": mode,
        }
        path = fixture.root / "recipe.json"
        write_json(path, config)
        original = path.read_bytes()

        def train(experiment, *, model_id, runs_root, device, run_id, expected_manifest_sha256):
            run = Path(runs_root) / "mock" / model_id / run_id
            seed = run / "seed-0"
            seed.mkdir(parents=True)
            for filename in ("best.pt", "last.pt", "shared_encoder.pt", "summary.json", "history.json", "events.jsonl"):
                (seed / filename).write_bytes(b"fixture")
            write_json(seed / "history.json", {"epochs": [
                {"epoch": 1, "train": {"objective": 0.6},
                 "validation": {"objective": 0.5, "joint_ift_accuracy": 0.4, "component_ce": 0.5}},
            ]})
            write_json(run / "summary.json", {"status": "complete", "seed_results": [
                {"seed": 0, "best_epoch": 1, "test_metrics": {"objective": 0.5}},
            ]})
            write_json(run / "manifest.json", {
                "run_type": "training_run", "status": "complete", "model_id": model_id, "run_id": run_id,
                "experiment_id": "mock", "seeds": [0], "seed_status": {"0": "complete"},
                "experiment_hashes": {"manifest.json": expected_manifest_sha256},
                "artifacts": {file.relative_to(run).as_posix(): sha256_file(file)
                              for file in run.rglob("*") if file.is_file()},
            })
            return {"run_dir": run}

        def report(*, refs, pins, spec, reports_root, source_sha256, allow_execution):
            self.assertTrue(allow_execution)
            self.assertEqual({key: sha256_file(Path(root) / "manifest.json") for key, root in refs.items()}, pins)
            target = Path(reports_root) / spec["id"]
            target.mkdir()
            write_json(target / "report.json", {"inputs": pins})
            seal_bundle(target, "encoder_geometry_report", {"inputs": pins})
            return target

        with (patch.dict(sys.modules, {"phonetics.training": SimpleNamespace(train_experiment=train)}),
              patch("phonetics.encoder_jobs.run_encoder_workflow", side_effect=report) as geometry):
            loaded = load_config(path)
            self.assertEqual(run_pipeline(loaded, through="training", allow_training=True)["status"], "complete")
            self.assertFalse((fixture.root / "evaluation").exists())
            self.assertTrue((fixture.root / "reports/training-report/learning-curves.png").is_file())
            self.assertTrue((fixture.root / "reports/training-report/validation-accuracy.png").is_file())
            result = run_pipeline(loaded)
            self.assertEqual(result["status"], "complete" if mode == "local" else "notebook-ready")
            loaded = load_config(path)
            self.assertEqual(run_pipeline(loaded), result)
            self.assertEqual(geometry.call_count, int(mode == "local"))
        self.assertEqual(path.read_bytes(), original)
        self.assertTrue((fixture.root / "reports/training-report/report.md").is_file())
        notebook, = generate_encoder_notebooks(loaded)
        if mode == "notebook":
            self.assertEqual(notebook["notebook"], result["notebook"])
        cells = json.loads(Path(notebook["notebook"]).read_text(encoding="utf-8"))["cells"]
        settings = "".join(cells[-2]["source"])
        completed = load_state(loaded)["completed"]
        for key in ("prepare-experiment", "prepare-evaluation", "train:components"):
            self.assertIn(completed[key]["manifest_sha256"], settings)
        self.assertIn("ALLOW_EXTRACTION = True", settings)
        self.assertIn("run_encoder_workflow", "".join(cells[-1]["source"]))

        occurrences = Path(completed["prepare-evaluation"]["path"]) / "occurrences.jsonl"
        content = occurrences.read_bytes()
        occurrences.write_bytes(content + b"\n")
        with self.assertRaisesRegex(ValueError, "artifact checksum mismatch"):
            run_pipeline(loaded)
        occurrences.write_bytes(content)

        manifest = Path(completed["train:components"]["path"]) / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            generate_encoder_notebooks(loaded)
        with self.assertRaisesRegex(ValueError, "completed stage manifest changed"):
            run_pipeline(loaded)

    def test_configured_cli_keeps_nonzero_readings_and_preserves_context(self):
        fixture = character_experiment.CharacterExperimentTests()
        fixture.fixture_labels = ("a1", "gu1", "gu2", "ma1", "ma2")
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        experiment = fixture.build()
        config = json.loads((Path(__file__).parents[1] / "configs/notebook/experiment.bilstm.json").read_text(encoding="utf-8"))
        config.pop("encoder_geometry")
        config["evaluation"] = {
            "id": "fixture-geometry", "experiment_root": experiment.name,
            "experiment_manifest_sha256": sha256_file(experiment / "manifest.json"),
            "output_root": "evaluations", "excluded_pinyin": ["gu1", "qu1"],
            "exclusion_source": {"run_id": "fixture-source", "manifest_sha256": "b" * 64},
            "exclusion_confirmation": {"run_id": "fixture-audit", "manifest_sha256": "a" * 64},
        }
        config_path = fixture.root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["prepare-evaluation", "--config", str(config_path)]), 0)
        target = fixture.root / "evaluations" / config["evaluation"]["id"]
        manifest = validate_geometry_bundle(target)
        self.assertEqual(manifest["identity"]["exclusion_confirmation"], config["evaluation"]["exclusion_confirmation"])
        inventory = manifest["identity"]["inventory"]
        self.assertEqual(inventory["eligible"], ["gu1", "gu2", "ma1", "ma2"])
        self.assertEqual(inventory["zero_initial"], ["a1"])
        self.assertEqual(inventory["excluded_by_list"], ["gu1"])
        self.assertEqual(inventory["configured_but_unobserved"], ["qu1"])
        test_ids = set(json.loads((experiment / "splits.json").read_text())["test"])
        sentences = read_jsonl(target / "sentences.jsonl")
        self.assertEqual({row["sample_id"] for row in sentences}, test_ids)
        self.assertTrue(all(len(row["text"]) == 5 for row in sentences))
        occurrences = read_jsonl(target / "occurrences.jsonl")
        self.assertEqual(len(occurrences), 4 * len(test_ids))
        self.assertEqual({(row["character_position"], row["pinyin"]) for row in occurrences},
                         {(1, "gu1"), (2, "gu2"), (3, "ma1"), (4, "ma2")})
        self.assertEqual(manifest["identity"]["occurrence_counts"]["zero_initial"], len(test_ids))
        self.assertEqual(manifest["identity"]["syllable_pair_count"], 6)
        digest = sha256_file(target / "manifest.json")
        _, validation = compute_retained_population(target, expected_manifest_sha256=digest)
        self.assertEqual(validation["pair_count"], 3)
        self.assertEqual(run_stage(load_config(config_path), "prepare-evaluation"), target)
        self.assertEqual(sha256_file(target / "manifest.json"), digest)
        rebuild_config = load_config(config_path)
        rebuild_config["evaluation"]["output_root"] = str(fixture.root / "rebuild")
        rebuilt = run_stage(rebuild_config, "prepare-evaluation")
        self.assertEqual(sha256_file(rebuilt / "manifest.json"), digest)

        config["evaluation"]["excluded_pinyin"] = ["gu2"]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "use a new output ID"):
            run_stage(load_config(config_path), "prepare-evaluation")
        self.assertEqual(sha256_file(target / "manifest.json"), digest)

        config["evaluation"]["experiment_manifest_sha256"] = "0" * 64
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "experiment manifest SHA256 mismatch"):
            run_stage(load_config(config_path), "prepare-evaluation")
