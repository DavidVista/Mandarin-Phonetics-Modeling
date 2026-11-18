"""CPU-only encoder geometry contract and numerical regression checks."""
import sys
import subprocess
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import test_character_experiment as character_experiment
from phonetics.artifacts import sha256_file, write_json
from phonetics.bundles import seal_bundle, validate_bundle
from phonetics.encoder_jobs import run_configured_report, run_encoder_workflow
from phonetics.experiment import SELECTION_RULE
from phonetics.notebooks import generate_encoder_notebooks
from phonetics.evaluation.encoder_extraction import (
    EXTRACTION_TYPE, SEED_TYPE, expected_unknown_positions, extract_hidden_states, load_positions,
)
from phonetics.evaluation.geometry_bundle import prepare_geometry_bundle
from phonetics.evaluation.reference_geometry import evaluate_geometry
from phonetics.evaluation.training_summary import TRAINING_MODELS, TRAINING_SEEDS
from phonetics.research.encoder_geometry import contextual_distances, run_encoder_geometry


class EncoderGeometryTests(unittest.TestCase):
    def test_weighted_distances_match_explicit_distinct_occurrence_pairs(self):
        rows = [{"pinyin": a, "character": g} for a, g in
                (("a", "甲"), ("a", "甲"), ("a", "乙"), ("b", "丙"))]
        hidden = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        result = contextual_distances(["a", "b"], rows, hidden)
        weights = [.25, .25, .5]
        diagonal = sum(weights[i]*weights[j]*(1-hidden[i]@hidden[j])
                       for i in range(3) for j in range(3) if i != j)
        diagonal /= 1 - sum(w*w for w in weights)
        self.assertAlmostEqual(result[0, 0], diagonal)
        self.assertAlmostEqual(result[0, 1], sum(weights[i]*(1-hidden[i]@hidden[3]) for i in range(3)))
        self.assertTrue(np.isnan(result[1, 1]))
        tables, _ = evaluate_geometry(["a", "b"], [("b", "a", "1"), ("b", "a", "2")],
                                      np.stack([[[0., 1.], [1., 0.]], result]),
                                      tolerance=1e-12, systems=("DIMSIM", "encoder"), contextual=True)
        overall = next(row for row in tables["controlled_match"]
                       if row["system"] == "encoder" and row["from_mask"] == "All")
        self.assertEqual((overall["planned_triples"], overall["triples"]), (2, 1))

    def test_guard_does_not_import_torch(self):
        with self.assertRaises(PermissionError):
            extract_hidden_states(evaluation_root=None, evaluation_pin=None, experiment_root=None,
                                  experiment_pin=None, training_root=None, training_pin=None,
                                  model=None, output=None, device=None, batch_size=None, source_sha256=None)
        self.assertNotIn("torch", sys.modules)

    def test_report_reads_all_seeds_and_exports_figures(self):
        self.check_report(TRAINING_MODELS, TRAINING_SEEDS)

    def test_report_accepts_lambda_three_and_three_seeds(self):
        models = (*TRAINING_MODELS, "components-acoustic-lambda-3")
        self.check_report(models, (0, 1, 2), contrasts=[["components", models[-1]]])

    def test_report_accepts_extension_checkpoint_from_the_original_experiment(self):
        extension = "syllable-acoustic-lambda-1"
        self.check_report((*TRAINING_MODELS, extension), (0, 1, 2),
                          contrasts=[["components-acoustic-lambda-1", extension]], extension=extension)

    def check_report(self, models, seeds, contrasts=None, extension=None):
        from phonetics.visualization.core import VisualizationError, load_protocol, require_font
        try:
            require_font(load_protocol().style["font"]["family"])
        except VisualizationError as error:
            self.skipTest(str(error))
        fixture = character_experiment.CharacterExperimentTests()
        fixture.fixture_labels = ("ba1", "ba2", "ma1", "ma2")
        fixture.fixture_suffix = "妈麻马"
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.config["seed_count"] = len(seeds)
        fixture.config["model_ids"] = [model for model in models if model != extension]
        experiment = fixture.build()
        experiment_pin = sha256_file(experiment / "manifest.json")
        evaluation = prepare_geometry_bundle({
            "id": "geometry", "experiment_root": str(experiment), "experiment_manifest_sha256": experiment_pin,
            "output_root": str(fixture.root / "evaluations"), "excluded_pinyin": ["ma2"],
            "exclusion_source": {"run_id": "fixture", "manifest_sha256": "a" * 64},
        })
        evaluation_pin = sha256_file(evaluation / "manifest.json")
        manifest, occurrences = load_positions(evaluation, evaluation_pin)
        positions = [row["occurrence_id"] for row in occurrences]
        test_pin = manifest["identity"]["test_ids_sha256"]
        unknown, seed_hashes = expected_unknown_positions(experiment, experiment_pin, occurrences, test_pin)
        pins = {"experiment": experiment_pin, "evaluation": evaluation_pin}
        exports = {}
        checkpoints = {}
        runs = {}
        rng = np.random.default_rng(21)
        for model in models:
            run = fixture.root / "runs" / model
            runs[model] = str(run)
            artifacts = {}
            for seed in seeds:
                seed_dir = run / f"seed-{seed}"
                seed_dir.mkdir(parents=True)
                metrics = {"objective": 0.1}
                write_json(seed_dir / "summary.json", {
                    "seed": seed, "status": "complete", "best_epoch": 1,
                    "best_validation_metrics": metrics, "best_validation_objective": 0.1,
                    "seed_input_hashes": seed_hashes[seed],
                })
                write_json(seed_dir / "history.json", {"epochs": [{"epoch": 1, "validation": metrics}]})
                (seed_dir / "best.pt").write_bytes(b"opaque checkpoint fixture")
                for path in seed_dir.iterdir():
                    artifacts[path.relative_to(run).as_posix()] = sha256_file(path)
            model_source = Path(__file__).resolve().parents[1] / "src/phonetics/model.py"
            write_json(run / "manifest.json", {
                "schema_version": 2 if model == models[-1] else 1,
                "run_type": "training_run", "status": "complete",
                "model_id": model, "seeds": list(seeds), "config": fixture.config,
                "seed_status": {str(seed): "complete" for seed in seeds},
                "selection": SELECTION_RULE, "experiment_hashes": {"manifest.json": experiment_pin},
                "code_sha256": {"model.py": (
                    "4c9df43b451155582e1df117c0a5edaa60284a7b054fe34fad5cf75616ce6842"
                    if extension and model != extension else sha256_file(model_source))}, "artifacts": artifacts,
            })
            root = fixture.root / "exports" / model
            exports[model] = root
            checkpoints[model] = {}
            pins[model] = sha256_file(run / "manifest.json")
            identity = {"model": model, "evaluation_manifest_sha256": evaluation_pin,
                        "experiment_manifest_sha256": experiment_pin, "training_manifest_sha256": pins[model],
                        "seeds": list(seeds)}
            for seed in seeds:
                child = root / f"seed-{seed}"
                child.mkdir(parents=True)
                write_json(child / "occurrence_ids.json", positions)
                np.save(child / "hidden.npy", rng.normal(size=(len(positions), 3)).astype(np.float32))
                np.save(child / "unknown.npy", unknown[seed])
                checkpoints[model][seed] = {"seed_input_hashes": seed_hashes[seed], "best_epoch": 1,
                                            "checkpoint_sha256": artifacts[f"seed-{seed}/best.pt"]}
                seal_bundle(child, SEED_TYPE, {**identity, **checkpoints[model][seed], "seed": seed,
                                               "representation": "raw pre-head"})
            seal_bundle(root, EXTRACTION_TYPE, identity)
        arguments = {"evaluation_root": evaluation, "experiment_root": experiment, "pins": pins,
                     "exports": exports, "checkpoints": checkpoints, "output": fixture.root / "report",
                     "tolerance": 1e-12, "source_sha256": "c" * 64, "contrasts": contrasts}
        selected_seeds = checkpoints[models[0]]
        last_seed = seeds[-1]
        selected = selected_seeds.pop(last_seed)
        with self.assertRaisesRegex(ValueError, "checkpoint seeds"):
            run_encoder_geometry(**arguments)
        selected_seeds[last_seed] = selected
        export = exports[models[0]]
        child = export / f"seed-{seeds[-1]}"
        export_identity = validate_bundle(export, EXTRACTION_TYPE)["identity"]
        child_identity = validate_bundle(child, SEED_TYPE)["identity"]
        write_json(child / "occurrence_ids.json", list(reversed(positions)))
        seal_bundle(child, SEED_TYPE, child_identity)
        seal_bundle(export, EXTRACTION_TYPE, export_identity)
        with patch("phonetics.research.encoder_geometry.contextual_distances") as calculate:
            with self.assertRaisesRegex(ValueError, "occurrence alignment"):
                run_encoder_geometry(**arguments)
            calculate.assert_not_called()
        self.assertFalse(arguments["output"].exists())
        if contrasts:
            refs = {"experiment": str(experiment), "evaluation": str(evaluation), **runs}
            spec = {"id": "report", "representations_root": str(fixture.root / "exports")}
            missing_export = exports[models[-1]]
            moved = missing_export.rename(fixture.root / "moved-export")
            try:
                with patch("phonetics.encoder_jobs.extract_hidden_states") as extract:
                    with self.assertRaisesRegex(ValueError, "occurrence alignment"):
                        run_encoder_workflow(refs=refs, pins=pins, spec=spec, reports_root=fixture.root,
                                             source_sha256="c" * 64, allow_execution=True)
                    extract.assert_not_called()
            finally:
                moved.rename(missing_export)
        write_json(child / "occurrence_ids.json", positions)
        seal_bundle(child, SEED_TYPE, child_identity)
        seal_bundle(export, EXTRACTION_TYPE, export_identity)
        for field, wrong in (("best_epoch", 2), ("checkpoint_sha256", "0" * 64)):
            selected = checkpoints[models[0]][seeds[0]]
            original = selected[field]
            selected[field] = wrong
            with self.assertRaisesRegex(ValueError, "checkpoint provenance mismatch"):
                run_encoder_geometry(**arguments)
            selected[field] = original
        if contrasts:
            refs = {"experiment": str(experiment), "evaluation": str(evaluation), **runs}
            spec = {"id": "report", "training_runs": runs, "contrasts": contrasts,
                    "representations_root": str(fixture.root / "exports"), "device": "cpu",
                    "batch_size": 8, "tolerance": 1e-12, "allow_extraction": True}
            config = {"encoder_geometry": spec, "paths": {"generated": str(fixture.root / "generated")},
                      "notebooks": {"install_dependencies": False},
                      "evaluation": {"experiment_root": str(experiment), "id": evaluation.name,
                                     "output_root": str(evaluation.parent),
                                     "experiment_manifest_sha256": experiment_pin}}
            notebook = generate_encoder_notebooks(config)[0]["notebook"]
            cells = json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]
            self.assertIn("contrasts", "".join(cells[-2]["source"]))
            self.assertIn("ALLOW_EXTRACTION = True", "".join(cells[-2]["source"]))
            self.assertNotIn("TRAINING_MODELS", "".join(cells[-1]["source"]))
            with self.assertRaisesRegex(ValueError, "one training run"):
                run_encoder_workflow(refs={**refs, "extra": str(experiment)}, pins=pins, spec=spec,
                                     reports_root=fixture.root, source_sha256="c" * 64)
            result = run_encoder_workflow(refs=refs, pins=pins, spec=spec, reports_root=fixture.root,
                                          source_sha256="c" * 64, mount=fixture.root)
            if extension:
                config["paths"]["reports"] = str(fixture.root / "cli")
                cli_result = run_configured_report(config)
                notebook_root = fixture.root / "notebook"
                code = "\n".join("".join(cells[i]["source"]) for i in (2, 3)) + (
                    f"\nRESULTS_ROOT = Path({str(notebook_root)!r})\n"
                    f"INPUT_ROOT = Path({str(fixture.root)!r})\n"
                    "ALLOW_EXTRACTION = False\n" + "".join(cells[4]["source"])
                )
                subprocess.run([sys.executable, "-X", "utf8", "-"], input=code.encode(),
                               cwd=fixture.root, check=True, capture_output=True)
                notebook_result = notebook_root / "reports" / spec["id"]
                for name in ("same_mask_breakdown", "shared_character_residual", "shared_character_matched_agreement"):
                    relative = Path("tables") / f"{name}_across_seeds.csv"
                    self.assertEqual((notebook_result / relative).read_bytes(), (cli_result / relative).read_bytes())
        else:
            result = run_encoder_geometry(**arguments)
        validate_bundle(result, "encoder_geometry_report")
        self.assertEqual(len(list((result / "views").glob("*.js"))), len(seeds) + 1)
        self.assertEqual(len(list((result / "figures").glob("*.png"))), len(models) + 1)
        figure = result / "figures" / "residual-histograms-by-mask"
        self.assertTrue(figure.with_suffix(".png").read_bytes().startswith(b"\x89PNG"))
        self.assertTrue(figure.with_suffix(".eps").read_bytes().startswith(b"%!PS-Adobe"))
        metadata = json.loads(figure.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["scientific_provenance"]["tail_cutoff_ties"], "include")
        order = metadata["resolved_data"]["models"]
        self.assertEqual(set(order), set(models))
        if "components-acoustic-lambda-3" in models:
            self.assertLess(order.index("components-acoustic-lambda-3"), order.index("components-acoustic-lambda-10"))
        self.assertIn("planned_triples", (result / "tables/controlled_match.csv").read_text())
        self.assertIn("valid_full_matches", (result / "tables/coverage.csv").read_text())
        self.assertIn("Component mismatch count", (result / "tables/pairwise_spearman.csv").read_text())
        self.assertIn("model_ties", (result / "tables/same_mask_agreement.csv").read_text())
        page = (result / "encoder_geometry.html").read_text(encoding="utf-8")
        for name in ("shared_character_residual", "shared_character_matched_agreement", "same_mask_breakdown"):
            self.assertIn(f"tables/{name}_across_seeds.csv", page)
        self.assertIn("BiLSTM evaluation report", page)
        paired_model = contrasts[0][1] if contrasts else "components-acoustic-lambda-1"
        self.assertIn(paired_model, (result / "tables/paired_differences.csv").read_text())
        self.assertNotIn("torch", sys.modules)
