"""CPU-only encoder geometry contract and numerical regression checks."""
import sys
import unittest

import numpy as np

import test_character_experiment as character_experiment
from phonetics.artifacts import sha256_file, write_json
from phonetics.bundles import seal_bundle, validate_bundle
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
        from phonetics.visualization.core import VisualizationError, require_arial
        try:
            require_arial()
        except VisualizationError as error:
            self.skipTest(str(error))
        fixture = character_experiment.CharacterExperimentTests()
        fixture.fixture_labels = ("ba1", "ba2", "ma1", "ma2")
        fixture.fixture_suffix = "妈麻马"
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.config["seed_count"] = len(TRAINING_SEEDS)
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
        rng = np.random.default_rng(21)
        for model in TRAINING_MODELS:
            root = fixture.root / "exports" / model
            exports[model] = root
            checkpoints[model] = {}
            pins[model] = "b" * 64
            identity = {"model": model, "evaluation_manifest_sha256": evaluation_pin,
                        "experiment_manifest_sha256": experiment_pin, "training_manifest_sha256": pins[model],
                        "seeds": list(TRAINING_SEEDS)}
            for seed in TRAINING_SEEDS:
                child = root / f"seed-{seed}"
                child.mkdir(parents=True)
                write_json(child / "occurrence_ids.json", positions)
                np.save(child / "hidden.npy", rng.normal(size=(len(positions), 3)).astype(np.float32))
                np.save(child / "unknown.npy", unknown[seed])
                checkpoints[model][seed] = {"seed_input_hashes": seed_hashes[seed], "best_epoch": 1,
                                            "checkpoint_sha256": "a" * 64}
                seal_bundle(child, SEED_TYPE, {**identity, **checkpoints[model][seed], "seed": seed,
                                               "representation": "raw pre-head"})
            seal_bundle(root, EXTRACTION_TYPE, identity)
        arguments = {"evaluation_root": evaluation, "experiment_root": experiment, "pins": pins,
                     "exports": exports, "checkpoints": checkpoints, "output": fixture.root / "report",
                     "tolerance": 1e-12, "source_sha256": "c" * 64}
        for field, wrong in (("best_epoch", 2), ("checkpoint_sha256", "0" * 64)):
            selected = checkpoints[TRAINING_MODELS[0]][TRAINING_SEEDS[0]]
            original = selected[field]
            selected[field] = wrong
            with self.assertRaisesRegex(ValueError, "checkpoint provenance mismatch"):
                run_encoder_geometry(**arguments)
            selected[field] = original
        result = run_encoder_geometry(**arguments)
        validate_bundle(result, "encoder_geometry_report")
        self.assertEqual(len(list((result / "views").glob("*.js"))), 6)
        self.assertEqual(len(list((result / "figures").glob("*.png"))), 5)
        self.assertIn("planned_triples", (result / "tables/controlled_match.csv").read_text())
        self.assertIn("valid_full_matches", (result / "tables/coverage.csv").read_text())
        self.assertIn("Component mismatch count", (result / "tables/pairwise_spearman.csv").read_text())
        self.assertIn("model_ties", (result / "tables/same_mask_agreement.csv").read_text())
        self.assertIn("components-acoustic-lambda-1", (result / "tables/paired_differences.csv").read_text())
        self.assertNotIn("torch", sys.modules)
