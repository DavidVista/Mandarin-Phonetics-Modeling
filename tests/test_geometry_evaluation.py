"""Exercise the configured evaluation stage on a sealed character experiment."""
import contextlib
import io
import json
import unittest
from pathlib import Path

import test_character_experiment as character_experiment
from phonetics.artifacts import read_jsonl, sha256_file
from phonetics.cli import main
from phonetics.evaluation.distribution import compute_retained_population
from phonetics.evaluation.geometry_bundle import validate_geometry_bundle
from phonetics.pipeline import load_config, run_stage


class GeometryEvaluationTests(unittest.TestCase):
    def test_configured_cli_filters_readings_and_preserves_context(self):
        fixture = character_experiment.CharacterExperimentTests()
        fixture.fixture_labels = ("a1", "gu1", "gu2", "ma1", "ma2")
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        experiment = fixture.build()
        config = json.loads((Path(__file__).parents[1] / "configs/experiment.example.json").read_text(encoding="utf-8"))
        config["evaluation"].update(
            experiment_root=experiment.name,
            experiment_manifest_sha256=sha256_file(experiment / "manifest.json"),
            output_root="evaluations", excluded_pinyin=["gu1", "qu1"],
            exclusion_confirmation={"run_id": "fixture-audit", "manifest_sha256": "a" * 64},
        )
        config_path = fixture.root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["prepare-evaluation", "--config", str(config_path)]), 0)
        target = fixture.root / "evaluations" / config["evaluation"]["id"]
        manifest = validate_geometry_bundle(target)
        self.assertEqual(manifest["identity"]["exclusion_confirmation"], config["evaluation"]["exclusion_confirmation"])
        inventory = manifest["identity"]["inventory"]
        self.assertEqual(inventory["eligible"], ["gu2", "ma1", "ma2"])
        self.assertEqual(inventory["zero_initial"], ["a1"])
        self.assertEqual(inventory["excluded_by_list"], ["gu1"])
        self.assertEqual(inventory["configured_but_unobserved"], ["qu1"])
        test_ids = set(json.loads((experiment / "splits.json").read_text())["test"])
        sentences = read_jsonl(target / "sentences.jsonl")
        self.assertEqual({row["sample_id"] for row in sentences}, test_ids)
        self.assertTrue(all(len(row["text"]) == 5 for row in sentences))
        occurrences = read_jsonl(target / "occurrences.jsonl")
        self.assertEqual(len(occurrences), 3 * len(test_ids))
        self.assertEqual({(row["character_position"], row["pinyin"]) for row in occurrences},
                         {(2, "gu2"), (3, "ma1"), (4, "ma2")})
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
