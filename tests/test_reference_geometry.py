"""Check shared reference comparisons against an exhaustive small population."""
import itertools
import json
import pickle
from pathlib import Path
import unittest

import numpy as np

import test_character_experiment as character_experiment
from phonetics.artifacts import sha256_file, write_json, write_jsonl
from phonetics.evaluation.geometry_bundle import prepare_geometry_bundle
from phonetics.evaluation.graduality_io import EXTRACTION
from phonetics.research.reference_geometry import _MODELS
from phonetics.workflows import run_workflow, validate_workflow_result
from phonetics.evaluation.reference_geometry import SYSTEMS, SLICES, evaluate_geometry


class ReferenceGeometryTests(unittest.TestCase):
    def test_controlled_and_agreement_match_direct_triple_enumeration(self):
        components = list(itertools.product("abc", "xy", "12"))
        labels = ["".join(parts) for parts in components]
        n = len(labels)
        rng = np.random.default_rng(24)
        distances = rng.integers(0, 5, size=(3, n, n))
        distances = np.triu(distances, 1) + np.triu(distances, 1).transpose(0, 2, 1)
        tables, maps = evaluate_geometry(labels, components, distances, tolerance=1e-12)
        controlled = [[] for _ in SYSTEMS]
        agreed = {name: [[] for _ in SYSTEMS] for name in SLICES}
        ties = dict.fromkeys(SLICES, 0)
        for anchor in range(n):
            a = components[anchor]
            for left, right in itertools.combinations(range(n), 2):
                l, r = components[left], components[right]
                lm = {i for i in range(3) if a[i] == l[i]}
                rm = {i for i in range(3) if a[i] == r[i]}
                changes = {i for i in range(3) if l[i] != r[i]}
                if lm < rm and changes == rm - lm:
                    category, near, far = SLICES[0], right, left
                elif rm < lm and changes == lm - rm:
                    category, near, far = SLICES[0], left, right
                elif lm == rm and anchor not in (left, right):
                    category, near, far = SLICES[2], left, right
                elif not lm <= rm and not rm <= lm and anchor not in (left, right):
                    category, near, far = SLICES[1], left, right
                else:
                    continue
                differences = distances[:, anchor, far] - distances[:, anchor, near]
                for model, difference in enumerate(differences):
                    if category == SLICES[0]:
                        controlled[model].append(difference)
                    if differences[0] != 0:
                        agreed[category][model].append((1 + np.sign(difference) * np.sign(differences[0])) / 2)
                ties[category] += differences[0] == 0
        self.assertEqual(sum(maps["controlled_cells"][:, 3]), len(controlled[0]))
        for model, system in enumerate(SYSTEMS):
            row = next(row for row in tables["controlled_match"] if row["system"] == system and row["from_mask"] == "All")
            self.assertEqual(row["triples"], len(controlled[model]))
            self.assertAlmostEqual(row["mean_effect"], np.mean(controlled[model]))
            self.assertAlmostEqual(row["win_rate"], np.mean((1 + np.sign(controlled[model])) / 2))
            for category in SLICES:
                row = next(row for row in tables["dimsim_agreement"] if row["system"] == system and row["slice"] == category)
                expected = agreed[category][model]
                self.assertEqual(row["dimsim_ties"], ties[category])
                self.assertEqual(row["comparisons"], len(expected))
                if expected:
                    self.assertAlmostEqual(row["agreement"], np.mean(expected))
                else:
                    self.assertIsNone(row["agreement"])
        for index, category in enumerate(SLICES[1:]):
            self.assertEqual(maps["agreement_counts"][index].sum(), 2 * len(agreed[category][0]))

    def test_workflow_publishes_sealed_tables_and_rendered_artifacts(self):
        from phonetics.visualization.core import VisualizationError, require_arial

        try:
            require_arial()
        except VisualizationError as error:
            self.skipTest(str(error))
        fixture = character_experiment.CharacterExperimentTests()
        fixture.fixture_labels = ("ba1", "ba2", "ma1", "ma2")
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        experiment = fixture.build()
        evaluation = prepare_geometry_bundle({
            "id": "geometry", "experiment_root": str(experiment),
            "experiment_manifest_sha256": sha256_file(experiment / "manifest.json"),
            "output_root": str(fixture.root / "evaluations"), "excluded_pinyin": [],
            "exclusion_source": {"run_id": "fixture", "manifest_sha256": "a" * 64},
        })
        params = {"evaluation_root": str(evaluation), "tolerance": 1e-12}
        for index, (key, identifier) in enumerate(_MODELS.items()):
            root = fixture.root / key
            (root / "reports").mkdir(parents=True)
            model = {"id": identifier, **EXTRACTION}
            write_json(root / "config.json", {"model": model})
            write_json(root / "reports/inference.json", {"errors": 0})
            write_jsonl(root / "sound_inventory.jsonl",
                        ({"label": label, "sha256": "b" * 64} for label in fixture.fixture_labels))
            vectors = {label: np.array([1., position + index, (position + 1) ** 2])
                       for position, label in enumerate(fixture.fixture_labels)}
            (root / "pinyin_embeddings.pkl").write_bytes(pickle.dumps(vectors))
            files = {path.relative_to(root).as_posix(): sha256_file(path)
                     for path in root.rglob("*") if path.is_file()}
            write_json(root / "manifest.json", {
                "bundle_type": "acoustic_embeddings", "schema_version": 1, "model": model,
                "inventory_sha256": files["sound_inventory.jsonl"], "artifacts": files,
                "count": len(vectors), "dimension": 3,
            })
            params[key] = str(root)
        workflow = {"id": "reference", "workflow": "reference-geometry", "parameters": params}
        result = run_workflow(workflow, fixture.root / "reports")
        self.assertEqual(run_workflow(workflow, fixture.root / "reports"), result)
        self.assertIn("reference_geometry.html", validate_workflow_result(result)["artifacts"])
        page = (result / "reference_geometry.html").read_text(encoding="utf-8")
        self.assertNotIn("@@", page)
        self.assertIn("4 syllables", page)
        self.assertIn("6 unique pairs", page)
        self.assertIn("const rankGroupSize=0;", page)
        self.assertIn("const prepared=", page)
        self.assertIn("Coverage and report files", page)
        self.assertNotIn("char-zh-cn-geometry-v1", page)
        inventory = json.loads((result / "data/inventory.json").read_text())
        self.assertEqual(inventory["labels"], sorted(fixture.fixture_labels))
        stems = ("dimsim-distances", "base-cosine-distances", "zh-cn-cosine-distances")
        self.assertEqual({path.name for path in (result / "figures").iterdir()},
                         {f"{stem}.{suffix}" for stem in stems for suffix in ("png", "eps", "metadata.json")})
        metadata = [json.loads((result / f"figures/{stem}.metadata.json").read_text()) for stem in stems]
        self.assertEqual(metadata[1]["resolved_data"]["color_limits"], metadata[2]["resolved_data"]["color_limits"])
        self.assertTrue(all(item["resolved_figure"]["xlabel"] == "Candidate"
                            and item["resolved_figure"]["ylabel"] == "Anchor"
                            and not item["resolved_data"]["axis_tick_labels"] for item in metadata))
        self.assertTrue(all(f"figures/{stem}.png" in page for stem in stems))
        self.assertTrue((result / "tables/dimsim_agreement.csv").is_file())
        with np.load(result / "data/distances.npz", allow_pickle=False) as data:
            self.assertTrue(np.all(data["distances"].diagonal(axis1=1, axis2=2) == 0))
