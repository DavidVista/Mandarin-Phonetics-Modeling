import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "phonetics"
phonetics_package = types.ModuleType("phonetics"); phonetics_package.__path__ = [str(SOURCE)]
evaluation_package = types.ModuleType("phonetics.evaluation"); evaluation_package.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics_package); sys.modules.setdefault("phonetics.evaluation", evaluation_package)
bundle = importlib.import_module("phonetics.evaluation.bundle")
pinyin = importlib.import_module("phonetics.pinyin")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_artifact_hash(bundle_path, relative):
    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][relative] = digest(bundle_path / relative)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class ContextualBundleTests(unittest.TestCase):
    def make_experiment(self, root, name="exp"):
        exp = root / name; exp.mkdir()
        labels = ["ma1", "ma2", "mi1", "mi2", "ba1", "ba2", "bi1", "bi2"]
        labels.extend(["a1", "da1"])
        counts = {label: 2 + (i % 3) for i, label in enumerate(labels[:8])}
        counts.update({"a1": 2, "da1": 1})
        initial_labels = ["", "m", "b", "d"]
        final_labels = ["a", "i"]
        tone_labels = ["1", "2"]
        initial_ids = {value: i for i, value in enumerate(initial_labels)}
        final_ids = {value: i for i, value in enumerate(final_labels)}
        tone_ids = {value: i for i, value in enumerate(tone_labels)}
        acoustic_ids, component_labels = [], []
        for i, label in enumerate(labels):
            parsed = pinyin.parse_components(label)
            for _ in range(counts[label]):
                acoustic_ids.append(i)
                component_labels.append([initial_ids[parsed.initial], final_ids[parsed.final], tone_ids[parsed.tone]])
        alphabet = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地玄黄"
        sample = {"sample_id": "s", "text": "".join(alphabet[i % len(alphabet)] for i in range(len(acoustic_ids))), "acoustic_ids": acoustic_ids,
                  "component_labels": component_labels}
        vocab = {"acoustic_tokens": labels, "labels": {"initials": initial_labels, "finals": final_labels, "tones": tone_labels}}
        (exp / "samples.jsonl").write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")
        (exp / "splits.json").write_text(json.dumps({"train": [], "validation": [], "test": ["s"]}), encoding="utf-8")
        (exp / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (exp / "config.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        manifest = {"experiment_id": "fixture", "pinyin_contract": pinyin.PINYIN_CONTRACT_VERSION,
                    "parser_provenance": pinyin.PARSER_PROVENANCE,
                    "artifacts": {name: digest(exp / name) for name in ("samples.jsonl", "splits.json", "vocab.json", "config.json")}}
        (exp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return exp

    def prepare(self, root, evaluation_id="one", q_draws=1000):
        exp = self.make_experiment(root, "exp-" + evaluation_id)
        return bundle.prepare_evaluation_bundle(
            evaluation_id, experiment_path=exp, output_root=root / "out",
            distance_fn=lambda a, b: 0.0 if a == b else 20.0,
            metric_provenance={"metric_name": "synthetic"}, code_provenance={"fixture": "0" * 64},
            protocol_config={"dimsim_natural_count": 20, "dimsim_balanced_count": 240,
                            "anchor_q_draws_per_edge": q_draws})

    def test_schema3_default_q_table_and_no_aggregates(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp); path = self.prepare(root)
            loaded = bundle.load_evaluation_bundle(path)
            self.assertEqual(len(loaded["triplets"]), 12000)
            self.assertEqual(set(loaded["pairs"]), {"dimsim-natural", "dimsim-balanced"})
            self.assertEqual(len(loaded["pairs"]["dimsim-natural"]), 20)
            self.assertEqual(len(loaded["pairs"]["dimsim-balanced"]), 240)
            self.assertEqual(len(loaded["occurrences"]), 26)
            zero_ids = {row["occurrence_id"] for row in loaded["occurrences"] if not row["initial"]}
            self.assertEqual(len(zero_ids), 2)
            self.assertTrue(all(row["left_id"] not in zero_ids and row["right_id"] not in zero_ids
                                for mode in loaded["pairs"].values() for row in mode))
            self.assertNotIn("groups", loaded)
            self.assertFalse((path / "groups.jsonl").exists())
            self.assertFalse((path / "pairs/character_pinyin.jsonl").exists())
            counts = loaded["coverage"]["anchor_q"]["edges"]
            self.assertEqual(len(counts), 12)
            self.assertTrue(all(row["draw_count"] == 1000 for row in counts.values()))
            self.assertLess(len({row["triplet_id"] for row in loaded["triplets"]}), len(loaded["triplets"]))
            self.assertTrue(all(len({r[k] for k in ("anchor_id", "high_id", "low_id")}) == 3
                                for r in loaded["triplets"]))

    def test_reproducible_q_and_balanced_plan(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            first = self.prepare(root, "first", 3)
            second = self.prepare(root, "second", 3)
            a, b = bundle.load_evaluation_bundle(first), bundle.load_evaluation_bundle(second)
            self.assertEqual(a["triplets"], b["triplets"])
            self.assertEqual(a["pairs"]["dimsim-natural"], b["pairs"]["dimsim-natural"])
            self.assertEqual(a["pairs"]["dimsim-balanced"], b["pairs"]["dimsim-balanced"])
            pair_counts = Counter(row["pair_id"] for row in a["pairs"]["dimsim-balanced"])
            self.assertLess(max(pair_counts.values()), len(a["pairs"]["dimsim-balanced"]))
            self.assertLess(len(pair_counts), len(a["pairs"]["dimsim-balanced"]))
            self.assertEqual(len({row["draw_id"] for row in a["pairs"]["dimsim-balanced"]}), 240)
            occurrence_by_id = {row["occurrence_id"]: row for row in a["occurrences"]}
            singleton_sid = next(row["syllable_id"] for row in a["syllables"] if row["occurrence_count"] == 1 and row["initial"])
            singleton_draws = [row for row in a["pairs"]["dimsim-balanced"] if row["first_syllable_id"] == singleton_sid]
            self.assertTrue(singleton_draws)
            self.assertTrue(all(row["first_syllable_id"] != row["second_syllable_id"] for row in singleton_draws))
            self.assertTrue(all(occurrence_by_id[row["first_id"]]["initial"] and occurrence_by_id[row["second_id"]]["initial"]
                                for row in a["pairs"]["dimsim-balanced"]))
            self.assertTrue(all(row["syllable_id"] for row in a["occurrences"]))
            self.assertTrue(all("h" not in row and "vector" not in row for row in a["occurrences"] + a["syllables"]))

    def test_seeded_q_syllable_hierarchy_and_sampler_rng_independence(self):
        import numpy as np
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            data = bundle.load_evaluation_bundle(self.prepare(Path(tmp), q_draws=1))
            q_module = importlib.import_module("phonetics.evaluation.anchor_sampling")
            d_module = importlib.import_module("phonetics.evaluation.dimsim_sampling")
            occurrences, syllables = data["occurrences"], data["syllables"]
            member_ids = {row["syllable_id"]: row["occurrence_ids"] for row in syllables}
            metadata = {row["syllable_id"]: row for row in syllables}

            def overlap(anchor_id, candidate_id):
                return "".join(str(int(metadata[anchor_id][key] == metadata[candidate_id][key]))
                               for key in ("initial", "final", "tone"))

            # Independently replay only the anchor-level draw choices while
            # consuming all lower-level draws to retain the documented stream.
            rng = np.random.Generator(np.random.PCG64(20261026))
            expected_anchors = []
            for component, low_mask, high_mask in q_module.EDGES:
                anchors, highs_by_anchor, lows_by_anchor = [], {}, {}
                for sid in sorted(metadata):
                    if high_mask == "111" and len(member_ids[sid]) < 2:
                        continue
                    highs = [x for x in sorted(metadata) if overlap(sid, x) == high_mask and (x != sid or high_mask == "111")]
                    lows = [x for x in sorted(metadata) if overlap(sid, x) == low_mask and x != sid]
                    if highs and lows:
                        anchors.append(sid); highs_by_anchor[sid] = highs; lows_by_anchor[sid] = lows
                for _ in range(1000):
                    anchor = anchors[int(rng.integers(len(anchors)))]; expected_anchors.append(anchor)
                    anchor_occ = member_ids[anchor][int(rng.integers(len(member_ids[anchor])))]
                    high_sid = highs_by_anchor[anchor][int(rng.integers(len(highs_by_anchor[anchor])))]
                    low_sid = lows_by_anchor[anchor][int(rng.integers(len(lows_by_anchor[anchor])))]
                    high_occurrences = [x for x in member_ids[high_sid] if x != anchor_occ]
                    rng.integers(len(high_occurrences)); rng.integers(len(member_ids[low_sid]))
            actual, _ = q_module.sample_anchor_q(occurrences, syllables)
            self.assertEqual([row["anchor_syllable_id"] for row in actual], expected_anchors)

            balanced = d_module.sample_balanced(occurrences, syllables, count=240, seed=20261026)
            eligible = {row["syllable_id"]: row["occurrence_ids"] for row in syllables if row["initial"]}
            ids = sorted(eligible); rng = np.random.Generator(np.random.PCG64(20261026)); expected = []
            for index in range(240):
                first_sid = ids[int(rng.integers(len(ids)))]
                seconds = [sid for sid in ids if sid != first_sid or len(eligible[sid]) >= 2]
                second_sid = seconds[int(rng.integers(len(seconds)))]
                first_id = eligible[first_sid][int(rng.integers(len(eligible[first_sid])))]
                second_choices = [oid for oid in eligible[second_sid] if oid != first_id]
                second_id = second_choices[int(rng.integers(len(second_choices)))]
                expected.append((first_id, second_id, d_module.pair_identity(first_id, second_id), index, first_sid, second_sid))
            self.assertEqual(balanced, expected)

            q_before, natural_before = q_module.sample_anchor_q(occurrences, syllables, draws_per_edge=2), d_module.sample_natural(occurrences, 20)
            d_module.sample_balanced(occurrences, syllables, count=500, seed=999)
            q_after, natural_after = q_module.sample_anchor_q(occurrences, syllables, draws_per_edge=2), d_module.sample_natural(occurrences, 20)
            self.assertEqual(q_before, q_after)
            self.assertEqual(natural_before, natural_after)

    def test_loader_rejects_semantic_status_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = self.prepare(Path(tmp), q_draws=2)
            pair_path = path / "pairs/dimsim-balanced.jsonl"
            rows = [json.loads(x) for x in pair_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["distance_status"] = "ok"; rows[0]["dimsim_distance"] = None
            pair_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
            manifest_path = path / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["pairs/dimsim-balanced.jsonl"] = digest(pair_path)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                bundle.load_evaluation_bundle(path)

    def test_loader_rejects_rehashed_sampler_pinyin_and_q_coverage_forgery(self):
        mutations = (
            ("config.json", lambda value: value["sampling"]["anchor_q"].update(sampler_version=999), "anchor_q"),
            ("config.json", lambda value: value["sampling"]["anchor_q"].update(sampler_version=True), "anchor_q"),
            ("config.json", lambda value: value["sampling"]["anchor_q"].update(triplet_schema_version=999), "anchor_q"),
            ("config.json", lambda value: value["sampling"]["dimsim_balanced"].update(sampler_version=999), "dimsim_balanced"),
            ("pairs/dimsim-natural.jsonl", lambda value: value[0].update(left_pinyin="invalid"), "Pinyin labels"),
            ("coverage.json", lambda value: value["anchor_q"]["unique_triplets_by_edge"].update(
                {key: 999999 for key in value["anchor_q"]["unique_triplets_by_edge"]}), "coverage"),
        )
        for relative, mutate, message in mutations:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
                path = self.prepare(Path(tmp), q_draws=1)
                artifact = path / relative
                if relative.endswith(".jsonl"):
                    value = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
                else:
                    value = json.loads(artifact.read_text(encoding="utf-8"))
                mutate(value)
                payload = ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in value)
                           if relative.endswith(".jsonl") else json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
                artifact.write_text(payload, encoding="utf-8")
                manifest_path = path / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["artifacts"][relative] = digest(artifact)
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    bundle.load_evaluation_bundle(path)

    def test_loader_rejects_boolean_triplet_schema_version(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = self.prepare(Path(tmp), q_draws=1)
            table = path / "triplets/anchor-q.jsonl"
            rows = [json.loads(line) for line in table.read_text(encoding="utf-8").splitlines()]
            rows[0]["schema_version"] = True
            table.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
            manifest_path = path / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["triplets/anchor-q.jsonl"] = digest(table)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema or edge"):
                bundle.load_evaluation_bundle(path)

    def test_loader_requires_well_formed_provenance_without_source_access(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            path = self.prepare(root, "source-removed", q_draws=1)
            source = root / "exp-source-removed"
            shutil.rmtree(source)
            self.assertEqual(len(bundle.load_evaluation_bundle(path)["occurrences"]), 26)

        mutations = (
            (lambda m: m.update(code_provenance=None), "code_provenance"),
            (lambda m: m.update(code_provenance={}), "code_provenance"),
            (lambda m: m.update(code_provenance={"module": "ABC" * 21 + "A"}), "code_provenance"),
            (lambda m: m["source_experiment"].update(hashes=None), "source_experiment"),
            (lambda m: m["source_experiment"]["hashes"].pop("samples.jsonl"), "source_experiment"),
            (lambda m: m["source_experiment"]["hashes"].update(extra="0" * 64), "source_experiment"),
            (lambda m: m["source_experiment"]["hashes"].update(config_json="bad"), "source_experiment"),
            (lambda m: m["source_experiment"].update(path=" "), "source_experiment"),
            (lambda m: m["source_experiment"].update(experiment_id=None), "source_experiment"),
        )
        for mutate, message in mutations:
            with self.subTest(mutation=mutate), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
                path = self.prepare(Path(tmp), q_draws=1)
                manifest_path = path / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    bundle.load_evaluation_bundle(path)

    def test_loader_rejects_q_rows_outside_global_order(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = self.prepare(Path(tmp), q_draws=2)
            table = path / "triplets/anchor-q.jsonl"
            rows = [json.loads(line) for line in table.read_text(encoding="utf-8").splitlines()]
            rows[:2], rows[2:4] = rows[2:4], rows[:2]
            write_jsonl(table, rows); update_artifact_hash(path, "triplets/anchor-q.jsonl")
            with self.assertRaisesRegex(ValueError, "global edge order"):
                bundle.load_evaluation_bundle(path)

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = self.prepare(Path(tmp), q_draws=2)
            table = path / "triplets/anchor-q.jsonl"
            rows = [json.loads(line) for line in table.read_text(encoding="utf-8").splitlines()]
            rows[0], rows[1] = rows[1], rows[0]
            write_jsonl(table, rows); update_artifact_hash(path, "triplets/anchor-q.jsonl")
            with self.assertRaisesRegex(ValueError, "draw index/order"):
                bundle.load_evaluation_bundle(path)

    def test_loader_requires_exact_json_primitive_types(self):
        cases = (
            ("manifest", lambda p, m: m.update(schema_version=3.0), None, "bundle schema"),
            ("manifest", lambda p, m: m.update(schema_version=True), None, "bundle schema"),
            ("config", lambda p, m: m.update(schema_version=3.0), "config.json", "config version"),
            ("config", lambda p, m: m["population"].update(aggregates=0, zero_initial_excluded_from_dimsim=1), "config.json", "population"),
            ("config", lambda p, m: m["sampling"]["anchor_q"].update(draws_per_edge=1.0), "config.json", "anchor_q"),
            ("manifest-counts", lambda p, m: m["counts"].update(sentences=True), None, "counts"),
        )
        for area, mutate, artifact, message in cases:
            with self.subTest(area=area), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
                path = self.prepare(Path(tmp), q_draws=1)
                manifest_path = path / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                target_path = manifest_path if area in ("manifest", "manifest-counts") else path / artifact
                value = manifest if target_path == manifest_path else json.loads(target_path.read_text(encoding="utf-8"))
                mutate(path, value)
                if target_path != manifest_path:
                    target_path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                    manifest["artifacts"][artifact] = digest(target_path)
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    bundle.load_evaluation_bundle(path)

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = self.prepare(Path(tmp), q_draws=2)
            table = path / "triplets/anchor-q.jsonl"
            rows = [json.loads(line) for line in table.read_text(encoding="utf-8").splitlines()]
            rows[1]["draw_index"] = True
            write_jsonl(table, rows); update_artifact_hash(path, "triplets/anchor-q.jsonl")
            with self.assertRaisesRegex(ValueError, "draw index/order"):
                bundle.load_evaluation_bundle(path)

    def test_q_coverage_failure_reports_all_required_edges(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = self.prepare(Path(tmp), q_draws=1)
            data = bundle.load_evaluation_bundle(path)
            sampler = importlib.import_module("phonetics.evaluation.anchor_sampling")
            only_syllable = data["syllables"][:1]
            only_ids = set(only_syllable[0]["occurrence_ids"])
            occurrences = [row for row in data["occurrences"] if row["occurrence_id"] in only_ids]
            with self.assertRaisesRegex(ValueError, "coverage failure") as caught:
                sampler.sample_anchor_q(occurrences, only_syllable, draws_per_edge=1)
            diagnostic = json.loads(str(caught.exception).split("diagnostics=", 1)[1])
            self.assertEqual(len(diagnostic["missing_edges"]), 12)

    def test_portable_preparation_functions_execute_on_fixture(self):
        notebook_path = ROOT / "notebooks" / "Datasets" / "Evaluation Setup" / "prepare-evaluation.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = lambda cell: "".join(cell["source"])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp); exp = self.make_experiment(root, "portable-exp")
            namespace = {"__name__": "portable_preparation_fixture"}
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code" and (cell["id"].endswith("build") or "-module-" in cell["id"] or cell["id"] == "prep-pinyin"):
                    exec(compile(code(cell), f"portable:{cell['id']}", "exec"), namespace)
            config = code(next(c for c in notebook["cells"] if c["id"] == "prep-config"))
            exec(compile(config, "portable:prep-config", "exec"), namespace)
            namespace.update({"EXPERIMENT_PATH": exp, "OUTPUT_ROOT": root / "portable-out", "EVALUATION_ID": "portable",
                              "DIMSIM_NATURAL_PAIR_COUNT": 20, "DIMSIM_BALANCED_DRAW_COUNT": 40,
                              "ANCHOR_Q_DRAWS_PER_EDGE": 2})
            class SyntheticMetric:
                provenance = {"metric_name": "synthetic"}
                def __call__(self, left, right): return 0.0 if left == right else 5.0
            namespace["DimsimDistance"] = SyntheticMetric
            run = code(next(c for c in notebook["cells"] if c["id"] == "prep-run"))
            exec(compile(run, "portable:prep-run", "exec"), namespace)
            loaded = namespace["load_evaluation_bundle"](namespace["bundle_path"])
            self.assertEqual(len(loaded["triplets"]), 24)
            self.assertNotIn("groups", loaded)


if __name__ == "__main__": unittest.main()
