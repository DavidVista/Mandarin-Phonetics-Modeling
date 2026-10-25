import importlib
import sys
import types
import unittest
import tempfile
import json
from dataclasses import dataclass
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "phonetics"
phonetics_package = types.ModuleType("phonetics"); phonetics_package.__path__ = [str(SOURCE)]
evaluation_package = types.ModuleType("phonetics.evaluation"); evaluation_package.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics_package); sys.modules.setdefault("phonetics.evaluation", evaluation_package)
module = importlib.import_module("phonetics.evaluation.dimsim_alignment")


@dataclass
class Author:
    consonant: str
    vowel: str
    tone: int


@dataclass
class Strict:
    initial: str
    final: str
    tone: str


A = {
    "ma1": Author("m", "a", 1), "mu1": Author("m", "u", 1),
    "ju1": Author("j", "u", 1), "liu1": Author("l", "iu", 1),
    "a1": Author("", "a", 1), "bad1": ValueError("bad A"),
}
B = {
    "ma1": Strict("m", "a", "1"), "mu1": Strict("m", "u", "1"),
    "ju1": Strict("j", "v", "1"), "liu1": Strict("l", "iou", "1"),
    "a1": Strict("", "a", "1"), "bad1": ValueError("bad B"),
}


def parser(table):
    def parse(label):
        value = table[label]
        if isinstance(value, Exception):
            raise value
        return value
    return parse


def bundle():
    labels = ["ma1", "mu1", "ju1", "liu1", "a1", "bad1"]
    return {
        "groups": [{"group_kind": "syllable", "pinyin": label} for label in labels[:-1]],
        "occurrences": [{"pinyin": label} for label in labels],
    }


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rows = module.build_decompositions(
            bundle(), author_parser=parser(A), strict_parser=parser(B)
        )
        self.by_label = {row["label"]: row for row in self.rows}

    def test_inventory_raw_components_categories_and_zero_policy(self):
        self.assertEqual([row["label"] for row in self.rows], sorted(A))
        self.assertEqual(self.by_label["ma1"]["category"], "exact")
        self.assertEqual(self.by_label["ju1"]["category"], "u_vs_umlaut")
        self.assertEqual(self.by_label["liu1"]["category"], "contracted_final")
        self.assertEqual(self.by_label["bad1"]["category"], "parse_error")
        self.assertEqual(self.by_label["ju1"]["a_final"], "u")
        self.assertEqual(self.by_label["ju1"]["b_final"], "v")
        self.assertEqual(module.eligible_labels(self.rows), ["ju1", "liu1", "ma1", "mu1"])

    def test_affected_union_is_unique_and_has_closed_form_size(self):
        pairs = module.affected_pair_plan(self.rows)
        # n=4, k=2: 2*3 - 2*1/2 = 5.
        self.assertEqual(len(pairs), 5)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(left < right for left, right in pairs))
        self.assertNotIn(("ma1", "mu1"), pairs)

    def test_pair_engine_preserves_directions_and_full_symmetry_scope(self):
        order = {label: index for index, label in enumerate(sorted(A))}
        scorer_a = lambda left, right: order[left] * 10.0 + order[right]
        scorer_b = lambda left, right: order[left] * 20.0 + order[right]
        reproduce = lambda left, right: scorer_a(
            next(label for label, obj in A.items() if obj is left),
            next(label for label, obj in A.items() if obj is right),
        )
        result = module.evaluate_alignment(
            self.rows, scorer_a=scorer_a, scorer_b=scorer_b,
            reproduction_scorer=reproduce, author_parser=parser(A),
        )
        self.assertEqual(len(result["affected_pairs"]), 5)
        self.assertEqual(result["symmetry_summary"]["planned"], 6)
        self.assertEqual(result["symmetry_summary"]["valid"], 6)
        self.assertEqual(result["symmetry_summary"]["a_asymmetric"], 6)
        self.assertEqual(result["symmetry_summary"]["b_asymmetric"], 6)
        row = result["affected_pairs"][0]
        self.assertNotEqual(row["a_forward"], row["a_reverse"])
        self.assertEqual(row["delta_forward"], row["b_forward"] - row["a_forward"])
        self.assertEqual(row["e_a"], abs(row["a_forward"] - row["a_reverse"]))
        self.assertEqual({row["left_category"], row["right_category"]} & {"u_vs_umlaut", "contracted_final"},
                         {row["left_category"], row["right_category"]} - {"exact"})

    def test_invalid_and_reproduction_mismatch_are_explicit(self):
        def scorer_a(left, right):
            if {left, right} == {"ma1", "mu1"}:
                return float("nan")
            return 1.0
        result = module.evaluate_alignment(
            self.rows, scorer_a=scorer_a, scorer_b=lambda _l, _r: 2.0,
            reproduction_scorer=lambda _l, _r: 99.0, author_parser=parser(A),
        )
        self.assertEqual(result["symmetry_summary"]["valid"], 0)
        self.assertEqual(result["symmetry_summary"]["invalid"], 6)
        self.assertTrue(all(not row["joint_valid"] for row in result["symmetry_exceptions"]))
        self.assertTrue(any(row["a_forward_status"] == "nonfinite" for row in result["symmetry_exceptions"]))
        self.assertTrue(all(row["delta_forward"] is None for row in result["affected_pairs"]))

    def test_invalid_parser_shapes_and_compound_category(self):
        malformed_a = dict(A); malformed_a["ma1"] = Author(None, "a", 1)
        rows = module.build_decompositions(bundle(), author_parser=parser(malformed_a), strict_parser=parser(B))
        self.assertEqual(next(row for row in rows if row["label"] == "ma1")["category"], "parse_error")
        changed_b = dict(B); changed_b["ju1"] = Strict("q", "v", "1")
        rows = module.build_decompositions(bundle(), author_parser=parser(A), strict_parser=parser(changed_b))
        self.assertEqual(next(row for row in rows if row["label"] == "ju1")["category"], "initial+u_vs_umlaut")

    def test_other_final_difference_preserves_compound_category_and_scope(self):
        cases = [
            (Strict("m", "o", "1"), "other"),
            (Strict("n", "o", "1"), "initial+other"),
            (Strict("m", "o", "2"), "tone+other"),
            (Strict("n", "o", "2"), "initial+tone+other"),
        ]
        for components, expected in cases:
            with self.subTest(expected=expected):
                changed = dict(B); changed["ma1"] = components
                rows = module.build_decompositions(bundle(), author_parser=parser(A), strict_parser=parser(changed))
                row = next(row for row in rows if row["label"] == "ma1")
                self.assertEqual(row["category"], expected)
                self.assertTrue(row["difference_flags"]["final_changed"])
                alignment = module.evaluate_alignment(rows, scorer_a=lambda _a, _b: 1.0,
                    scorer_b=lambda _a, _b: 1.0, reproduction_scorer=lambda _a, _b: 1.0,
                    author_parser=parser(A))
                result = module.summarize_statistics(rows, alignment, bootstrap_replicates=2)
                self.assertIn(expected, {item["scope"] for item in result["statistics_by_category"]})
                self.assertEqual(next(item for item in result["statistics_by_syllable"]
                                      if item["syllable"] == "ma1")["category"], expected)

    def test_weighted_spearman_matches_expansion_and_bootstrap_is_deterministic(self):
        x, y, weights = [1, 2, 3], [3, 1, 2], [2, 1, 3]
        weighted, status = module.weighted_spearman(x, y, weights)
        expanded_x = [value for value, weight in zip(x, weights) for _ in range(weight)]
        expanded_y = [value for value, weight in zip(y, weights) for _ in range(weight)]
        ordinary, _ = module.weighted_spearman(expanded_x, expanded_y, [1] * len(expanded_x))
        self.assertEqual(status, "ok"); self.assertAlmostEqual(weighted, ordinary)

        scorer = lambda left, right: float(sorted(A).index(left) + sorted(A).index(right))
        alignment = module.evaluate_alignment(
            self.rows, scorer_a=scorer, scorer_b=scorer,
            reproduction_scorer=lambda left, right: scorer(
                next(label for label, obj in A.items() if obj is left), next(label for label, obj in A.items() if obj is right)),
            author_parser=parser(A),
        )
        one = module.summarize_statistics(self.rows, alignment, seed=7, bootstrap_replicates=20)
        two = module.summarize_statistics(self.rows, alignment, seed=7, bootstrap_replicates=20)
        self.assertEqual(one, two)
        self.assertFalse(one["use_both_directions"])
        self.assertEqual(len(one["statistics_by_syllable"]), 2)
        self.assertEqual(one["bootstrap_diagnostics"]["replicates"], 20)

    def test_atomic_export_has_fixed_files_hashes_and_refuses_overwrite(self):
        scorer = lambda left, right: float(sorted(A).index(left) + sorted(A).index(right))
        alignment = module.evaluate_alignment(
            self.rows, scorer_a=scorer, scorer_b=scorer,
            reproduction_scorer=lambda left, right: scorer(
                next(label for label, obj in A.items() if obj is left), next(label for label, obj in A.items() if obj is right)),
            author_parser=parser(A),
        )
        statistics = module.summarize_statistics(self.rows, alignment, seed=3, bootstrap_replicates=3)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            target = Path(temporary) / "run"
            module.export_alignment(target, config={"run_id": "run"}, decompositions=self.rows,
                                    alignment=alignment, statistics=statistics, provenance={"fixture": True})
            expected = {"config.json", "manifest.json", "decompositions.csv", "coverage.csv", "pairs.csv.gz",
                        "symmetry-summary.csv", "symmetry-exceptions.csv.gz", "statistics-by-syllable.csv",
                        "statistics-by-category.csv", "bootstrap-diagnostics.json"}
            self.assertEqual({path.name for path in target.iterdir()}, expected)
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["artifacts"]), expected - {"manifest.json"})
            with self.assertRaises(FileExistsError):
                module.export_alignment(target, config={}, decompositions=[], alignment={}, statistics={}, provenance={})

    def test_metric_tolerance_matches_directional_math_isclose_at_large_scale(self):
        pair = {"pair_id": "p", "joint_valid": True, "a_forward": 1e9, "a_reverse": 1e9 + 0.05,
                "b_forward": 1e9 + 0.04, "b_reverse": 1e9 + 0.09,
                "e_a": 0.05, "e_b": 0.05}
        observations = [{"pair": pair, "direction": "forward", "a": pair["a_forward"],
                         "b": pair["b_forward"], "valid": True}]
        metrics, _ = module._metrics(observations, atol=1e-12, rtol=1e-10)
        self.assertEqual(metrics["changed_fraction"], 0.0)
        self.assertEqual(metrics["asymmetric_fraction_a"], 0.0)
        self.assertEqual(metrics["asymmetric_fraction_b"], 0.0)


    def synthetic_alignment(self, *, asymmetric=False, shift=0.0):
        index = {label: i for i, label in enumerate(sorted(A))}
        def score(left, right):
            return float(index[left] * (10 if asymmetric else 1) + index[right])
        def reproduce(left, right):
            return score(next(label for label, obj in A.items() if obj is left),
                         next(label for label, obj in A.items() if obj is right))
        return module.evaluate_alignment(self.rows, scorer_a=score,
                                        scorer_b=lambda left, right: score(left, right) + shift,
                                        reproduction_scorer=reproduce, author_parser=parser(A))

    def test_cached_ranks_match_independent_expanded_ranks_with_ties_and_zero_weights(self):
        import math
        import numpy as np
        x, y, weights = [1., 1., 3., 4.], [4., 2., 2., 1.], [2, 0, 3, 1]
        def ordinary_ranks(values):
            ordered = sorted(values)
            return [sum(i + 1 for i, item in enumerate(ordered) if item == value) / ordered.count(value)
                    for value in values]
        ex = [value for value, count in zip(x, weights) for _ in range(count)]
        ey = [value for value, count in zip(y, weights) for _ in range(count)]
        rx, ry = ordinary_ranks(ex), ordinary_ranks(ey)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        expected = sum((a-mx)*(b-my) for a,b in zip(rx,ry)) / math.sqrt(
            sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
        prepared = {"rank_groups_a": np.unique(x, return_inverse=True)[1],
                    "rank_groups_b": np.unique(y, return_inverse=True)[1]}
        actual, status = module._cached_spearman(prepared, np.asarray(weights, dtype=float))
        self.assertEqual(status, "ok")
        self.assertAlmostEqual(actual, expected)

    def test_bootstrap_does_not_resort_each_replica_and_delta_is_paired(self):
        from unittest.mock import patch
        alignment = self.synthetic_alignment(asymmetric=True, shift=3.0)
        original = module.np.unique
        calls = []
        for replicates in (2, 20):
            with patch.object(module.np, "unique", wraps=original) as unique:
                result = module.summarize_statistics(self.rows, alignment, seed=7, bootstrap_replicates=replicates)
            calls.append(unique.call_count)
        self.assertEqual(calls[0], calls[1])
        for row in result["statistics_by_category"] + result["statistics_by_syllable"]:
            self.assertAlmostEqual(row["mean_delta_ci_low"], 3.0)
            self.assertAlmostEqual(row["mean_delta_ci_high"], 3.0)
        overall = result["statistics_by_category"][0]
        self.assertGreater(overall["mean_b_ci_high"] - overall["mean_a_ci_low"], 3.0)

    def test_shared_weights_across_category_scopes_and_focal_directions(self):
        from unittest.mock import patch
        alignment = self.synthetic_alignment(asymmetric=True, shift=3.0)
        original = module._metrics_prepared
        traces = []
        def traced(prepared, weights, **kwargs):
            traces.append(dict(zip(prepared["pair_ids"], weights)))
            return original(prepared, weights, **kwargs)
        reps = 3
        with patch.object(module, "_metrics_prepared", side_effect=traced):
            result = module.summarize_statistics(self.rows, alignment, seed=4, bootstrap_replicates=reps)
        categories = len(result["statistics_by_category"])
        point_calls = categories + len(result["statistics_by_syllable"])
        aggregate = traces[point_calls:point_calls + categories * reps]
        for offset in range(0, len(aggregate), categories):
            overall = aggregate[offset]
            for category in aggregate[offset+1:offset+categories]:
                self.assertTrue(all(overall[key] == value for key, value in category.items()))
        focal = traces[point_calls + categories * reps:]
        self.assertTrue(focal)
        for offset in range(0, len(focal), 2):
            self.assertEqual(focal[offset], focal[offset+1])

    def test_run_guards_reject_before_any_analysis(self):
        from unittest.mock import patch
        import hashlib
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            base = Path(temporary); source = base / "input"; source.mkdir()
            manifest = source / "manifest.json"; manifest.write_text("{}", encoding="utf-8")
            config = {"run_id": "test", "input_path": source, "output_root": base / "output",
                      "input_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
            cases = [{**config, "run_id": "../escape"}, {**config, "run_id": "replace-with-explicit-run-id"},
                     {**config, "output_root": source}, {**config, "input_manifest_sha256": "0" * 64},
                     {**config, "bootstrap_replicates": True}, {**config, "atol": float("nan")},
                     {**config, "seed": -1}]
            with patch.object(module, "build_decompositions") as numerical:
                for candidate in cases:
                    with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                        module.run_alignment(candidate)
                numerical.assert_not_called()

    def test_empty_affected_export_retains_metric_headers_and_coverage_denominators(self):
        import csv, gzip
        exact = [row for row in self.rows if row["category"] == "exact"]
        scorer = lambda _left, _right: 1.0
        alignment = module.evaluate_alignment(exact, scorer_a=scorer, scorer_b=scorer,
                                             reproduction_scorer=lambda _a, _b: 1.0, author_parser=parser(A))
        statistics = module.summarize_statistics(exact, alignment, bootstrap_replicates=2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            output = module.export_alignment(Path(temporary) / "run", config={}, decompositions=exact,
                                             alignment=alignment, statistics=statistics, provenance={})
            with gzip.open(output / "pairs.csv.gz", "rt", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertIn("a_forward_status", reader.fieldnames)
                self.assertEqual(list(reader), [])
            with (output / "statistics-by-syllable.csv").open(encoding="utf-8") as handle:
                self.assertIn("spearman_ci_low", csv.DictReader(handle).fieldnames)
            with (output / "coverage.csv").open(encoding="utf-8") as handle:
                self.assertTrue(all(row["denominator"] for row in csv.DictReader(handle)))


if __name__ == "__main__":
    unittest.main()
