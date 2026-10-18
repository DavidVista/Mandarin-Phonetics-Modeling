import importlib
import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise unittest.SkipTest(f"scientific evaluation dependencies unavailable: {error}")

SOURCE = Path(__file__).resolve().parents[1] / "src" / "phonetics"
phonetics_package = types.ModuleType("phonetics"); phonetics_package.__path__ = [str(SOURCE)]
evaluation_package = types.ModuleType("phonetics.evaluation"); evaluation_package.__path__ = [str(SOURCE / "evaluation")]
sys.modules.setdefault("phonetics", phonetics_package); sys.modules.setdefault("phonetics.evaluation", evaluation_package)
module = importlib.import_module("phonetics.evaluation.analysis")


class AnalysisTests(unittest.TestCase):
    def test_spearman_monotonic_reverse_ties_and_constant(self):
        self.assertAlmostEqual(module.spearman_result([1, 2, 3], [2, 4, 9])["rho"], 1.0)
        self.assertAlmostEqual(module.spearman_result([1, 2, 3], [9, 4, 2])["rho"], -1.0)
        self.assertGreater(module.spearman_result([1, 1, 2, 3], [1, 2, 2, 4])["rho"], 0)
        self.assertEqual(module.spearman_result([1, 1], [2, 3])["undefined_reason"], "constant_distance")

    def test_distance_bins_boundaries_and_invalids(self):
        self.assertEqual(module.distance_bin(0), ("ok", "[0,10)"))
        self.assertEqual(module.distance_bin(119.9), ("ok", "[110,120)"))
        self.assertEqual(module.distance_bin(120), ("ok", "[120,+inf)"))
        self.assertEqual(module.distance_bin(-0.1)[0], "invalid_negative")
        self.assertEqual(module.distance_bin(float("nan"))[0], "invalid_nonfinite")

    def test_cosine_retains_invalid_metric_rows(self):
        pairs = [{"pair_id": "p", "left_id": "a", "right_id": "b", "dimsim_distance": None,
                  "distance_status": "invalid:KeyError", "overlap_mask": "000", "no_zero_initial_diagnostic": True}]
        rows = module.cosine_rows(pairs, ["a", "b"], np.array([[1, 0], [0, 1]], dtype=np.float32))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["analysis_status"], "invalid:KeyError")

    def test_paired_delta_and_join_mismatch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            for name, rho in (("left", 0.2), ("right", 0.5)):
                run = root / name; (run / "similarities").mkdir(parents=True)
                manifest = {"run_label": name, "bundle_manifest_sha256": "b", "pair_plan_sha256": {x: x for x in module.LEVELS}}
                (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                metrics = {level: {"overall": {"rho": rho}, "strata": {
                    **{f"{value:03b}": {"rho": rho} for value in range(8)}, "no_zero_initial": {"rho": rho}
                }} for level in module.LEVELS}
                (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                for level in module.LEVELS:
                    row = {"pair_id": "p", "cosine_similarity": 0.4 + rho, "hidden_dissimilarity": 0.6 - rho}
                    (run / "similarities" / f"{level}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                manifest["metrics_sha256"] = hashlib.sha256((run / "metrics.json").read_bytes()).hexdigest()
                manifest["similarity_table_sha256"] = {level: hashlib.sha256((run / "similarities" / f"{level}.jsonl").read_bytes()).hexdigest() for level in module.LEVELS}
                (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = module.compare_paired_runs(root / "left", root / "right", output_dir=root / "comparison")
            result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(result["levels"]["contextual"]["delta_rho_right_minus_left"], 0.3)
            self.assertTrue((output / "joined" / "contextual.jsonl").is_file())
            changed = {"pair_id": "q", "cosine_similarity": 0.9, "hidden_dissimilarity": 0.1}
            (root / "right" / "similarities" / "contextual.jsonl").write_text(json.dumps(changed) + "\n", encoding="utf-8")
            right_manifest = json.loads((root / "right" / "manifest.json").read_text(encoding="utf-8"))
            right_manifest["similarity_table_sha256"]["contextual"] = hashlib.sha256((root / "right" / "similarities" / "contextual.jsonl").read_bytes()).hexdigest()
            (root / "right" / "manifest.json").write_text(json.dumps(right_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different ordered pair IDs"):
                module.compare_paired_runs(root / "left", root / "right", output_dir=root / "second")


if __name__ == "__main__":
    unittest.main()
