"""Torch-free checks for arbitrary coefficients and objective identifiers."""
import math
import unittest

from phonetics.objectives import canonical_model_id, combined_model_id, parse_model_id
from phonetics.experiment import _validate_config, default_experiment_config, training_epoch_limit


class ObjectiveIdentifierTests(unittest.TestCase):
    def test_finite_weights_round_trip_without_allowlist_or_precision_loss(self):
        for weight in (0, .25, 2.5, 3, -0.5, 17.123456789, 1e-12, 1e20, 1.0000001):
            with self.subTest(weight=weight):
                model_id = combined_model_id(weight)
                self.assertEqual(parse_model_id(model_id), ("components-acoustic", float(weight)))
                self.assertEqual(canonical_model_id(model_id), model_id)
        self.assertEqual(combined_model_id(1.0), "components-acoustic-lambda-1")
        self.assertNotEqual(combined_model_id(1.0000001), combined_model_id(1.0000002))
        self.assertEqual(canonical_model_id("components-acoustic-lambda-2.50"), combined_model_id(2.5))

    def test_reject_nonfinite_nonnumeric_and_invalid_identifiers(self):
        for weight in (math.nan, math.inf, -math.inf, True, "2.5", None):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                combined_model_id(weight)
        for model_id in (None, "components-acoustic", "components-acoustic-lambda-nan",
                         "components-acoustic-lambda-inf", "components-acoustic-lambda-1e999",
                         "components-acoustic-lambda-../x"):
            with self.subTest(model_id=model_id), self.assertRaises(ValueError):
                parse_model_id(model_id)

    def test_config_accepts_arbitrary_objectives_and_rejects_alias_duplicates(self):
        cfg = default_experiment_config()
        cfg["model_ids"] = ["syllable", "components", "acoustic", combined_model_id(.37), combined_model_id(-2)]
        self.assertEqual(_validate_config(cfg, 1024), cfg)
        cfg["model_ids"] = ["components-acoustic-lambda-2.5", "components-acoustic-lambda-2.50"]
        with self.assertRaises(ValueError):
            _validate_config(cfg, 1024)


    def test_epoch_overrides_are_optional_and_only_target_configured_models(self):
        cfg = default_experiment_config()
        self.assertEqual(training_epoch_limit(_validate_config(cfg, 1024), "acoustic"), 20)
        cfg["training"]["epochs_by_model"] = {"acoustic": 40}
        self.assertEqual(_validate_config(cfg, 1024), cfg)
        self.assertEqual(training_epoch_limit(cfg, "acoustic"), 40)
        self.assertEqual(training_epoch_limit(cfg, "components"), 20)
        for overrides in (None, [], {"unknown": 10}, {"acoustic": 0}, {"acoustic": -1},
                          {"acoustic": True}, {"acoustic": 20.0}, {"acoustic": "20"}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "epochs_by_model"):
                cfg["training"]["epochs_by_model"] = overrides
                _validate_config(cfg, 1024)


if __name__ == "__main__":
    unittest.main()
