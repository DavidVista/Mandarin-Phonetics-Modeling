"""Configuration, execution guards and CLI checks without Torch."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from phonetics.cli import main
from phonetics.pipeline import load_config, run_stage


class PipelineCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / "tests")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((PROJECT / "configs/experiment.example.json").read_text(encoding="utf-8"))
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_paths_resolve_from_config_even_from_another_working_directory(self):
        before = load_config(self.path)
        previous = Path.cwd()
        try:
            os.chdir(PROJECT / "src")
            after = load_config(self.path)
        finally:
            os.chdir(previous)
        self.assertEqual(before, after)
        self.assertEqual(Path(before["paths"]["source"]), (self.root / self.config["paths"]["source"]).resolve())

    def test_training_guard_fails_before_importing_torch(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(["train", "--experiment", "missing", "--manifest-sha256", "a" * 64,
                      "--model", "syllable", "--device", "cpu", "--output", str(self.root / "runs")])
        self.assertEqual(result.exception.code, 1)
        self.assertNotIn("torch", sys.modules)
        self.assertFalse((self.root / "runs").exists())

    def test_prepare_inference_guard_precedes_source_side_effects(self):
        config = load_config(self.path)
        config["acoustic"] = {"existing_bundle": None, "manifest_sha256": None}
        with patch("phonetics.pipeline.export_corpus", side_effect=AssertionError("unexpected source operation")):
            with self.assertRaisesRegex(PermissionError, "would require inference"):
                run_stage(config, "prepare")
        self.assertNotIn("torch", sys.modules)

    def test_missing_sample_size_is_rejected(self):
        del self.config["experiment"]["sample_size"]
        self.path.write_text(json.dumps(self.config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sample_size"):
            load_config(self.path)

    def test_installed_module_help_works_outside_checkout(self):
        result = subprocess.run([sys.executable, "-m", "phonetics", "--help"], cwd=self.root,
                                capture_output=True, text=True, check=True)
        self.assertIn("prepare-experiment", result.stdout)
        self.assertIn("download-sounds", result.stdout)


if __name__ == "__main__":
    unittest.main()
