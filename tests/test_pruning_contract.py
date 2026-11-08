"""Prevent retired APIs from reentering the character preparation path."""
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class PruningContractTests(unittest.TestCase):
    def test_legacy_api_and_module_are_absent_without_importing_torch(self):
        code = f'''import sys, importlib.util
sys.path.insert(0, {str(PROJECT / "src")!r})
import phonetics
from phonetics.preparation.config import validate_model_config
from phonetics.source_preparation import unpack_sounds, export_duyu, download_sounds
for name in ('prepare_dataset', 'prepare_corpus', 'assemble_dataset', 'validate_prepared', 'prepare_acoustic_bundle'):
    assert name not in phonetics.__all__
    assert not hasattr(phonetics, name)
assert importlib.util.find_spec('phonetics.ids') is None
assert 'torch' not in sys.modules
'''
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_retired_files_are_not_part_of_current_checkout(self):
        for relative in ("src/phonetics/ids.py", "src/phonetics/preparation/pipeline.py",
                         "src/phonetics/preparation/assembly.py", "src/phonetics/preparation/corpus_bundle.py",
                         "src/phonetics/preparation/corpus_processing.py", "src/phonetics/preparation/readers.py",
                         "src/phonetics/preparation/acoustic_bundle.py", "datasets/ids-decomposition",
                         "notebooks/Evaluation/contextual-evaluation.ipynb",
                         "notebooks/Evaluation/contextual-evaluation-composite.ipynb"):
            self.assertFalse((PROJECT / relative).exists(), relative)
