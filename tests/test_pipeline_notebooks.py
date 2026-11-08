"""Generated SDK snapshots and exact input pins without ML execution."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from phonetics.execution import create_job, validate_job
from phonetics.pipeline import load_config
import test_character_experiment as fixtures


class PipelineNotebookTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CharacterExperimentTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.experiment = self.fixture.build()
        self.config = load_config(PROJECT / 'configs/experiment.example.json')
        self.config['experiment'] = self.fixture.config
        self.config['experiment_id'] = self.experiment.name
        self.config['paths']['generated'] = str(self.root / 'generated')

    def test_schema_pins_order_and_isolated_sdk_import(self):
        import nbformat
        job = create_job(self.config, 'training', self.experiment, model_id='syllable')
        notebook = nbformat.read(job['notebook'], as_version=4)
        nbformat.validate(notebook)
        cells = {c.id: c.source for c in notebook.cells}
        self.assertLess(list(cells).index('dependencies'), list(cells).index('sdk'))
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                ast.parse(cell.source)
                self.assertNotIn('%%writefile', cell.source)
                self.assertNotIn('RUN_TRAINING', cell.source)
        self.assertIn(validate_job(job['path'])['input_manifest_sha256'], cells['configuration'])
        self.assertIn('allow_execution=True', cells['run'])
        script = cells['sdk'] + '\n' + cells['configuration'] + '\nfrom phonetics.cli import main\nmain(["--help"])\nassert "torch" not in sys.modules\n'
        # A script file avoids Windows command-line length limits.
        path = self.root / 'bootstrap.py'
        path.write_text(script, encoding='utf-8')
        result = subprocess.run([sys.executable, '-S', str(path)], cwd=self.root,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('generate-notebooks', result.stdout)

    def test_tampering_and_deterministic_regeneration(self):
        job = create_job(self.config, 'training', self.experiment, model_id='syllable')
        before = Path(job['notebook']).read_bytes()
        self.assertEqual(job, create_job(self.config, 'training', self.experiment, model_id='syllable'))
        self.assertEqual(before, Path(job['notebook']).read_bytes())
        (Path(job['path']) / 'job.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_job(job['path'])

    def test_generation_preserves_input(self):
        before = {p: p.read_bytes() for p in self.experiment.rglob('*') if p.is_file()}
        create_job(self.config, 'training', self.experiment, model_id='components')
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertNotIn('torch', sys.modules)

    def test_all_objectives_validate_once_without_copying_data(self):
        from unittest.mock import patch
        from phonetics.notebooks import generate_notebooks
        from phonetics.experiment import validate_experiment
        self.config['paths']['experiments'] = str(self.experiment.parent)
        with patch('phonetics.experiment.validate_experiment', wraps=validate_experiment) as check, patch('phonetics.execution.shutil.copytree', side_effect=AssertionError('must not copy experiment')):
            jobs = generate_notebooks(self.config, stage='training')
        self.assertEqual(check.call_count, 1)
        self.assertEqual(len(jobs), len(self.config['experiment']['model_ids']))
        for job in jobs:
            self.assertFalse((Path(job['path']) / 'input').exists())
            self.assertFalse((Path(job['path']) / 'sdk').exists())

    def test_relocated_experiment_uses_current_path_and_preserves_old_job(self):
        import json, shutil
        first = create_job(self.config,'training',self.experiment,model_id='syllable')
        old_files = {p:p.read_bytes() for p in Path(first['path']).rglob('*') if p.is_file()}
        moved = self.root/'relocated'/self.experiment.name
        shutil.copytree(self.experiment,moved)
        shutil.rmtree(self.experiment)
        second = create_job(self.config,'training',moved,model_id='syllable')
        self.assertNotEqual(first['job_id'],second['job_id'])
        self.assertEqual(old_files,{p:p.read_bytes() for p in old_files})
        notebook = json.loads(Path(second['notebook']).read_text(encoding='utf-8'))
        cells = {c['id']:''.join(c['source']) for c in notebook['cells']}
        script = cells['sdk']+'\n'+cells['configuration']+f'\nassert INPUT_PATH == Path({str(moved.resolve())!r})\nassert "torch" not in sys.modules\n'
        path = self.root/'relocation.py';path.write_text(script,encoding='utf-8')
        result = subprocess.run([sys.executable,'-S',str(path)],cwd=self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        attached = self.root/'attached'/'experiment'
        shutil.copytree(moved,attached)
        (moved/'manifest.json').write_text('{}',encoding='utf-8')
        mapping = f'\nREAL_PATH = Path\ndef mapped_path(value):\n    return REAL_PATH({str(attached.parent)!r}) if str(value) == "/kaggle/input" else REAL_PATH(value)\nmapped_path.cwd = REAL_PATH.cwd\nPath = mapped_path\n'
        script = cells['sdk']+mapping+cells['configuration']+f'\nassert INPUT_PATH == REAL_PATH({str(attached.resolve())!r})\nassert "torch" not in sys.modules\n'
        path.write_text(script,encoding='utf-8')
        result = subprocess.run([sys.executable,'-S',str(path)],cwd=self.root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_changed_config_and_nested_output_are_rejected(self):
        self.config['experiment']['training']['epochs'] += 1
        with self.assertRaisesRegex(ValueError, 'differs from the configured experiment'):
            create_job(self.config, 'training', self.experiment, model_id='syllable')
        self.config['experiment']['training']['epochs'] -= 1
        self.config['paths']['generated'] = str(self.experiment/'generated')
        with self.assertRaisesRegex(ValueError, 'outside its input tree'):
            create_job(self.config, 'training', self.experiment, model_id='syllable')
