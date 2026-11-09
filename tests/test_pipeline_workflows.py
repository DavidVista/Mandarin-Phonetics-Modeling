"""Local workflow adapters validate inputs before optional analysis dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT/'tests')]
from phonetics.workflows import resolve_parameters, workflow_pins, run_workflow, validate_workflow_result
from test_training_summary import make_training_fixture


class WorkflowTests(unittest.TestCase):
    def test_local_training_figures_reuse_and_input_tampering(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/'tests') as directory:
            root=Path(directory)
            runs=make_training_fixture(root)
            workflow={'id':'report','workflow':'training-visualization',
                      'parameters':{'training_runs':{k:str(v['path']) for k,v in runs.items()},
                                    'summary_run_id':'summary','mode':'prepare'}}
            output=run_workflow(workflow,root/'reports')
            validate_workflow_result(output)
            self.assertEqual(len(list(output.rglob('*.png'))),5)
            self.assertEqual(run_workflow(workflow,root/'reports'),output)
            from phonetics.execution import create_job
            from phonetics.pipeline import load_config
            config = load_config(PROJECT/'configs/experiment.example.json')
            config['paths']['generated'] = str(root/'generated')
            job = create_job(config, 'visualization', None, workflow=workflow)
            notebook = json.loads(Path(job['notebook']).read_text(encoding='utf-8'))
            cells = {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}
            script = root/'portable.py'
            script.write_text('\n'.join(cells[key] for key in ('sdk', 'configuration', 'run')),
                              encoding='utf-8')
            process = subprocess.run([sys.executable, str(script)], cwd=root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            portable = root/'results'/job['job_id']/'payload'
            self.assertTrue(portable.parent.with_suffix('.zip').is_file())
            for name in ('selected-checkpoints.md', 'selected-metrics.md', 'selected-summary.md'):
                local_table = output/'outputs'/'training-report'/'report'/'tables'/name
                remote_table = portable/'outputs'/'training-report'/'report'/'tables'/name
                self.assertEqual(local_table.read_bytes(), remote_table.read_bytes())
            one=Path(next(iter(runs.values()))['path'])/'manifest.json'
            one.write_text(one.read_text()+' ',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'inputs or parameters changed'):
                run_workflow(workflow,root/'reports')
        self.assertNotIn('torch',sys.modules)

    def test_config_rejects_unknown_parameters_and_unsupported_character_inputs(self):
        with self.assertRaisesRegex(ValueError,'invalid'):
            resolve_parameters('dimsim-distribution',{'evaluation_root':'x','typo':True},str)
        with tempfile.TemporaryDirectory(dir=PROJECT/'tests') as directory:
            root=Path(directory);(root/'data').write_text('x')
            from phonetics.bundles import seal_bundle
            seal_bundle(root,'character_experiment',{})
            workflow={'id':'report','workflow':'dimsim-distribution','parameters':{'evaluation_root':str(root)}}
            with self.assertRaisesRegex(ValueError,'character evaluation is separate'):
                workflow_pins(workflow)

    def test_reference_geometry_checks_base_before_calculation(self):
        with tempfile.TemporaryDirectory(dir=PROJECT/'tests') as directory:
            root = Path(directory)
            params = {'base_root':str(root/'base'), 'zh_cn_root':str(root/'zh'), 'evaluation_root':str(root/'eval')}
            workflow = {'id':'report','workflow':'reference-geometry','parameters':params}
            with self.assertRaisesRegex(ValueError, 'Base embedding bundle is missing'):
                workflow_pins(workflow)
            base = root/'base'; base.mkdir()
            (base/'manifest.json').write_text(json.dumps({'model':{'padding':'legacy_one_second'}}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'requires padding=none'):
                workflow_pins(workflow)
            params['base_manifest_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'Base embedding manifest SHA256 mismatch'):
                workflow_pins(workflow)

    def test_packaging_checks_every_input_before_creating_outputs(self):
        from phonetics.execution import create_job
        from phonetics.pipeline import load_config
        from phonetics.bundles import seal_bundle
        with tempfile.TemporaryDirectory(dir=PROJECT/'tests') as directory:
            root = Path(directory)
            inputs = []
            for name in ('syllable','components'):
                path = root/name;path.mkdir();(path/'summary.json').write_text('{}')
                seal_bundle(path,'log-fixture',{})
                inputs.append(path)
            workflow = {'id':'report','workflow':'training-visualization','parameters':{
                'training_runs':{name:str(path) for name,path in zip(('syllable','components'),inputs)},
                'summary_run_id':'summary','mode':'prepare'}}
            config = load_config(PROJECT/'configs/experiment.example.json')
            # Use the second input to detect implementations checking only the first.
            before = {p.relative_to(root).as_posix():(p.is_dir(),p.read_bytes() if p.is_file() else None)
                      for p in root.rglob('*')}
            for output in (inputs[1], inputs[1]/'generated'):
                config['paths']['generated'] = str(output)
                with self.subTest(output=output), self.assertRaisesRegex(ValueError,'outside its input tree'):
                    create_job(config,'visualization',None,workflow=workflow)
                after = {p.relative_to(root).as_posix():(p.is_dir(),p.read_bytes() if p.is_file() else None)
                         for p in root.rglob('*')}
                self.assertEqual(before,after)

    def test_nested_evaluation_settings_remain_in_the_workflow_contract(self):
        import copy
        from phonetics.workflows import workflow_contract
        workflow = {'id':'report','workflow':'evaluation-visualization','parameters':{
            'anchor_q_output':'q','dimsim_output':'d','random_control_output':'r','bundle_path':'b',
            'derived_run_ids':{'interpretation':'interpretation-1','presentation_data':'data-1'},'derived_mode':'prepare'}}
        relocated = copy.deepcopy(workflow);relocated['parameters']['bundle_path'] = '/remote/b'
        self.assertEqual(workflow_contract(workflow),workflow_contract(relocated))
        relocated['parameters']['derived_run_ids']['interpretation'] = 'different'
        self.assertNotEqual(workflow_contract(workflow),workflow_contract(relocated))
