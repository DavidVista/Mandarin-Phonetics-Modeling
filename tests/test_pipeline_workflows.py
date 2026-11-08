"""Local workflow adapters validate inputs before optional analysis dependencies."""
import json
from pathlib import Path
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
