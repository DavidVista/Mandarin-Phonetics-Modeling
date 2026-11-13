"""Pause/resume and receipt validation without importing Torch."""
import copy
import json
from pathlib import Path
import sys
import shutil
import wave
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from phonetics.architecture import normalize_architecture
from phonetics.artifacts import sha256_file, write_json
from phonetics.bundles import seal_bundle
from phonetics.execution import run_pipeline, load_state, _write_state, import_result, execute_job
from phonetics.experiment import _validate_config
from phonetics.pipeline import load_config
import test_character_experiment as fixtures


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CharacterExperimentTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.experiment = self.fixture.build()
        self.acoustic = self.root / 'acoustic'
        self.acoustic.mkdir()
        for name in ('labels.json', 'targets.npy', 'inventory.jsonl', 'inference.json'):
            shutil.copyfile(self.fixture.prepared / name, self.acoustic / name)
        seal_bundle(self.acoustic, 'acoustic_targets', {'inventory_manifest_sha256': 'a' * 64})
        self.config = load_config(PROJECT / 'configs/experiment.example.json')
        self.config['acoustic'] = {'existing_bundle': str(self.acoustic),
                                   'manifest_sha256': sha256_file(self.acoustic / 'manifest.json')}
        self.config['experiment_id'] = self.experiment.name
        self.config['experiment'] = self.fixture.config
        self.config['paths'].update(state=str(self.root / 'state.json'), generated=str(self.root / 'generated'),
                                    experiments=str(self.root), runs=str(self.root / 'runs'))

    def mock_stage(self, config, stage, **kwargs):
        return self.acoustic if stage == 'prepare-embeddings' else self.experiment

    def make_result(self, pending, *, run_tree=False):
        from phonetics.execution import validate_job
        job = validate_job(pending['path'])
        root = self.root / 'download'
        payload = root / 'payload'
        if run_tree:
            payload = root / 'runs' / job['experiment_id'] / job['parameters']['model_id'] / job['parameters']['run_id']
        payload.mkdir(parents=True)
        (payload / 'opaque.pt').write_bytes(b'opaque checkpoint fixture; never deserialize')
        cfg = self.fixture.config
        seeds = list(range(cfg['seed_count']))
        for seed in seeds:
            directory = payload / f'seed-{seed}'
            directory.mkdir()
            for name in ('best.pt', 'last.pt', 'shared_encoder.pt', 'summary.json', 'history.json', 'events.jsonl'):
                (directory / name).write_bytes(b'fixture')
        (payload / 'summary.json').write_bytes(b'fixture')
        manifest = {'run_type': 'training_run', 'status': 'complete', 'experiment_id': self.experiment.name,
                    'model_id': pending['model_id'], 'run_id': self.config['execution']['run_id'], 'config': cfg,
                    'seeds': seeds, 'seed_status': {str(s): 'complete' for s in seeds},
                    'experiment_hashes': {'manifest.json': job['input_manifest_sha256']},
                    'code_sha256': {n: job['sources'][n] for n in ('training.py','model.py','metrics.py','architecture.py','experiment.py')},
                    'runtime': {'torch':'2.11.0'},
                    'artifacts': {p.relative_to(payload).as_posix(): sha256_file(p) for p in payload.rglob('*') if p.is_file()}}
        write_json(payload / 'manifest.json', manifest)
        write_json(root / 'receipt.json', {'job_id': job['job_id'], 'stage': 'training',
                                          'input_manifest_sha256': job['input_manifest_sha256']})
        seal_bundle(root, 'pipeline_result', {'job_id': job['job_id']})
        return root

    def test_pause_import_resume_next_objective(self):
        with patch('phonetics.pipeline.run_stage', side_effect=self.mock_stage):
            waiting = run_pipeline(self.config)
            self.assertEqual((waiting['status'], waiting['model_id']), ('waiting', 'syllable'))
            self.assertEqual(run_pipeline(self.config)['job_id'], waiting['job_id'])
            imported = import_result(self.config, self.make_result(waiting, run_tree=True))
            self.assertTrue((imported / 'opaque.pt').exists())
            self.assertIsNone(load_state(self.config)['waiting'])
            self.assertEqual(run_pipeline(self.config)['model_id'], 'components')
        self.assertNotIn('torch', sys.modules)

    def test_import_legacy_training_payload_after_sdk_upgrade(self):
        from phonetics.execution import sdk_files
        old_sources = sdk_files()
        old_sources['execution.py'] += '\n# Previous SDK snapshot\n'
        with patch('phonetics.execution.sdk_files', return_value=old_sources), patch('phonetics.pipeline.run_stage', side_effect=self.mock_stage):
            pending = run_pipeline(self.config)
            result = self.make_result(pending)
            old_state = load_state(self.config)
        with self.assertRaisesRegex(ValueError, 'configuration or SDK changed'):
            load_state(self.config)
        mismatched = copy.deepcopy(old_state)
        mismatched['identity']['sdk_sha256'] = '0' * 64
        _write_state(self.config['paths']['state'], mismatched)
        with self.assertRaisesRegex(ValueError, 'job SDK differs from the pending state SDK'):
            import_result(self.config, result)
        self.assertFalse(Path(self.config['paths']['runs']).exists())
        _write_state(self.config['paths']['state'], old_state)
        imported = import_result(self.config, result)
        self.assertTrue((imported / 'opaque.pt').exists())
        self.assertIsNone(load_state(self.config)['waiting'])
        with patch('phonetics.pipeline.run_stage', side_effect=self.mock_stage):
            self.assertEqual(run_pipeline(self.config)['model_id'], 'components')

    def test_wrong_receipt_cannot_publish_or_advance(self):
        with patch('phonetics.pipeline.run_stage', side_effect=self.mock_stage):
            pending = run_pipeline(self.config)
        root = self.make_result(pending)
        receipt = json.loads((root / 'receipt.json').read_text())
        receipt['job_id'] = 'f' * 64
        write_json(root / 'receipt.json', receipt)
        seal_bundle(root, 'pipeline_result', {'job_id': pending['job_id']})
        with self.assertRaisesRegex(ValueError, 'receipt'):
            import_result(self.config, root)
        self.assertIsNotNone(load_state(self.config)['waiting'])
        self.assertFalse(Path(self.config['paths']['runs']).exists())

    def test_state_rejects_config_changes(self):
        _write_state(self.config['paths']['state'], load_state(self.config))
        self.config['execution']['training_device'] = 'cpu'
        with self.assertRaisesRegex(ValueError, 'configuration or SDK changed'):
            load_state(self.config)

    def test_encoder_and_training_validation(self):
        value = copy.deepcopy(self.fixture.config)
        value['architecture']['encoder']['type'] = 'bert'
        with self.assertRaisesRegex(ValueError, 'unsupported encoder'):
            _validate_config(value, 2)
        for key, bad in (('epochs', True), ('batch_size', 0), ('learning_rate', float('nan'))):
            value = copy.deepcopy(self.fixture.config)
            value['training'][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validate_config(value, 2)
        value = copy.deepcopy(self.fixture.config)
        value['training'].update(epochs=2, batch_size=8, learning_rate=0.002)
        self.assertEqual(_validate_config(value, 2), value)

    def test_execution_guards_precede_torch_import(self):
        self.config['execution']['training'] = 'local'
        with patch('phonetics.pipeline.run_stage', side_effect=self.mock_stage):
            with self.assertRaisesRegex(PermissionError, '--allow-training'):
                run_pipeline(self.config)
        with self.assertRaises(PermissionError):
            execute_job('missing', 'missing')
        self.assertNotIn('torch', sys.modules)

    def test_historical_architecture_remains_readable(self):
        old = {'embedding_dim': 64, 'sentence_bilstm_hidden_per_direction': 64, 'sentence_bilstm_layers': 1,
               'context_dim': 128, 'classification_hidden_dim': 64, 'classification_dropout': .5, 'acoustic_dim': 2}
        self.assertEqual(normalize_architecture(old), self.fixture.config['architecture'])

    def test_external_input_pin_checked_before_training_import(self):
        job_root = self.root/'reference-job';job_root.mkdir()
        write_json(job_root/'reference.json',{'path':str(self.experiment)})
        job = {'stage':'training','external_input':True,'input_reference':str(self.experiment),'input_manifest_sha256':'f'*64}
        with patch('phonetics.execution.validate_job',return_value=job):
            with self.assertRaisesRegex(ValueError,'does not match the job'):
                execute_job(job_root,self.root/'rejected-result',allow_execution=True)
        self.assertNotIn('torch',sys.modules)

    def test_acoustic_notebook_handoff_then_experiment(self):
        inventory = self.root / 'inventory'
        inventory.mkdir()
        with wave.open(str(inventory / 'ma1.wav'), 'wb') as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
            handle.writeframes(b'\0\0' * 16)
        from phonetics.artifacts import write_jsonl
        write_jsonl(inventory / 'inventory.jsonl', [{'label':'ma1','relative_path':'ma1.wav',
                    'sha256':sha256_file(inventory/'ma1.wav'),'sample_rate':16000,'channels':1,'frames':16}])
        seal_bundle(inventory, 'sound_inventory', {'source':'fixture'})
        self.config['acoustic'] = {'existing_bundle':None, 'manifest_sha256':None}
        self.config['execution']['acoustic'] = 'notebook'
        self.config['paths']['inventory'] = str(inventory)
        self.config['paths']['acoustic'] = str(self.root/'imported-acoustic')
        def stages(config, stage, **kwargs):
            return inventory if stage == 'prepare-inventory' else self.experiment
        with patch('phonetics.pipeline.run_stage', side_effect=stages):
            waiting = run_pipeline(self.config)
            self.assertEqual(waiting['stage'], 'acoustic')
        def synthetic(inventory, destination, *, model, cache_root):
            shutil.copytree(self.acoustic, destination)
            seal_bundle(destination, 'acoustic_targets', {'inventory_manifest_sha256':sha256_file(inventory/'manifest.json'),
                        'model':model,'contract':'acoustic-targets-1','backend':'transformers'})
        with patch('phonetics.acoustic.prepare_acoustic_targets', side_effect=synthetic):
            result = execute_job(waiting['path'], self.root/'result', allow_execution=True)
        imported = import_result(self.config, result)
        self.assertEqual(imported, Path(self.config['paths']['acoustic']))
        with patch('phonetics.pipeline.run_stage', side_effect=stages):
            done = run_pipeline(self.config, through='experiment')
        self.assertEqual(done['status'], 'complete')
        self.assertNotIn('torch', sys.modules)

    def test_visualization_import_binds_parameters_and_allows_relocated_inputs(self):
        from phonetics.execution import create_job, validate_job
        from phonetics.bundles import content_hash
        from test_training_summary import make_training_fixture
        runs = make_training_fixture(self.root/'visualization-inputs')
        workflow = {'id':'report','workflow':'training-visualization', 'parameters':{
            'training_runs':{k:str(v['path']) for k,v in runs.items()}, 'summary_run_id':'summary','mode':'prepare'}}
        self.config['visualization'] = [workflow]
        self.config['paths']['reports'] = str(self.root/'reports')
        pending = create_job(self.config,'visualization',None,workflow=workflow)
        state = load_state(self.config);state['waiting'] = pending
        _write_state(self.config['paths']['state'],state)
        job = validate_job(pending['path'])
        result = self.root/'visualization-result';payload = result/'payload';payload.mkdir(parents=True)
        (payload/'fixture.json').write_text('{}',encoding='utf-8')
        relocated = copy.deepcopy(workflow)
        relocated['parameters']['training_runs'] = {m:'/remote/job/inputs/'+m for m in runs}
        identity = {'workflow':relocated,'input_pins':job['workflow_pins'],'source_sha256':content_hash(job['sources'])}
        write_json(result/'receipt.json',{'job_id':job['job_id'],'stage':'visualization','input_manifest_sha256':None})
        # A self-consistent wrapper for different parameters must still be rejected.
        relocated['parameters']['summary_run_id'] = 'different-summary'
        seal_bundle(payload,'workflow_result',identity)
        seal_bundle(result,'pipeline_result',{'job_id':job['job_id']})
        with self.assertRaisesRegex(ValueError,'provenance differs'):
            import_result(self.config,result)
        self.assertIsNotNone(load_state(self.config)['waiting'])
        self.assertFalse(Path(self.config['paths']['reports']).exists())
        relocated['parameters']['summary_run_id'] = 'summary'
        seal_bundle(payload,'workflow_result',identity)
        seal_bundle(result,'pipeline_result',{'job_id':job['job_id']})
        target = import_result(self.config,result)
        self.assertTrue((target/'fixture.json').exists())
        self.assertIsNone(load_state(self.config)['waiting'])
