"""Local execution and content-bound notebook handoffs without eager ML imports."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
from contextlib import contextmanager
from time import perf_counter

from .artifacts import sha256_file, write_json, safe_artifact
from .bundles import content_hash, seal_bundle, staging_directory, validate_bundle

PREPARATION = ('download-sounds', 'prepare-inventory', 'prepare-corpus',
               'prepare-embeddings', 'prepare-data', 'prepare-experiment')
PREPARATION_LABELS = dict(zip(PREPARATION, (
    'Download source audio', 'Build sound inventory', 'Export and process corpus',
    'Prepare acoustic targets', 'Assemble prepared data', 'Build experiment bundle',
)))
WORKFLOWS = {'dimsim-distribution', 'dimsim-parser-alignment', 'reference-geometry',
             'training-visualization', 'evaluation-visualization', 'training-report', 'dimsim-human-agreement'}


@contextmanager
def _progress_step(progress, label):
    if progress is None:
        yield
        return
    progress(f'{label}: started')
    started = perf_counter()
    yield
    progress(f'{label}: completed in {perf_counter() - started:.1f}s')


def sdk_files():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): p.read_text(encoding='utf-8')
            for p in sorted(root.rglob('*')) if p.suffix in {'.py', '.toml'}}


def sdk_hashes(files=None):
    import hashlib
    return {name: hashlib.sha256(body.encode('utf-8')).hexdigest() for name, body in (sdk_files() if files is None else files).items()}


def validate_execution_config(config, resolve):
    spec = config['execution']
    required = {'acoustic', 'training', 'training_device', 'run_id'}
    if not isinstance(spec, dict) or not required <= set(spec) <= required | {'preparation', 'visualization', 'additional_model_ids'}:
        raise ValueError('execution requires acoustic, training, training_device and run_id')
    from .objectives import is_extension_model
    additional = spec.get('additional_model_ids', [])
    if (not isinstance(additional, list) or not all(is_extension_model(model) for model in additional)
            or len(set(additional)) != len(additional) or set(additional) & set(config['experiment']['model_ids'])):
        raise ValueError('execution.additional_model_ids must contain distinct new syllable-acoustic objectives')
    spec.setdefault('visualization', 'local')
    config.setdefault('visualization', [])
    for key in ('acoustic', 'training', 'visualization'):
        if spec[key] not in ('local', 'notebook'):
            raise ValueError(f'execution.{key} must be local or notebook')
    if spec.get('preparation', 'standard') not in ('standard', 'mock'):
        raise ValueError('execution.preparation must be standard or mock')
    if spec.get('preparation') == 'mock' and any(spec[key] != 'local' for key in ('acoustic', 'training', 'visualization')):
        raise ValueError('mock preparation requires local execution')
    safe_id(spec['run_id'])
    if not isinstance(spec['training_device'], str) or not spec['training_device']:
        raise ValueError('training_device must be explicit')
    notebook = config['notebooks']
    if not isinstance(notebook, dict) or set(notebook) != {'install_dependencies'} or type(notebook['install_dependencies']) is not bool:
        raise ValueError('notebooks.install_dependencies must be boolean')
    ids = set()
    for group in ('analysis', 'visualization'):
        jobs = config[group]
        if not isinstance(jobs, list):
            raise ValueError(f'{group} must be a list of configured workflows')
        for job in jobs:
            if not isinstance(job, dict) or set(job) != {'id', 'workflow', 'parameters'}:
                raise ValueError('workflow requires id, workflow and parameters')
            safe_id(job['id'])
            if job['id'] in ids or job['workflow'] not in WORKFLOWS or not isinstance(job['parameters'], dict):
                raise ValueError('invalid or duplicate workflow')
            if (group == 'visualization') != job['workflow'].endswith('-visualization'):
                raise ValueError('workflow is in the wrong config group')
            ids.add(job['id'])
            from .workflows import resolve_parameters
            params = job['parameters']
            if job['workflow'] == 'training-report' and 'training_runs' not in params:
                params = {**params, 'training_runs': training_run_paths(config)}
            if job['workflow'] in ('dimsim-distribution', 'reference-geometry'):
                evaluation = config.get('evaluation')
                if 'evaluation_root' not in params and evaluation is not None:
                    params = {'evaluation_root': str(Path(evaluation['output_root']) / evaluation['id']), **params}
            if job['workflow'] == 'dimsim-parser-alignment' and not {'experiment_root', 'evaluation_root'} & params.keys():
                params = {**params, 'experiment_root': str(Path(config['paths']['experiments']) / safe_id(config['experiment_id']))}
            if job['workflow'] == 'reference-geometry' and 'base_model' in params:
                params = {'zh_cn_root': config['paths']['acoustic'], **params}
            job['parameters'] = resolve_parameters(job['workflow'], params, resolve)


def safe_id(value):
    from .source_preparation.archive import WINDOWS_FORBIDDEN, WINDOWS_DEVICES
    if (not isinstance(value, str) or not value or value in {'.', '..'}
            or any(c in WINDOWS_FORBIDDEN for c in value) or value.endswith(('.', ' '))
            or value.split('.')[0].casefold() in WINDOWS_DEVICES):
        raise ValueError('run IDs must be portable path components')
    return value


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.state-', dir=path.parent)
    os.close(fd)
    try:
        write_json(name, state)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _state_identity(config):
    if config['schema_version'] != 2:
        raise ValueError('orchestration requires configuration schema 2')
    return {'config': config, 'sdk_sha256': content_hash(sdk_hashes()),
            'acoustic_source_sha256': (_manifest_pin(config['acoustic']['existing_bundle'])
                                        if config['acoustic']['existing_bundle'] else None)}


def load_state(config, *, _allow_pending_training_sdk=False, _allow_stale_sdk=False):
    identity = _state_identity(config)
    path = Path(config['paths']['state'])
    if path.exists():
        state = _read(path)
        stored = state.get('identity', {})
        pending = state.get('waiting') or {}
        # Import can finish an already issued training job after an SDK upgrade.
        accepts_upgrade = _allow_stale_sdk or (_allow_pending_training_sdk and isinstance(pending, dict)
                                               and pending.get('stage') == 'training')
        upgrade = (accepts_upgrade
                   and isinstance(stored, dict)
                   and isinstance(stored.get('sdk_sha256'), str)
                   and {**stored, 'sdk_sha256': identity['sdk_sha256']} == identity)
        if state.get('schema_version') != 1 or (stored != identity and not upgrade):
            raise ValueError('configuration or SDK changed; select a new state path/run ID')
        return state
    return {'schema_version': 1, 'identity': identity, 'completed': {}, 'waiting': None}


def _manifest_pin(path):
    return sha256_file(Path(path) / 'manifest.json')


def training_run_paths(config):
    root = (Path(config['paths']['runs']) / safe_id(config['experiment_id']))
    return {model: str(root / safe_id(model) / config['execution']['run_id'])
            for model in training_model_ids(config)}


def training_model_ids(config):
    return [*config['experiment']['model_ids'], *config['execution'].get('additional_model_ids', [])]


def completed_manifest(state, key, path):
    record = state['completed'].get(key)
    if record is None:
        raise ValueError(f'{key} must finish before it can be used as an input')
    if Path(record['path']).resolve() != Path(path).resolve():
        raise ValueError(f'{key} path differs from the configured path')
    if _manifest_pin(path) != record['manifest_sha256']:
        raise ValueError(f'{key} manifest differs from the completed stage')
    return record['manifest_sha256']


def evaluation_inputs(config, state=None):
    if 'evaluation' not in config:
        raise ValueError('prepare-evaluation requires the evaluation config section')
    spec = config['evaluation']
    if spec['experiment_manifest_sha256'] is not None:
        return spec
    state = load_state(config) if state is None else state
    return {**spec, 'experiment_manifest_sha256': completed_manifest(
        state, 'prepare-experiment', spec['experiment_root'])}


def prepare_configured_evaluation(config, state=None):
    from .evaluation.geometry_bundle import prepare_geometry_bundle, validate_geometry_bundle

    managed = 'evaluation' in config and config['evaluation']['experiment_manifest_sha256'] is None
    if managed and state is None:
        state = load_state(config)
    spec = evaluation_inputs(config, state)
    if not managed:
        return prepare_geometry_bundle(spec)
    target = Path(spec['output_root']) / spec['id']
    key = 'prepare-evaluation'
    if key in state['completed']:
        completed_manifest(state, key, target)
        validate_geometry_bundle(target)
        return target
    result = prepare_geometry_bundle(spec)
    state['completed'][key] = {'path': str(result), 'manifest_sha256': _manifest_pin(result)}
    _write_state(config['paths']['state'], state)
    return result


def preflight_copy(inputs, destination):
    """Reject overlap/symlinks before any destination directory is created."""
    destination = Path(destination).resolve()
    for path in inputs:
        source = Path(path)
        if source.resolve() == destination or source.resolve() in destination.parents:
            raise ValueError('handoff destination must be outside its input tree')
        if source.is_symlink() or any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('handoff inputs cannot contain symlinks')


def _copy_tree(source, destination, *, ignore=None):
    preflight_copy([source], destination)
    shutil.copytree(source, destination, ignore=ignore)


def _training_job_inputs(config, experiment, model_id, experiment_pin, sources):
    return {
        'stage': 'training',
        'parameters': {'model_id': model_id, 'device': config['execution']['training_device'],
                       'run_id': config['execution']['run_id']},
        'sources': sources, 'input_manifest_sha256': experiment_pin, 'external_input': True,
        'input_config': _read(Path(experiment) / 'config.json'),
        'input_reference': str(Path(experiment).resolve()), 'experiment_id': config['experiment_id'],
    }


def _existing_training_job(config, experiment, model_id, experiment_pin, sources):
    from .notebooks import training_notebook_matches

    identity = _training_job_inputs(config, experiment, model_id, experiment_pin, sources)
    job_id = content_hash(identity)
    root = Path(config['paths']['generated'])
    path = root / 'jobs' / job_id
    notebook = root / 'notebooks' / f'training-{job_id[:16]}.ipynb'
    if not path.exists() or not notebook.is_file():
        return None
    job = validate_job(path, expected_sources=sources)
    if job['job_id'] != job_id:
        raise ValueError('existing training job differs from the requested job')
    if not training_notebook_matches(notebook, job, config['notebooks']['install_dependencies']):
        return None
    return {'job_id': job_id, 'stage': 'training', 'path': str(path), 'notebook': str(notebook),
            'model_id': model_id, 'workflow_id': None, 'sdk_sha256': content_hash(sources)}


def _pending_sdk(state):
    pending = state['waiting']
    if pending['stage'] == 'training' and 'sdk_sha256' in pending:
        return pending['sdk_sha256']
    return state['identity']['sdk_sha256']


def refresh_training_sdk(config, state, jobs):
    pending = state['waiting']
    if pending is not None:
        old_job = validate_job(pending['path'], expected_sdk_sha256=_pending_sdk(state))
        if old_job['stage'] != 'training' or old_job['job_id'] != pending['job_id']:
            raise ValueError('pending training job identity mismatch')
        replacement, = [entry for entry in jobs if entry['model_id'] == old_job['parameters']['model_id']]
        new_job = validate_job(replacement['path'])
        fields = ('parameters', 'input_manifest_sha256', 'experiment_id', 'input_config')
        if any(new_job[field] != old_job.get(field) for field in fields):
            raise ValueError('replacement job differs in training parameters or input experiment')
    else:
        _verify_completed_manifests(state)
    if load_state(config, _allow_stale_sdk=True) != state:
        raise ValueError('run state changed during notebook generation')
    if pending is not None:
        state['waiting'] = replacement
    state['identity'] = _state_identity(config)
    state.pop('previous_training_sources', None)
    _write_state(config['paths']['state'], state)


def create_job(config, stage, input_path, *, model_id=None, workflow=None, reference=None, _generation=None):
    """Publish an immutable input package; notebooks are generated beside it."""
    context = {} if _generation is None else _generation
    if 'sources' not in context:
        context['sources'] = sdk_files()
    sources = context['sources']
    if stage == 'acoustic':
        from .sound_inventory import validate_sound_inventory
        validate_sound_inventory(input_path)
        parameters = {'model': config['model'] if reference is None else reference['parameters']['base_model']}
    elif stage == 'training':
        if model_id not in training_model_ids(config):
            raise ValueError('training objective is not configured for this run')
        from .experiment import validate_experiment
        key = str(Path(input_path).resolve())
        if key not in context.setdefault('experiments', {}):
            context['experiments'][key] = validate_experiment(input_path)
        from .architecture import normalize_architecture
        actual = _read(Path(input_path) / 'config.json')
        desired = dict(config['experiment'])
        actual['architecture'] = normalize_architecture(actual['architecture'])
        desired['architecture'] = normalize_architecture(desired['architecture'])
        manifest = _read(Path(input_path) / 'manifest.json')
        if actual != desired or manifest['identity']['experiment_id'] != config['experiment_id']:
            raise ValueError('experiment bundle differs from the configured experiment; prepare a new bundle first')
        parameters = {'model_id': model_id, 'device': config['execution']['training_device'],
                      'run_id': config['execution']['run_id']}
    elif stage == 'visualization':
        from .workflows import package_workflow
        parameters = {'workflow': workflow}
        # Workflow preparation validates all explicit inputs before publication.
        input_path = None
    else:
        raise ValueError('unsupported notebook stage')
    identity = {'stage': stage, 'parameters': parameters, 'sources': sdk_hashes(sources),
                'input_manifest_sha256': _manifest_pin(input_path) if input_path else None}
    if reference is not None:
        identity['reference_id'] = reference['id']
    if stage == 'training':
        identity = _training_job_inputs(config, input_path, model_id, identity['input_manifest_sha256'], identity['sources'])
    if workflow:
        from .workflows import workflow_pins
        identity['workflow_pins'] = workflow_pins(workflow)
    job_id = content_hash(identity)
    target = Path(config['paths']['generated']) / 'jobs' / job_id
    if input_path:
        inputs = [input_path]
    else:
        from .workflows import _inputs
        inputs = list(_inputs(workflow).values())
    # Include the notebook directory as well as the job: either may overlap an input.
    preflight_copy(inputs, target)
    preflight_copy(inputs, Path(config['paths']['generated']) / 'notebooks')
    if target.exists():
        validate_bundle(target, 'pipeline_job', identity=identity)
    else:
        with staging_directory(target) as temporary:
            if input_path and stage != 'training':
                _copy_tree(input_path, temporary / 'input')
            elif stage == 'visualization':
                package_workflow(workflow, temporary)
            for name, body in ({} if stage == 'training' else sources).items():
                path = temporary / 'sdk' / 'phonetics' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding='utf-8', newline='\n')
            write_json(temporary / 'job.json', {'job_id': job_id, **identity})
            seal_bundle(temporary, 'pipeline_job', identity)
    from .notebooks import render_job_notebook
    notebook = render_job_notebook(target, install_dependencies=config['notebooks']['install_dependencies'], _sources=sources)
    return {'job_id': job_id, 'stage': stage, 'path': str(target), 'notebook': str(notebook),
            'model_id': model_id, 'workflow_id': workflow['id'] if workflow else None,
            **({'reference_id': reference['id']} if reference is not None else {})}


def validate_job(path, *, expected_sources=None, expected_sdk_sha256=None):
    root = Path(path)
    manifest = validate_bundle(root, 'pipeline_job')
    job = _read(root / 'job.json')
    identity = {k: v for k, v in job.items() if k != 'job_id'}
    if identity != manifest['identity'] or content_hash(identity) != job['job_id']:
        raise ValueError('job identity mismatch')
    if expected_sdk_sha256 is not None:
        if content_hash(identity['sources']) != expected_sdk_sha256:
            raise ValueError('job SDK differs from the pending state SDK')
    elif identity['sources'] != (sdk_hashes() if expected_sources is None else expected_sources):
        raise ValueError('job SDK differs from the executing SDK')
    if identity['input_manifest_sha256'] and not identity.get('external_input') and _manifest_pin(root / 'input') != identity['input_manifest_sha256']:
        raise ValueError('job input pin mismatch')
    return job


def execute_job(path, destination, *, allow_execution=False, input_path=None):
    """Called by an explicitly executed notebook cell or a local job command."""
    if not allow_execution:
        raise PermissionError('job execution requires explicit authorization')
    root = Path(path)
    job = validate_job(root)
    with staging_directory(destination) as temporary:
        if job['stage'] == 'acoustic':
            from .acoustic import prepare_acoustic_targets
            prepare_acoustic_targets(root / 'input', temporary / 'payload',
                                     model=job['parameters']['model'], cache_root=temporary / 'cache')
            # Cache is execution-local and not part of the exported result.
            if (temporary / 'cache').exists():
                shutil.rmtree(temporary / 'cache')
        elif job['stage'] == 'training':
            selected = Path(input_path) if input_path is not None else (Path(job['input_reference']) if job.get('external_input') else root / 'input')
            if _manifest_pin(selected) != job['input_manifest_sha256']:
                raise ValueError('selected experiment manifest does not match the job')
            from .training import train_experiment
            train_experiment(selected, runs_root=temporary / 'runs',
                             expected_manifest_sha256=job['input_manifest_sha256'], **job['parameters'])
        else:
            from .workflows import run_packaged_workflow
            run_packaged_workflow(root, temporary / 'payload')
        write_json(temporary / 'receipt.json', {'job_id': job['job_id'], 'stage': job['stage'],
                                               'input_manifest_sha256': job['input_manifest_sha256']})
        seal_bundle(temporary, 'pipeline_result', {'job_id': job['job_id']})
    return Path(destination)


def validate_training_result(root, job):
    """Authenticate opaque checkpoint bytes and JSON contracts without Torch."""
    from .training_artifacts import (
        COMPACT_SCHEMA, CONTRACT_SOURCE, LEGACY_SCHEMA, TRAINING_RUN_SCHEMAS, required_training_artifacts,
    )

    root = Path(root)
    manifest = _read(root / 'manifest.json')
    params = job['parameters']
    schema = manifest.get('schema_version', LEGACY_SCHEMA)
    if type(schema) is not int or schema not in TRAINING_RUN_SCHEMAS:
        raise ValueError('unsupported training result schema')
    if (manifest.get('run_type') != 'training_run' or manifest.get('status') != 'complete'
            or manifest.get('model_id') != params['model_id'] or manifest.get('run_id') != params['run_id']
            or manifest.get('experiment_hashes', {}).get('manifest.json') != job['input_manifest_sha256']):
        raise ValueError('training result does not match the job')
    training = manifest.get('config', {}).get('training', {})
    if 'epoch_limit' in manifest or 'epochs_by_model' in training:
        from .experiment import training_epoch_limit
        epoch_limit = manifest.get('epoch_limit')
        if (type(epoch_limit) is not int or epoch_limit <= 0
                or epoch_limit != training_epoch_limit(manifest['config'], params['model_id'])):
            raise ValueError('training result epoch limit differs from the experiment')
    seeds = manifest.get('seeds')
    if not isinstance(seeds, list) or not seeds or manifest.get('seed_status') != {str(s): 'complete' for s in seeds}:
        raise ValueError('training seeds are incomplete')
    if 'sources' in job:
        codes = manifest.get('code_sha256', {})
        required = {'training.py', 'model.py', 'metrics.py', 'architecture.py', 'experiment.py'}
        job_schema = COMPACT_SCHEMA if CONTRACT_SOURCE in job['sources'] else LEGACY_SCHEMA
        if schema != job_schema:
            raise ValueError('training result schema differs from the job')
        if job_schema == COMPACT_SCHEMA:
            required.add(CONTRACT_SOURCE)
        from .bert import is_bert
        from .architecture import BERT_REQUIREMENTS
        if is_bert(manifest.get('config', {'architecture': {}})):
            required.add('bert.py')
            for requirement in BERT_REQUIREMENTS:
                name, expected = requirement.split('==')
                if manifest.get('runtime', {}).get(name) != expected:
                    raise ValueError('training result BERT runtime differs from the job')
        if (not required <= set(codes) or any(job['sources'].get(name) != digest for name, digest in codes.items())
                or manifest.get('runtime', {}).get('torch', '').split('+', 1)[0] != '2.11.0'):
            raise ValueError('training result SDK/runtime differs from the job')
    artifacts = manifest.get('artifacts', {})
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root / 'manifest.json'}
    if not artifacts or set(artifacts) != actual or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('training result artifact set mismatch')
    required = required_training_artifacts(seeds, schema)
    if not required <= set(artifacts):
        raise ValueError('training result lacks required checkpoints or logs')
    if schema == COMPACT_SCHEMA and set(artifacts) != required:
        raise ValueError('compact training result contains unexpected artifacts')
    for name, digest in artifacts.items():
        if sha256_file(safe_artifact(root, name)) != digest:
            raise ValueError('training result artifact checksum mismatch')
    return manifest


def import_result(config, result_path):
    state = load_state(config, _allow_pending_training_sdk=True)
    pending = state['waiting']
    if not pending:
        raise ValueError('pipeline has no pending notebook result')
    job = validate_job(pending['path'], expected_sdk_sha256=_pending_sdk(state))
    if job['job_id'] != pending['job_id'] or job['stage'] != pending['stage']:
        raise ValueError('job identity differs from the pending state')
    root = Path(result_path)
    validate_bundle(root, 'pipeline_result', identity={'job_id': job['job_id']})
    receipt = _read(root / 'receipt.json')
    if receipt != {'job_id': job['job_id'], 'stage': job['stage'], 'input_manifest_sha256': job['input_manifest_sha256']}:
        raise ValueError('result receipt does not match the pending job')
    payload = root / 'payload'
    if job['stage'] == 'acoustic':
        from .acoustic import validate_acoustic_targets
        manifest = validate_acoustic_targets(payload)
        expected = {'inventory_manifest_sha256': job['input_manifest_sha256'], 'model': job['parameters']['model'],
                    'contract': 'acoustic-targets-1', 'backend': 'transformers'}
        if manifest['identity'] != expected:
            raise ValueError('acoustic result parameters differ from the job')
        if 'reference_id' in job:
            reference, = [item for item in config['analysis'] if item['id'] == job['reference_id']
                          and item['workflow'] == 'reference-geometry' and 'base_model' in item['parameters']]
            target = Path(reference['parameters']['base_root'])
            key = 'reference-acoustic:' + reference['id']
        else:
            target = Path(config['paths']['acoustic'])
            key = 'prepare-embeddings'
    elif job['stage'] == 'training':
        expected_config = job['input_config'] if job.get('external_input') else _read(Path(pending['path']) / 'input' / 'config.json')
        experiment_id = job['experiment_id'] if job.get('external_input') else _read(Path(pending['path']) / 'input' / 'manifest.json')['identity']['experiment_id']
        if (root / 'runs').exists():
            if payload.exists():
                raise ValueError('training result contains both runs and legacy payload')
            parts = (experiment_id, job['parameters']['model_id'], job['parameters']['run_id'])
            payload = root / 'runs'
            for part in parts:
                payload = payload / safe_id(part)
        manifest = validate_training_result(payload, job)
        if (content_hash(manifest['config']) != content_hash(expected_config)
                or manifest['seeds'] != list(range(expected_config['seed_count']))
                or manifest['experiment_id'] != experiment_id):
            raise ValueError('training result config or seeds differ from the experiment')
        target = Path(config['paths']['runs']) / manifest['experiment_id'] / manifest['model_id'] / manifest['run_id']
        key = 'train:' + manifest['model_id']
    else:
        from .workflows import validate_workflow_result, workflow_contract
        manifest = validate_workflow_result(payload)
        identity = manifest['identity']
        original = job['parameters']['workflow']
        received = identity.get('workflow', {})
        if (received.get('id') != original['id'] or received.get('workflow') != original['workflow']
                or workflow_contract(received) != workflow_contract(original)
                or identity.get('input_pins') != job.get('workflow_pins')
                or identity.get('source_sha256') != content_hash(job['sources'])):
            raise ValueError('workflow result provenance differs from the pending job')
        target = Path(config['paths']['reports']) / pending['workflow_id']
        key = 'visualization:' + pending['workflow_id']
    if target.exists():
        def hashes(directory):
            if any(p.is_symlink() for p in directory.rglob('*')):
                raise ValueError('result destination must not contain symlinks')
            return {p.relative_to(directory).as_posix(): sha256_file(p) for p in directory.rglob('*') if p.is_file()}
        if target.is_symlink() or hashes(target) != hashes(payload):
            raise ValueError('result destination already contains different data')
    else:
        with staging_directory(target) as temporary:
            shutil.copytree(payload, temporary, dirs_exist_ok=True)
    state['completed'][key] = {'path': str(target), 'manifest_sha256': _manifest_pin(target), 'job_id': job['job_id']}
    state['waiting'] = None
    identity = _state_identity(config)
    if job['stage'] == 'training' and state['identity']['sdk_sha256'] != identity['sdk_sha256']:
        state.setdefault('previous_training_sources', job['sources'])
    state['identity'] = identity
    _write_state(config['paths']['state'], state)
    return target


def _prepare_reference_acoustic(config, state, workflow, allow_inference, progress=None):
    from .acoustic import prepare_acoustic_targets, validate_acoustic_targets

    params = workflow['parameters']
    target = Path(params['base_root'])
    inventory = config['paths']['inventory']
    key = 'reference-acoustic:' + workflow['id']
    if key in state['completed']:
        completed_manifest(state, key, target)
        validate_acoustic_targets(target)
        return None
    if not target.exists():
        if config['execution']['acoustic'] == 'notebook':
            with _progress_step(progress, 'Generate Wav2Vec2 Base notebook'):
                state['waiting'] = create_job(config, 'acoustic', inventory, reference=workflow)
                _write_state(config['paths']['state'], state)
            return {'status': 'waiting', **state['waiting']}
        if not allow_inference:
            raise PermissionError('reference acoustic inference requires --allow-inference')
    with _progress_step(progress, 'Prepare Wav2Vec2 Base targets'):
        path = prepare_acoustic_targets(inventory, target, model=params['base_model'], cache_root=config['paths']['cache'])
        state['completed'][key] = {'path': str(path), 'manifest_sha256': _manifest_pin(path)}
        _write_state(config['paths']['state'], state)
    return None


def _run_pipeline_workflow(config, state, group, workflow, progress=None):
    from .workflows import _inputs, run_workflow, validate_workflow_result, workflow_pins

    key = group + ':' + workflow['id']
    completed = {Path(record['path']).resolve(): record['manifest_sha256'] for record in state['completed'].values()}
    paths = {name: Path(path).resolve() for name, path in _inputs(workflow).items()}
    expected = {name: completed[path] for name, path in paths.items() if path in completed}
    if key in state['completed']:
        manifest = validate_workflow_result(state['completed'][key]['path'])
        pins = workflow_pins(workflow)
        if (manifest['identity']['input_pins'] != pins
                or any(pins.get(name) != digest for name, digest in expected.items())):
            raise ValueError('completed workflow inputs changed')
        return None
    if group == 'visualization' and config['execution']['visualization'] == 'notebook':
        with _progress_step(progress, f"Generate notebook for {workflow['id']}"):
            state['waiting'] = create_job(config, 'visualization', None, workflow=workflow)
            _write_state(config['paths']['state'], state)
        return {'status': 'waiting', **state['waiting']}
    with _progress_step(progress, f"Report {workflow['id']}"):
        path = run_workflow(workflow, config['paths']['reports'], expected_input_pins=expected)
        state['completed'][key] = {'path': str(path), 'manifest_sha256': _manifest_pin(path)}
        _write_state(config['paths']['state'], state)
    return None


def _verify_completed_manifests(state):
    for record in state['completed'].values():
        if _manifest_pin(record['path']) != record['manifest_sha256']:
            raise ValueError('a completed stage manifest changed')


def extend_training_state(config):
    """Adopt added training objectives while retaining the pinned experiment and completed runs."""
    path = Path(config['paths']['state'])
    state = _read(path)
    old = state['identity']['config']
    if old == config:
        return load_state(config, _allow_pending_training_sdk=True)
    previous = training_model_ids(old)
    models = training_model_ids(config)
    if (state.get('schema_version') != 1 or state['waiting'] is not None
            or models[:len(previous)] != previous or len(models) == len(previous)):
        raise ValueError('training extension requires an idle run and appended objectives')
    retained = copy.deepcopy(config)
    retained['execution'] = {key: value for key, value in retained['execution'].items() if key != 'additional_model_ids'}
    if 'additional_model_ids' in old['execution']:
        retained['execution']['additional_model_ids'] = old['execution']['additional_model_ids']
    old_reports = [job for job in old['analysis'] if job['workflow'] == 'training-report']
    new_reports = [job for job in config['analysis'] if job['workflow'] == 'training-report']
    for prior, current in zip(old_reports, new_reports, strict=True):
        expected = {'id': current['id'], 'workflow': 'training-report',
                    'parameters': {'training_runs': training_run_paths(config)}}
        if current != expected or prior['id'] == current['id']:
            raise ValueError('select a new training-report ID for the expanded inventory')
        retained['analysis'][config['analysis'].index(current)] = prior
    if 'encoder_geometry' in old and 'encoder_geometry' in config:
        prior, current = old['encoder_geometry'], config['encoder_geometry']
        if prior['id'] == current['id']:
            raise ValueError('select a new encoder_geometry ID for the expanded inventory')
        old_contrasts = prior.get('contrasts', [])
        if 'contrasts' not in prior and 'contrasts' in current:
            from .research.encoder_geometry import DEFAULT_CONTRASTS
            old_contrasts = [list(pair) for pair in DEFAULT_CONTRASTS if set(pair) <= set(previous)]
        new_contrasts = current.get('contrasts', [])
        if (new_contrasts[:len(old_contrasts)] != old_contrasts
                or any(set(pair) <= set(previous) for pair in new_contrasts[len(old_contrasts):])):
            raise ValueError('append only contrasts involving additional objectives')
        geometry = {**current, 'id': prior['id'], 'training_runs': prior['training_runs']}
        if 'contrasts' in prior:
            geometry['contrasts'] = old_contrasts
        else:
            geometry.pop('contrasts', None)
        retained['encoder_geometry'] = geometry
    identity = _state_identity(config)
    if retained != old or identity['acoustic_source_sha256'] != state['identity']['acoustic_source_sha256']:
        raise ValueError('training extension changes existing pipeline inputs or settings')
    _verify_completed_manifests(state)
    experiment = Path(config['paths']['experiments']) / config['experiment_id']
    pin = completed_manifest(state, 'prepare-experiment', experiment)
    for model in previous:
        root = training_run_paths(old)[model]
        completed_manifest(state, 'train:' + model, root)
        validate_training_result(root, {'parameters': {'model_id': model, 'run_id': config['execution']['run_id']},
                                        'input_manifest_sha256': pin})
    for report in old_reports:
        state['completed'].pop('analysis:' + report['id'], None)
    state['completed'].pop('encoder-geometry', None)
    state.pop('previous_training_sources', None)
    state['identity'] = identity
    _write_state(path, state)
    return state


def run_pipeline(config, *, through='visualization', allow_inference=False, allow_training=False,
                 extend_training=False, progress=None):
    from .pipeline import run_stage
    if through not in ('experiment', 'training', 'analysis', 'visualization'):
        raise ValueError('invalid final pipeline stage')
    if progress is not None:
        progress(f"Experiment {config['experiment_id']}: checking run state")
    state = (extend_training_state(config) if extend_training
             else load_state(config, _allow_pending_training_sdk=True))
    if state['waiting']:
        pending = state['waiting']
        job = validate_job(pending['path'], expected_sdk_sha256=_pending_sdk(state))
        if (job['job_id'], job['stage']) != (pending['job_id'], pending['stage']):
            raise ValueError('job identity differs from the pending state')
        if job['stage'] == 'training':
            from .notebooks import render_job_notebook, training_notebook_matches
            install = config['notebooks']['install_dependencies']
            if not training_notebook_matches(pending['notebook'], job, install):
                sources = sdk_files()
                if job['sources'] != sdk_hashes(sources):
                    raise ValueError('pending training notebook is unusable and requires its original SDK; '
                                     'a completed result can still be imported')
                with _progress_step(progress, f"Restore training notebook ({pending['model_id']})"):
                    pending['notebook'] = str(render_job_notebook(
                        pending['path'], install_dependencies=install, _sources=sources))
                    _write_state(config['paths']['state'], state)
        return {'status': 'waiting', **state['waiting']}
    _verify_completed_manifests(state)
    if progress is not None and state['completed']:
        progress(f"Verified manifests for {len(state['completed'])} recorded stages")
    preparation = PREPARATION[-2:] if config['execution'].get('preparation') == 'mock' else PREPARATION
    verified_paths = set()
    for stage in preparation:
        if stage in state['completed']:
            path = Path(state['completed'][stage]['path'])
            if path not in verified_paths:
                kind = _read(path / 'manifest.json')['bundle_type']
                if kind == 'sound_source':
                    from .source_preparation.download import validate_source_bundle
                    validate_source_bundle(path)
                else:
                    validate_bundle(path, kind)
                verified_paths.add(path)
            continue
        if stage == 'prepare-embeddings' and config['acoustic']['existing_bundle'] is None and config['execution']['acoustic'] == 'notebook':
            with _progress_step(progress, 'Generate acoustic notebook'):
                state['waiting'] = create_job(config, 'acoustic', config['paths']['inventory'])
                _write_state(config['paths']['state'], state)
            return {'status': 'waiting', **state['waiting']}
        with _progress_step(progress, PREPARATION_LABELS[stage]):
            path = run_stage(config, stage, allow_inference=allow_inference)
            state['completed'][stage] = {'path': str(path), 'manifest_sha256': _manifest_pin(path)}
            _write_state(config['paths']['state'], state)
    if through == 'experiment':
        return {'status': 'complete', 'through': through, 'path': state['completed']['prepare-experiment']['path']}
    experiment = Path(config['paths']['experiments']) / config['experiment_id']
    experiment_pin = completed_manifest(state, 'prepare-experiment', experiment)
    models = training_model_ids(config)
    for index, model_id in enumerate(models, start=1):
        key = 'train:' + model_id
        if key in state['completed']:
            job = {'parameters': {'model_id': model_id, 'run_id': config['execution']['run_id']},
                   'input_manifest_sha256': experiment_pin}
            validate_training_result(state['completed'][key]['path'], job)
            continue
        if config['execution']['training'] == 'notebook':
            sources = sdk_files()
            hashes = sdk_hashes(sources)
            previous = state.get('previous_training_sources')
            generation = {'sources': sources, 'experiments': {str(experiment.resolve()): True}}
            jobs = {}
            for remaining in models[index - 1:]:
                if 'train:' + remaining in state['completed']:
                    continue
                ready = _existing_training_job(config, experiment, remaining, experiment_pin, previous) if previous else None
                ready = ready or _existing_training_job(config, experiment, remaining, experiment_pin, hashes)
                if ready is None:
                    with _progress_step(progress, f'Generate training notebook ({remaining})'):
                        ready = create_job(config, 'training', experiment, model_id=remaining, _generation=generation)
                jobs[remaining] = ready
            state['waiting'] = jobs[model_id]
            _write_state(config['paths']['state'], state)
            return {'status': 'waiting', **state['waiting']}
        if not allow_training:
            raise PermissionError('local training requires --allow-training')
        from .training import train_experiment
        with _progress_step(progress, f'Train model {index}/{len(models)} ({model_id})'):
            result = train_experiment(experiment, model_id=model_id, runs_root=config['paths']['runs'],
                                      device=config['execution']['training_device'], run_id=config['execution']['run_id'],
                                      expected_manifest_sha256=experiment_pin)
            path = result['run_dir']
            state['completed'][key] = {'path': str(path), 'manifest_sha256': _manifest_pin(path)}
            _write_state(config['paths']['state'], state)
    for workflow in config['analysis']:
        if workflow['workflow'] == 'training-report':
            _run_pipeline_workflow(config, state, 'analysis', workflow, progress)
    if through == 'training':
        return {'status': 'complete', 'through': through}
    managed_evaluation = 'evaluation' in config and config['evaluation']['experiment_manifest_sha256'] is None
    if managed_evaluation:
        with _progress_step(progress if 'prepare-evaluation' not in state['completed'] else None,
                            'Build evaluation bundle'):
            prepare_configured_evaluation(config, state)
    for group in ('analysis', 'visualization'):
        for workflow in config[group]:
            if workflow['workflow'] == 'training-report':
                continue
            if workflow['workflow'] == 'reference-geometry' and 'base_model' in workflow['parameters']:
                waiting = _prepare_reference_acoustic(config, state, workflow, allow_inference, progress)
                if waiting:
                    return waiting
            waiting = _run_pipeline_workflow(config, state, group, workflow, progress)
            if waiting:
                return waiting
        if through == group:
            break
    if managed_evaluation and 'encoder_geometry' in config:
        geometry = config['encoder_geometry']
        if geometry.get('execution', 'local') == 'notebook':
            from .notebooks import generate_encoder_notebooks
            with _progress_step(progress, 'Generate encoder geometry notebook'):
                notebook, = generate_encoder_notebooks(config, state=state)
            return {'status': 'notebook-ready', **notebook}
        from .encoder_jobs import run_configured_report
        target = Path(config['paths']['reports']) / geometry['id']
        key = 'encoder-geometry'
        if key in state['completed']:
            completed_manifest(state, key, target)
            validate_bundle(target, 'encoder_geometry_report')
        else:
            with _progress_step(progress, 'Run encoder geometry'):
                path = run_configured_report(config, allow_execution=geometry.get('allow_extraction', False), state=state)
                state['completed'][key] = {'path': str(path), 'manifest_sha256': _manifest_pin(path)}
                _write_state(config['paths']['state'], state)
    return {'status': 'complete', 'through': through}
