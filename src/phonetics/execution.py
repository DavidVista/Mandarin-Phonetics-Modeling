"""Local execution and content-bound notebook handoffs without eager ML imports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from .artifacts import sha256_file, write_json, safe_artifact
from .bundles import content_hash, seal_bundle, staging_directory, validate_bundle

PREPARATION = ('download-sounds', 'prepare-inventory', 'prepare-corpus',
               'prepare-embeddings', 'prepare-data', 'prepare-experiment')
WORKFLOWS = {'dimsim-distribution', 'dimsim-parser-alignment', 'reference-geometry',
             'training-visualization', 'evaluation-visualization'}


def sdk_files():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): p.read_text(encoding='utf-8')
            for p in sorted(root.rglob('*')) if p.suffix in {'.py', '.toml'}}


def sdk_hashes(files=None):
    import hashlib
    return {name: hashlib.sha256(body.encode('utf-8')).hexdigest() for name, body in (sdk_files() if files is None else files).items()}


def validate_execution_config(config, resolve):
    spec = config['execution']
    if not isinstance(spec, dict) or set(spec) != {'acoustic', 'training', 'visualization', 'training_device', 'run_id'}:
        raise ValueError('execution requires acoustic, training, visualization, training_device and run_id')
    for key in ('acoustic', 'training', 'visualization'):
        if spec[key] not in ('local', 'notebook'):
            raise ValueError(f'execution.{key} must be local or notebook')
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
            job['parameters'] = resolve_parameters(job['workflow'], job['parameters'], resolve)


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


def load_state(config):
    if config['schema_version'] != 2:
        raise ValueError('orchestration requires configuration schema 2')
    path = Path(config['paths']['state'])
    identity = {'config': config, 'sdk_sha256': content_hash(sdk_hashes()),
                'acoustic_source_sha256': (_manifest_pin(config['acoustic']['existing_bundle'])
                                            if config['acoustic']['existing_bundle'] else None)}
    if path.exists():
        state = _read(path)
        if state.get('schema_version') != 1 or state.get('identity') != identity:
            raise ValueError('configuration or SDK changed; select a new state path/run ID')
        return state
    return {'schema_version': 1, 'identity': identity, 'completed': {}, 'waiting': None}


def _manifest_pin(path):
    return sha256_file(Path(path) / 'manifest.json')


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


def create_job(config, stage, input_path, *, model_id=None, workflow=None, _generation=None):
    """Publish an immutable input package; notebooks are generated beside it."""
    context = {} if _generation is None else _generation
    if 'sources' not in context:
        context['sources'] = sdk_files()
    sources = context['sources']
    if stage == 'acoustic':
        from .sound_inventory import validate_sound_inventory
        validate_sound_inventory(input_path)
        parameters = {'model': config['model']}
    elif stage == 'training':
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
    if stage == 'training':
        identity.update(external_input=True, input_config=_read(Path(input_path) / 'config.json'),
                        input_reference=str(Path(input_path).resolve()),
                        experiment_id=manifest['identity']['experiment_id'])
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
            'model_id': model_id, 'workflow_id': workflow['id'] if workflow else None}


def validate_job(path, *, expected_sources=None):
    root = Path(path)
    manifest = validate_bundle(root, 'pipeline_job')
    job = _read(root / 'job.json')
    identity = {k: v for k, v in job.items() if k != 'job_id'}
    if identity != manifest['identity'] or content_hash(identity) != job['job_id']:
        raise ValueError('job identity mismatch')
    if identity['sources'] != (sdk_hashes() if expected_sources is None else expected_sources):
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
            result = train_experiment(selected, runs_root=temporary / 'runs',
                                      expected_manifest_sha256=job['input_manifest_sha256'], **job['parameters'])
            shutil.move(str(result['run_dir']), temporary / 'payload')
            shutil.rmtree(temporary / 'runs')
        else:
            from .workflows import run_packaged_workflow
            run_packaged_workflow(root, temporary / 'payload')
        write_json(temporary / 'receipt.json', {'job_id': job['job_id'], 'stage': job['stage'],
                                               'input_manifest_sha256': job['input_manifest_sha256']})
        seal_bundle(temporary, 'pipeline_result', {'job_id': job['job_id']})
    return Path(destination)


def validate_training_result(root, job):
    """Authenticate opaque checkpoint bytes and JSON contracts without Torch."""
    root = Path(root)
    manifest = _read(root / 'manifest.json')
    params = job['parameters']
    if (manifest.get('run_type') != 'training_run' or manifest.get('status') != 'complete'
            or manifest.get('model_id') != params['model_id'] or manifest.get('run_id') != params['run_id']
            or manifest.get('experiment_hashes', {}).get('manifest.json') != job['input_manifest_sha256']):
        raise ValueError('training result does not match the job')
    seeds = manifest.get('seeds')
    if not isinstance(seeds, list) or not seeds or manifest.get('seed_status') != {str(s): 'complete' for s in seeds}:
        raise ValueError('training seeds are incomplete')
    if 'sources' in job:
        codes = manifest.get('code_sha256', {})
        required = {'training.py', 'model.py', 'metrics.py', 'architecture.py', 'experiment.py'}
        if (not required <= set(codes) or any(job['sources'].get(name) != digest for name, digest in codes.items())
                or manifest.get('runtime', {}).get('torch', '').split('+', 1)[0] != '2.11.0'):
            raise ValueError('training result SDK/runtime differs from the job')
    artifacts = manifest.get('artifacts', {})
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p != root / 'manifest.json'}
    if not artifacts or set(artifacts) != actual or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('training result artifact set mismatch')
    required = {'summary.json'} | {f'seed-{seed}/{name}' for seed in seeds for name in
                                   ('best.pt', 'last.pt', 'shared_encoder.pt', 'summary.json', 'history.json', 'events.jsonl')}
    if not required <= set(artifacts):
        raise ValueError('training result lacks required checkpoints or logs')
    for name, digest in artifacts.items():
        if sha256_file(safe_artifact(root, name)) != digest:
            raise ValueError('training result artifact checksum mismatch')
    return manifest


def import_result(config, result_path):
    state = load_state(config)
    pending = state['waiting']
    if not pending:
        raise ValueError('pipeline has no pending notebook result')
    job = validate_job(pending['path'])
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
        target = Path(config['paths']['acoustic'])
        key = 'prepare-embeddings'
    elif job['stage'] == 'training':
        manifest = validate_training_result(payload, job)
        expected_config = job['input_config'] if job.get('external_input') else _read(Path(pending['path']) / 'input' / 'config.json')
        experiment_id = job['experiment_id'] if job.get('external_input') else _read(Path(pending['path']) / 'input' / 'manifest.json')['identity']['experiment_id']
        if (manifest['config'] != expected_config or manifest['seeds'] != list(range(expected_config['seed_count']))
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
    _write_state(config['paths']['state'], state)
    return target


def run_pipeline(config, *, through='visualization', allow_inference=False, allow_training=False):
    from .pipeline import run_stage
    if through not in ('experiment', 'training', 'analysis', 'visualization'):
        raise ValueError('invalid final pipeline stage')
    state = load_state(config)
    if state['waiting']:
        validate_job(state['waiting']['path'])
        return {'status': 'waiting', **state['waiting']}
    for record in state['completed'].values():
        if _manifest_pin(record['path']) != record['manifest_sha256']:
            raise ValueError('a completed stage manifest changed')
    for stage in PREPARATION:
        if stage == 'prepare-embeddings' and stage in state['completed']:
            # Imported results have no historical-bundle import identity.
            from .acoustic import validate_acoustic_targets
            validate_acoustic_targets(state['completed'][stage]['path'])
            continue
        if stage == 'prepare-embeddings' and config['acoustic']['existing_bundle'] is None and config['execution']['acoustic'] == 'notebook':
            state['waiting'] = create_job(config, 'acoustic', config['paths']['inventory'])
            _write_state(config['paths']['state'], state)
            return {'status': 'waiting', **state['waiting']}
        path = run_stage(config, stage, allow_inference=allow_inference)
        state['completed'][stage] = {'path': str(path), 'manifest_sha256': _manifest_pin(path)}
        _write_state(config['paths']['state'], state)
    if through == 'experiment':
        return {'status': 'complete', 'through': through, 'path': state['completed']['prepare-experiment']['path']}
    experiment = Path(config['paths']['experiments']) / config['experiment_id']
    for model_id in config['experiment']['model_ids']:
        key = 'train:' + model_id
        if key in state['completed']:
            job = {'parameters': {'model_id': model_id, 'run_id': config['execution']['run_id']},
                   'input_manifest_sha256': _manifest_pin(experiment)}
            validate_training_result(state['completed'][key]['path'], job)
            continue
        if config['execution']['training'] == 'notebook':
            state['waiting'] = create_job(config, 'training', experiment, model_id=model_id)
            _write_state(config['paths']['state'], state)
            return {'status': 'waiting', **state['waiting']}
        if not allow_training:
            raise PermissionError('local training requires --allow-training')
        from .training import train_experiment
        result = train_experiment(experiment, model_id=model_id, runs_root=config['paths']['runs'],
                                  device=config['execution']['training_device'], run_id=config['execution']['run_id'],
                                  expected_manifest_sha256=_manifest_pin(experiment))
        path = result['run_dir']
        state['completed'][key] = {'path': str(path), 'manifest_sha256': _manifest_pin(path)}
        _write_state(config['paths']['state'], state)
    if through == 'training':
        return {'status': 'complete', 'through': through}
    from .workflows import run_workflow
    for group in ('analysis', 'visualization'):
        for workflow in config[group]:
            key = group + ':' + workflow['id']
            if key in state['completed']:
                from .workflows import validate_workflow_result, workflow_pins
                manifest = validate_workflow_result(state['completed'][key]['path'])
                if manifest['identity']['input_pins'] != workflow_pins(workflow):
                    raise ValueError('completed workflow inputs changed')
                continue
            if group == 'visualization' and config['execution']['visualization'] == 'notebook':
                state['waiting'] = create_job(config, 'visualization', None, workflow=workflow)
                _write_state(config['paths']['state'], state)
                return {'status': 'waiting', **state['waiting']}
            path = run_workflow(workflow, config['paths']['reports'])
            state['completed'][key] = {'path': str(path), 'manifest_sha256': _manifest_pin(path)}
            _write_state(config['paths']['state'], state)
        if through == group:
            break
    return {'status': 'complete', 'through': through}
