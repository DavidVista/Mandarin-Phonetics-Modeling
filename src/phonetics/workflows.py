"""Config-driven adapters for existing artifact-only research workflows."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

from .artifacts import sha256_file, write_json, safe_artifact
from .bundles import content_hash, seal_bundle, staging_directory, validate_bundle

PATH_FIELDS = {
    'dimsim-distribution': {'evaluation_root'},
    'dimsim-parser-alignment': {'evaluation_root'},
    'reference-geometry': {'base_root', 'zh_cn_root', 'evaluation_root'},
    'training-visualization': set(),
    'evaluation-visualization': {'anchor_q_output', 'dimsim_output', 'random_control_output', 'bundle_path'},
}
PARAMETERS = {
    'dimsim-distribution': set(),
    'dimsim-parser-alignment': {'seed', 'bootstrap_replicates', 'confidence', 'atol', 'rtol'},
    'reference-geometry': {'tolerance'},
    'training-visualization': {'training_runs', 'summary_run_id', 'mode'},
    'evaluation-visualization': {'derived_run_ids', 'derived_mode'},
}


def resolve_parameters(name, parameters, resolve):
    paths = PATH_FIELDS[name]
    allowed = paths | PARAMETERS[name]
    if set(parameters) - allowed or not paths <= set(parameters):
        raise ValueError(f'invalid {name} parameters; allowed keys: {sorted(allowed)}')
    result = copy.deepcopy(parameters)
    for key in paths:
        result[key] = resolve(result[key])
    if name == 'training-visualization':
        if set(result) != allowed or result['mode'] not in ('prepare', 'reuse'):
            raise ValueError('training visualization requires training_runs, summary_run_id and mode')
        if result['mode'] == 'reuse':
            raise ValueError('config-driven training visualization currently supports prepare; use the SDK for explicit summary reuse')
        from .execution import safe_id
        safe_id(result['summary_run_id'])
        if not isinstance(result['training_runs'], dict):
            raise ValueError('training_runs must map model IDs to explicit run directories')
        result['training_runs'] = {key: resolve(value) for key, value in result['training_runs'].items()}
    if name == 'evaluation-visualization':
        if not {'derived_run_ids', 'derived_mode'} <= set(result) or result['derived_mode'] != 'prepare':
            raise ValueError('config-driven evaluation visualization requires explicit derived_run_ids and prepare mode')
        if not isinstance(result['derived_run_ids'], dict) or set(result['derived_run_ids']) != {'interpretation', 'presentation_data'}:
            raise ValueError('derived_run_ids requires interpretation and presentation_data')
        from .execution import safe_id
        for value in result['derived_run_ids'].values():
            safe_id(value)
    return result


def _inputs(workflow):
    params = workflow['parameters']
    result = {k: params[k] for k in sorted(PATH_FIELDS[workflow['workflow']])}
    if workflow['workflow'] == 'training-visualization':
        result.update({'training:' + k: p for k, p in params['training_runs'].items()})
    return result


def workflow_contract(workflow):
    """Canonicalize only input locations; preserve every other configured value."""
    result = copy.deepcopy(workflow)
    try:
        name = result['workflow']
        params = result['parameters']
        for field in PATH_FIELDS[name]:
            if not isinstance(params[field], str) or not params[field]:
                raise ValueError('workflow input locations must be nonempty strings')
            params[field] = {'input': field}
        if name == 'training-visualization':
            if not isinstance(params['training_runs'], dict) or any(not isinstance(p, str) or not p for p in params['training_runs'].values()):
                raise ValueError('training input locations must be a path mapping')
            params['training_runs'] = {model: {'input': 'training:' + model} for model in params['training_runs']}
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid workflow result specification') from exc
    return result


def _verify_files(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    artifacts = manifest.get('artifacts')
    if isinstance(artifacts, list):
        artifacts = {r['path']: r['sha256'] for r in artifacts}
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('workflow input has no authenticated artifacts')
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('workflow inputs must not contain symlinks')
    for name, digest in artifacts.items():
        # Rendering training logs does not require checkpoint tensor files.
        if manifest.get('run_type') == 'training_run' and Path(name).name not in {'summary.json', 'history.json'}:
            continue
        if sha256_file(safe_artifact(root, name)) != digest:
            raise ValueError('workflow input artifact checksum mismatch')
    return manifest


def workflow_pins(workflow):
    pins = {}
    for name, path in _inputs(workflow).items():
        manifest = _verify_files(path)
        if manifest.get('bundle_type') == 'character_experiment':
            raise ValueError('existing research workflows do not accept character experiment bundles; character evaluation is separate')
        pins[name] = sha256_file(Path(path) / 'manifest.json')
    return pins


def package_workflow(workflow, destination):
    workflow_pins(workflow)
    from .execution import _copy_tree, preflight_copy
    preflight_copy(_inputs(workflow).values(), destination)
    copied = copy.deepcopy(workflow)
    for index, (name, path) in enumerate(_inputs(workflow).items()):
        relative = f'inputs/{index}'
        if name.startswith('training:'):
            _copy_tree(path, Path(destination) / relative, ignore=shutil.ignore_patterns('*.pt'))
        else:
            _copy_tree(path, Path(destination) / relative)
        if name.startswith('training:'):
            copied['parameters']['training_runs'][name.split(':', 1)[1]] = relative
        else:
            copied['parameters'][name] = relative
    write_json(Path(destination) / 'workflow.json', copied)


def _dispatch(workflow, project_root):
    from .execution import sdk_hashes
    name = workflow['workflow']
    params = copy.deepcopy(workflow['parameters'])
    if name in ('dimsim-distribution', 'dimsim-parser-alignment'):
        for key in PATH_FIELDS[name]:
            params[key] = Path(params[key])
            pin_name = key.removesuffix('_root') + '_manifest_sha256'
            params[pin_name] = sha256_file(params[key] / 'manifest.json')
        if name == 'dimsim-distribution':
            from .research.dimsim_distribution import DimsimDistributionConfig as Config, run_dimsim_distribution as run
        else:
            from .research.dimsim_parser_alignment import DimsimParserAlignmentConfig as Config, run_dimsim_parser_alignment as run
        return run(Config(project_root=Path(project_root), run_id=workflow['id'], **params))
    sources = sdk_hashes()
    params.update(source_hashes=sources, source_digest=content_hash(sources))
    if name == 'training-visualization':
        from .visualization.training import run_training_visualization
        params['training_runs'] = {model: {'path': path, 'manifest_sha256': sha256_file(Path(path) / 'manifest.json')}
                                   for model, path in params['training_runs'].items()}
        return run_training_visualization(project_root, workflow['id'], **params)
    from .visualization.evaluation_workflow import run_complete_evaluation_report
    params['bundle_manifest_sha256'] = sha256_file(Path(params['bundle_path']) / 'manifest.json')
    return run_complete_evaluation_report(project_root, workflow['id'], **params)


def validate_workflow_result(path):
    return validate_bundle(path, 'workflow_result')


def run_workflow(workflow, output_root):
    from .execution import sdk_hashes
    pins = workflow_pins(workflow)
    identity = {'workflow': workflow, 'input_pins': pins, 'source_sha256': content_hash(sdk_hashes())}
    target = Path(output_root) / workflow['id']
    for path in _inputs(workflow).values():
        if Path(path).resolve() == target.resolve() or Path(path).resolve() in target.resolve().parents:
            raise ValueError('workflow output must be outside input directories')
    if target.exists():
        validate_bundle(target, 'workflow_result', identity=identity)
        return target
    with staging_directory(target) as temporary:
        # Existing APIs write several related immutable outputs under outputs/.
        if workflow['workflow'] == 'reference-geometry':
            from .research.reference_geometry import run_reference_geometry
            run_reference_geometry(temporary, workflow['parameters'], pins)
        else:
            _dispatch(workflow, temporary)
        seal_bundle(temporary, 'workflow_result', identity)
    return target


def run_packaged_workflow(job_root, destination):
    root = Path(job_root)
    workflow = json.loads((root / 'workflow.json').read_text(encoding='utf-8'))
    workflow['parameters'] = resolve_parameters(workflow['workflow'], workflow['parameters'],
                                                lambda p: str(safe_artifact(root, p)))
    result = run_workflow(workflow, Path(destination).parent / 'workflow-output')
    shutil.move(str(result), destination)
    (Path(destination).parent / 'workflow-output').rmdir()
