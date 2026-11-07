"""One artifact-only workflow: numerical interpretation, then six-figure report."""
from pathlib import Path

from ..evaluation.evaluation_outputs import _finite_json, _verify_inventory, load_evaluation_summary
from ..evaluation.interpretation_inputs import run_saved_interpretation, load_saved_q_batch
from ..evaluation.interpretation_reporting import build_interpretation_presentation_data, write_interpretation_presentation_data
from ..research.artifacts import safe_run_id, ensure_output_outside_inputs, file_sha256
from .evaluation import _check_paired_outputs, _guard_not_in_immutable_run
from .evaluation_report import write_evaluation_report
from .interpretation_report import load_interpretation_presentation_data


def run_complete_evaluation_report(project_root, presentation_id, *, anchor_q_output, dimsim_output,
        random_control_output, bundle_path, bundle_manifest_sha256, derived_run_ids,
        derived_mode, source_hashes, source_digest, protocol=None, resolved_config=None):
    """Prepare or explicitly reuse derived data; never perform inference/scoring."""
    if derived_mode not in {'prepare', 'reuse'}:
        raise ValueError('derived_mode must be prepare or reuse')
    if not isinstance(derived_run_ids, dict) or set(derived_run_ids) != {'interpretation', 'presentation_data'}:
        raise ValueError('two explicit derived run IDs are required')
    base = Path(project_root).resolve() / 'outputs'
    interp = base / 'contextual-interpretation' / safe_run_id(derived_run_ids['interpretation'])
    prepared_path = base / 'contextual-interpretation-data' / safe_run_id(derived_run_ids['presentation_data'])
    report_path = base / 'contextual-evaluation-report' / safe_run_id(presentation_id)
    inputs = [Path(p).resolve(strict=True) for p in (anchor_q_output,dimsim_output,random_control_output,bundle_path)]
    for target in ([report_path,interp,prepared_path] if derived_mode=='prepare' else [report_path]):
        ensure_output_outside_inputs(target, inputs)
        if target.exists(): raise FileExistsError(f'refusing to overwrite output: {target}')
        _guard_not_in_immutable_run(target)
    # Validate fonts and original paired outputs before creating numerical artifacts.
    from . import load_protocol, apply_style
    p = protocol or load_protocol(); apply_style(p)
    aq,ds = load_evaluation_summary(inputs[0]),load_evaluation_summary(inputs[1])
    _check_paired_outputs(aq,ds)
    if (file_sha256(inputs[3]/'manifest.json') != bundle_manifest_sha256
            or aq['manifest']['scientific_input_pins']['bundle_manifest_sha256'] != bundle_manifest_sha256):
        raise ValueError('explicit bundle identity differs from evaluation outputs')
    from ..evaluation.evaluation_outputs import load_random_control_summary
    random = load_random_control_summary(inputs[2])
    if random['manifest']['scientific_input_pins']['bundle_manifest_sha256'] != bundle_manifest_sha256:
        raise ValueError('random control bundle identity mismatch')
    if derived_mode=='prepare':
        run_saved_interpretation(dimsim_root=inputs[1],
            dimsim_manifest_sha256=file_sha256(inputs[1]/'manifest.json'),
            bundle_path=inputs[3],bundle_manifest_sha256=bundle_manifest_sha256,
            project_root=project_root,run_id=interp.name,source_hashes=source_hashes)
        interpretation = _finite_json(interp/'data/interpretation.json')
        q_batch = load_saved_q_batch(inputs[0])
        prepared = build_interpretation_presentation_data(q_batch,interpretation)
        write_interpretation_presentation_data(project_root,prepared_path.name,prepared,
            input_roots=[interp,inputs[0]],source_hashes=source_hashes)
        del q_batch,interpretation,prepared
    else:
        # Verify exact saved runs and their original inputs; no new numerical fit.
        im = _finite_json(interp/'manifest.json')
        _verify_inventory(interp,'contextual-interpretation',interp.name,set(im['required_artifacts']))
        interpretation = _finite_json(interp/'data/interpretation.json')
        saved = interpretation['config']['saved_inputs']
        if (saved['bundle_manifest_sha256'] != bundle_manifest_sha256
                or saved['dimsim_manifest_sha256'] != file_sha256(inputs[1]/'manifest.json')):
            raise ValueError('reused interpretation has different source inputs')
        _,pm = load_interpretation_presentation_data(prepared_path)
        hashes={r['sha256'] for r in pm['inputs']}
        if not {file_sha256(interp/'manifest.json'),file_sha256(inputs[0]/'manifest.json')} <= hashes:
            raise ValueError('reused prepared data has different source inputs')
        del interpretation
    return write_evaluation_report(project_root,presentation_id,
        anchor_q_output=inputs[0],dimsim_output=inputs[1],random_control_output=inputs[2],
        interpretation_data_output=prepared_path,source_hashes=source_hashes,
        source_digest=source_digest,protocol=p,resolved_config={**(resolved_config or {}),
            'derived_mode':derived_mode,'derived_run_ids':derived_run_ids,
            'bundle_path':str(inputs[3]),'bundle_manifest_sha256':bundle_manifest_sha256})
