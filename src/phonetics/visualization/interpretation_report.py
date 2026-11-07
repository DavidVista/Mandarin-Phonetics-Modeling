"""Export new interpretation tables and Residual Q–Q from immutable data."""
from __future__ import annotations

from pathlib import Path
import re
import matplotlib.pyplot as plt

from ..evaluation.evaluation_outputs import _finite_json, _verify_inventory, _evaluation_run, _write
from ..research.artifacts import file_sha256
from . import load_protocol, apply_style, export_figure
from .evaluation import _guard_not_in_immutable_run, MODEL_LABELS
from .interpretation import build_residual_qq


def load_interpretation_presentation_data(root):
    root = Path(root).resolve(strict=True)
    names = ['coefficients.md', 'anchor-q-groups.md', 'anchor-q-group-differences.md']
    names += [f'top-50-{m}.md' for m in (*MODEL_LABELS, 'dimsim')]
    expected = {'config.json', 'data/presentation_data.json', 'data/top_pairs.json', 'diagnostics/summary.json'}
    expected.update('tables/' + name for name in names)
    manifest = _verify_inventory(root, 'contextual-interpretation-data', root.name, expected)
    data = _finite_json(root / 'data/presentation_data.json')
    if (data.get('schema_version') != 1 or set(data.get('tables', {})) != set(names)
            or data.get('bundle_manifest_sha256') != manifest['scientific_input_pins'].get('evaluation_bundle_manifest_sha256')
            or data.get('provenance') != manifest['scientific_input_pins'].get('model_seed_provenance')):
        raise ValueError('interpretation presentation identity mismatch')
    for name, body in data['tables'].items():
        if (root / 'tables' / name).read_text(encoding='utf-8') != body:
            raise ValueError('readable tables disagree with prepared data')
    if _finite_json(root / 'data/top_pairs.json') != data['top_pairs']:
        raise ValueError('top-pair identities disagree with prepared data')
    return data, manifest


def write_interpretation_report(project_root, run_id, *, prepared_output, source_hashes, source_digest, protocol=None):
    """No fitting, scoring or standardization; read prepared outputs only."""
    if not isinstance(source_digest, str) or re.fullmatch(r'[0-9a-f]{64}',source_digest) is None:
        raise ValueError('explicit source digest is required')
    data, manifest = load_interpretation_presentation_data(prepared_output)
    p = protocol or load_protocol(); apply_style(p)
    target = Path(project_root).resolve() / 'outputs/contextual-interpretation-report' / run_id
    if target.exists(): raise FileExistsError('refusing to overwrite interpretation report')
    _guard_not_in_immutable_run(target)
    expected = {'config.json','data/top_pairs.json','diagnostics/summary.json'}
    expected.update('tables/'+name for name in data['tables'])
    expected.update('figures/residual-qq.'+ext for ext in ('png','eps','metadata.json'))
    config = {'source_digest':source_digest, 'prepared_manifest_sha256':file_sha256(Path(prepared_output)/'manifest.json'),
              'style_sha256':p.style_hash, 'registry_sha256':p.registry_hash}
    with _evaluation_run(project_root,'contextual-interpretation-report',run_id,
            input_roots=[prepared_output],config=config,source_hashes=source_hashes,
            expected_files=expected,scientific_pins=manifest['scientific_input_pins'],
            profile='contextual_interpretation_report',figure_exports_on_complete='complete') as root:
        for name, body in data['tables'].items():
            with (root/'tables'/name).open('x',encoding='utf-8',newline='\n') as stream: stream.write(body)
        _write(root/'data/top_pairs.json',data['top_pairs'])
        fig = build_residual_qq(data['residual_qq'],p)
        try:
            export_figure(fig,root/'figures','residual-qq','residual_qq',p,
                          scientific_provenance={'source_digest':source_digest,
                              'prepared_manifest_sha256':config['prepared_manifest_sha256']})
        finally: plt.close(fig)
        _write(root/'diagnostics/summary.json',{'figure_count':1,'table_count':len(data['tables']),
                                              'sample':'dimsim-balanced','refit':False})
    return root
