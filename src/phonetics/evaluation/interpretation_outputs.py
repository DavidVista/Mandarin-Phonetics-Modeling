"""Immutable numerical interpretation artifacts; no figures or model runtime."""
from pathlib import Path

from .evaluation_outputs import _evaluation_run, _write, _verify_inventory
from .interpretation import readable_tables


def write_interpretation_outputs(project_root, run_id, result, *, input_roots, source_hashes):
    """Persist a pure analysis result with explicit input identities and code pins.

    Callers must hash-verify source artifacts before building the result. Input
    manifests and supplied source hashes are preserved by the shared lifecycle.
    """
    tables = readable_tables(result)
    expected = {'config.json', 'data/interpretation.json', 'diagnostics/summary.json'}
    expected.update('tables/' + name for name in tables)
    expected.update('data/top-50-' + model + '.json' for model in result['top_pairs'])
    with _evaluation_run(project_root, 'contextual-interpretation', run_id,
                         input_roots=input_roots, config=result['config'], source_hashes=source_hashes,
                         expected_files=expected, scientific_pins={
                             'bundle_manifest_sha256': result['bundle_manifest_sha256'],
                             'model_seed_provenance': result['provenance']}) as root:
        _write(root / 'data/interpretation.json', result)
        _write(root / 'diagnostics/summary.json', result['diagnostics'])
        for model, table in result['top_pairs'].items():
            _write(root / f'data/top-50-{model}.json', table)
        for name, text in tables.items():
            with (root / 'tables' / name).open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
    _verify_inventory(Path(root), 'contextual-interpretation', run_id, expected)
    return root
