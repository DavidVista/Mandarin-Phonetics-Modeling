"""Derived Q12/9/3 tables and residual Q–Q data, without rendering."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .anchor_q_reporting import build_anchor_q_report, EDGE_ORDER, _edge_ids
from .interpretation import MODELS, SEEDS, readable_tables


def build_q_group_tables(batch):
    """Equal transition weights; MC precision retains matching seed draws."""
    report = build_anchor_q_report(batch, reference_model='syllable')
    if {r['model_id'] for r in batch['results']} != set(MODELS):
        raise ValueError('Q groups require the five learned models')
    arrays = {}
    for row in batch['results']:
        _, edges = _edge_ids(row['score']['draw_scores'], list(EDGE_ORDER))
        arrays[(row['model_id'], row['seed'])] = {e: np.asarray(v, dtype=float) for e, v in edges.items()}
    groups = {'all12': list(EDGE_ORDER),
              'partial9': [e for e in EDGE_ORDER if not e.endswith('->111')],
              'full3': [e for e in EDGE_ORDER if e.endswith('->111')]}
    seed_rows, summary, deltas, seed_deltas = [], [], [], []
    def estimate(values):
        means = [float(a.mean()) for a in values]
        se = (math.sqrt(sum(float(a.var(ddof=1)) / a.size for a in values)) / len(values)
              if all(a.size > 1 for a in values) else None)
        return float(np.mean(means)), se
    for group, edges in groups.items():
        for metric in ('initial', 'final', 'tone', 'overall'):
            selected = [e for e in edges if metric == 'overall' or e.startswith(metric + ':')]
            for model in MODELS:
                qs, diffs = [], []
                for seed in SEEDS:
                    q, se = estimate([arrays[(model, seed)][e] for e in selected])
                    delta, dse = estimate([arrays[(model, seed)][e] - arrays[('syllable', seed)][e] for e in selected])
                    qs.append(q); diffs.append(delta)
                    seed_rows.append({'group': group, 'metric': metric, 'model_id': model,
                                      'seed': seed, 'q': q, 'mc_se': se})
                    seed_deltas.append({'group': group, 'metric': metric, 'model_id': model,
                                        'seed': seed, 'delta_q': delta, 'mc_se': dse})
                # First average matched scores across seeds, then estimate MC SE.
                _, se = estimate([np.mean([arrays[(model, s)][e] for s in SEEDS], axis=0) for e in selected])
                _, dse = estimate([np.mean([arrays[(model, s)][e] - arrays[('syllable', s)][e]
                                          for s in SEEDS], axis=0) for e in selected])
                base = {'group': group, 'metric': metric, 'model_id': model,
                        'transition_count': len(selected), 'transition_weight': 1 / len(selected)}
                summary.append({**base, 'mean': float(np.mean(qs)), 'sample_sd': float(np.std(qs, ddof=1)),
                                'mc_se': se, 'seed_values': qs})
                deltas.append({**base, 'reference_model': 'syllable', 'mean': float(np.mean(diffs)),
                               'sample_sd': float(np.std(diffs, ddof=1)), 'mc_se': dse, 'seed_values': diffs})
    dispersions={(r['model_id'],r['edge_id']):r['sample_sd'] for r in report['tables']['model_edge_seed_summary']}
    forest=[{**r,'sample_sd':dispersions[(r['model_id'],r['edge_id'])]} for r in report['plot_data']['edges']]
    return {'config': {'groups': groups, 'weights': 'equal transitions within each group and metric',
                       'seed_ids': list(SEEDS), 'reference_model': 'syllable',
                       'mc_se': 'matched draw seed averages; stratified independent transitions',
                       'bootstrap': False, 'corpus_ci': False},
            'seed_metrics': seed_rows, 'summary': summary, 'paired_deltas': deltas,
            'seed_paired_deltas': seed_deltas, 'forest': forest}


def build_residual_qq_data(residuals, *, max_quantiles=401):
    """Per-seed sample-SD standardization; retain both extreme order statistics."""
    if type(max_quantiles) is not int or max_quantiles < 2:
        raise ValueError('max_quantiles must be an integer >= 2')
    if set(residuals) != {*MODELS, 'dimsim'}:
        raise ValueError('six residual outcomes are required')
    curves = []
    normal = NormalDist()
    size = None
    for model in (*MODELS, 'dimsim'):
        seeds = {None: residuals[model]} if model == 'dimsim' else residuals[model]
        if model != 'dimsim' and set(seeds) != {str(s) for s in SEEDS}:
            raise ValueError('exact five separate residual seeds are required')
        for seed, values in seeds.items():
            y = np.asarray(values, dtype=float)
            if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
                raise ValueError('finite nonempty residual vectors are required')
            if size is None: size = len(y)
            if len(y) != size: raise ValueError('residual populations differ')
            sd = float(np.std(y, ddof=1)) if len(y) > 1 else 0.
            reason = 'fewer_than_two_draws' if len(y) < 2 else 'constant_residual' if sd == 0 else None
            row = {'model_id': model, 'seed': None if seed is None else int(seed),
                   'draw_count': len(y), 'residual_mean': float(y.mean()), 'sample_sd': sd,
                   'status': reason or 'ok', 'normal_quantiles': [], 'standardized_residuals': []}
            if reason is None:
                idx = np.unique(np.rint(np.linspace(0, len(y)-1, min(len(y), max_quantiles))).astype(int))
                row['normal_quantiles'] = [normal.inv_cdf((int(i)+.5)/len(y)) for i in idx]
                row['standardized_residuals'] = ((np.sort(y)[idx]-y.mean()) / sd).tolist()
            curves.append(row)
    return {'config': {'max_quantiles': max_quantiles, 'plotting_positions': '(rank + 0.5) / n',
                       'standardization': 'within seed; sample SD', 'extremes_retained': True,
                       'sample': 'dimsim-balanced', 'draw_multiplicity': 'preserved'}, 'curves': curves}


def build_interpretation_presentation_data(q_batch, interpretation):
    """Prepare tables and plot data once; the renderer never refits outcomes."""
    if q_batch['bundle_manifest_sha256'] != interpretation['bundle_manifest_sha256']:
        raise ValueError('Q and residual bundle pins differ')
    q_pins = {(r['model_id'], r['seed']): {k: r[k] for k in
               ('run_id', 'run_manifest_sha256', 'checkpoint_sha256')} for r in q_batch['results']}
    expected = {(m, p['seed']): {k: p[k] for k in ('run_id', 'run_manifest_sha256', 'checkpoint_sha256')}
                for m, rows in interpretation['provenance'].items() for p in rows}
    if q_pins != expected or len(q_batch['results']) != 25:
        raise ValueError('Q and residual training provenance differ')
    q = build_q_group_tables(q_batch)
    tables = readable_tables(interpretation)
    for name, rows in [('anchor-q-groups', q['summary']), ('anchor-q-group-differences', q['paired_deltas'])]:
        lines = ['| Model | Group | Metric | Mean | Seed SD | MC SE |', '| --- | --- | --- | --- | --- | --- |']
        for row in rows:
            se = '—' if row['mc_se'] is None else f"{row['mc_se']:.6f}"
            lines.append(f"| {row['model_id']} | {row['group']} | {row['metric']} | {row['mean']:.6f} | {row['sample_sd']:.6f} | {se} |")
        tables[name + '.md'] = '\n'.join(lines) + '\n'
    return {'schema_version': 1, 'bundle_manifest_sha256': interpretation['bundle_manifest_sha256'],
            'provenance': interpretation['provenance'], 'q_groups': q,
            'regression_summary': interpretation['summary'], 'regression_per_seed': interpretation['per_seed'],
            'top_pairs': interpretation['top_pairs'],
            'residual_qq': build_residual_qq_data(interpretation['residuals']), 'tables': tables}


def write_interpretation_presentation_data(project_root, run_id, prepared, *, input_roots, source_hashes):
    """Persist derived data separately from numerical and presentation runs."""
    from .evaluation_outputs import _evaluation_run, _write
    expected = {'config.json', 'data/presentation_data.json', 'data/top_pairs.json', 'diagnostics/summary.json'}
    expected.update('tables/' + name for name in prepared['tables'])
    with _evaluation_run(project_root, 'contextual-interpretation-data', run_id, input_roots=input_roots,
            source_hashes=source_hashes, config={'q': prepared['q_groups']['config'], 'qq': prepared['residual_qq']['config']},
            expected_files=expected, scientific_pins={'evaluation_bundle_manifest_sha256': prepared['bundle_manifest_sha256'],
                                                     'model_seed_provenance': prepared['provenance']}) as root:
        _write(root / 'data/presentation_data.json', prepared)
        _write(root / 'data/top_pairs.json', prepared['top_pairs'])
        _write(root / 'diagnostics/summary.json', {'table_count': len(prepared['tables']),
                                                 'residual_curve_count': len(prepared['residual_qq']['curves'])})
        for name, body in prepared['tables'].items():
            if '/' in name or '\\' in name or not name.endswith('.md'):
                raise ValueError('unsafe table filename')
            with (root / 'tables' / name).open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(body)
    return root
