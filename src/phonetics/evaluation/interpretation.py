"""Descriptive eight-mask fits on fixed balanced contextual pair scores.

No model execution, sampling, significance tests or plotting occurs here.
Repeated draws retain their original weight in every fit and correlation.
"""
from __future__ import annotations

import copy
import math
from typing import Iterable

import numpy as np

MODELS = ('syllable', 'components', 'acoustic',
          'components-acoustic-lambda-1', 'components-acoustic-lambda-10')
SEEDS = tuple(range(5))
MASKS = tuple(f'{i:03b}' for i in range(8))
LABELS = dict(zip(MODELS, ('Syllable', 'Components', 'Acoustic', 'Combined λ1', 'Combined λ10')))
CONFIG = {'sample': 'dimsim-balanced', 'mask_order': ['initial', 'final', 'tone'],
          'masks': list(MASKS), 'models': list(MODELS), 'seeds': list(SEEDS),
          'learned_outcome': 'cosine_similarity', 'reference_outcome': '-dimsim_distance',
          'fit': 'eight categorical means; no separate intercept', 'r2': 'centered',
          'draw_weights': 'original multiplicity', 'seed_summary': 'mean and sample SD',
          'top_limit': 50, 'top_ranking': 'positive raw residual; learned mean across seeds',
          'top_deduplication': 'unordered occurrence pair', 'inference': False,
          'bootstrap': False, 'p_values': False}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def categorical_fit(outcome, masks):
    """Return category means, raw residuals and centered descriptive R²."""
    y = np.asarray([_number(v, 'outcome') for v in outcome], dtype=np.float64)
    masks = list(masks)
    if not len(y) or len(y) != len(masks) or any(m not in MASKS for m in masks):
        raise ValueError('nonempty aligned outcomes and binary I/F/T masks are required')
    beta, counts, diagnostics = {}, {}, []
    residual = np.empty_like(y)
    for mask in MASKS:
        idx = np.array([m == mask for m in masks])
        counts[mask] = int(idx.sum())
        beta[mask] = float(y[idx].mean()) if counts[mask] else None
        if counts[mask]:
            residual[idx] = y[idx] - beta[mask]
            if np.ptp(y[idx]) == 0:
                diagnostics.append({'mask': mask, 'reason': 'constant_within_mask'})
        else:
            diagnostics.append({'mask': mask, 'reason': 'empty_category'})
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum(residual ** 2))
    if sst == 0:
        diagnostics.append({'reason': 'constant_outcome'})
    return {'beta': beta, 'counts': counts, 'r2': 1 - sse / sst if sst else None,
            'r2_status': 'ok' if sst else 'constant_outcome', 'sse': sse, 'sst': sst,
            'residuals': residual.tolist(), 'diagnostics': diagnostics}


def residual_spearman(left, right):
    """Average tied ranks; undefined constants remain null, without p-values."""
    a, b = (np.asarray([_number(v, 'residual') for v in values]) for values in (left, right))
    if len(a) != len(b) or not len(a):
        raise ValueError('nonempty aligned residual vectors are required')
    reason = ('fewer_than_two_draws' if len(a) < 2 else
              'constant_learned_residual' if np.ptp(a) == 0 else
              'constant_reference_residual' if np.ptp(b) == 0 else None)
    if reason:
        return {'rho': None, 'status': reason}
    def ranks(x):
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1) / 2)[inv]
    return {'rho': float(np.corrcoef(ranks(a), ranks(b))[0, 1]), 'status': 'ok'}


def _summary(values):
    available = [v for v in values if v is not None]
    complete = len(available) == len(SEEDS)
    return {'values': values, 'mean': float(np.mean(available)) if complete else None,
            'sample_sd': float(np.std(available, ddof=1)) if complete else None,
            'defined_seeds': len(available), 'status': 'ok' if complete else 'undefined_seed_values'}


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} must be a nonempty string')
    return value


def build_interpretation(records: Iterable[dict], occurrences: list[dict], sentences: list[dict], *,
                         bundle_manifest_sha256: str) -> dict:
    """Consume saved DIMSIM result records one at a time; retain compact arrays.

    The caller supplies hash-verified input records and bundle metadata. This pure
    API validates their identities and provenance but does not verify file bytes.
    """
    if (not isinstance(bundle_manifest_sha256, str) or len(bundle_manifest_sha256) != 64
            or any(c not in '0123456789abcdef' for c in bundle_manifest_sha256)):
        raise ValueError('invalid bundle manifest SHA-256')
    occ = {r['occurrence_id']: r for r in occurrences}
    contexts = {r['sentence_id']: r for r in sentences}
    if len(occ) != len(occurrences) or len(contexts) != len(sentences):
        raise ValueError('duplicate occurrence or sentence IDs')
    fits, provenance, canonical, metadata, reference = {}, {}, None, None, None
    pair_identity_map = {}
    for record in records:
        model, seed = record.get('model_id'), record.get('seed')
        if model not in MODELS or type(seed) is not int or seed not in SEEDS or (model, seed) in fits:
            raise ValueError('invalid or duplicate model/seed')
        score = record['score']
        if (score.get('model_id') != model or score.get('seed') != seed
                or type(score.get('seed')) is not int
                or score.get('bundle_manifest_sha256') != bundle_manifest_sha256):
            raise ValueError('score identity or bundle mismatch')
        pin = {k: _text(record.get(k), k) for k in ('run_id', 'run_manifest_sha256', 'checkpoint_sha256')}
        for field in ('run_manifest_sha256', 'checkpoint_sha256'):
            if len(pin[field]) != 64 or any(c not in '0123456789abcdef' for c in pin[field]):
                raise ValueError('invalid provenance hash')
        if model in provenance and any(pin[k] != provenance[model][0][k]
                                       for k in ('run_id', 'run_manifest_sha256')):
            raise ValueError('mixed training runs')
        provenance.setdefault(model, []).append({'seed': seed, **pin})
        rows = score['modes']['dimsim-balanced']['rows']
        if not rows:
            raise ValueError('balanced rows are required')
        ids, values, distances, repeated_scores = [], [], [], {}
        for row in rows:
            if row.get('analysis_status') != 'ok' or row.get('distance_status') != 'ok':
                raise ValueError('invalid balanced draw status; no silent filtering')
            mask = row.get('overlap_mask')
            if mask not in MASKS:
                raise ValueError('invalid overlap mask')
            identity = tuple(_text(row.get(k), k) for k in ('draw_id', 'pair_id', 'left_id', 'right_id'))
            if identity[2] == identity[3]:
                raise ValueError('self-comparison is not permitted')
            distance = _number(row.get('dimsim_distance'), 'DIMSIM distance')
            cosine = _number(row.get('cosine_similarity'), 'cosine similarity')
            if distance < 0 or not -1 <= cosine <= 1:
                raise ValueError('outcome outside valid range')
            pair_key = tuple(sorted(identity[2:]))
            if pair_key in repeated_scores and repeated_scores[pair_key] != cosine:
                raise ValueError('repeated occurrence pair has inconsistent scores')
            repeated_scores[pair_key] = cosine
            ids.append((*identity, mask, distance))
            values.append(cosine); distances.append(distance)
        if len({r[0] for r in ids}) != len(ids):
            raise ValueError('duplicate draw IDs')
        if canonical is None:
            canonical = ids
            metadata = []
            reverse_pairs = {}
            for draw_id, pair_id, left, right, mask, distance in ids:
                unordered = tuple(sorted((left, right)))
                if pair_id in pair_identity_map and pair_identity_map[pair_id] != unordered:
                    raise ValueError('pair ID collision')
                if unordered in reverse_pairs and reverse_pairs[unordered] != (pair_id, mask, distance):
                    raise ValueError('inconsistent repeated occurrence pair')
                pair_identity_map[pair_id] = unordered
                reverse_pairs[unordered] = (pair_id, mask, distance)
                ends = []
                for oid in (left, right):
                    if oid not in occ:
                        raise ValueError('missing occurrence metadata')
                    o = occ[oid]
                    sentence = contexts.get(o['sample_id'])
                    if sentence is None:
                        raise ValueError('missing sentence context')
                    position = o['character_position']
                    if (type(position) is not int or not 0 <= position < len(sentence['text'])
                            or sentence['text'][position] != o['character']):
                        raise ValueError('occurrence/context alignment mismatch')
                    ends.append({'occurrence_id': oid, 'sentence_id': o['sample_id'],
                                 'position': position, 'text': sentence['text'],
                                 'character': o['character'], 'pinyin': o['pinyin']})
                actual_mask = ''.join('1' if occ[left][k] == occ[right][k] else '0'
                                      for k in ('initial', 'final', 'tone'))
                if actual_mask != mask:
                    raise ValueError('mask disagrees with occurrence components')
                metadata.append({'draw_id': draw_id, 'pair_id': pair_id, 'mask': mask,
                                 'left': ends[0], 'right': ends[1]})
            reference = categorical_fit([-d for d in distances], [r[4] for r in ids])
        elif ids != canonical:
            raise ValueError('balanced draw identities, order, masks or distances differ')
        fits[(model, seed)] = categorical_fit(values, [r[4] for r in ids])
    if set(fits) != {(m, s) for m in MODELS for s in SEEDS}:
        raise ValueError('exact five models by five seeds are required')
    summary, per_seed, residuals, tops = [], [], {}, {}
    for model in (*MODELS, 'dimsim'):
        if model == 'dimsim':
            row = {'model': model, 'beta': reference['beta'], 'r2': reference['r2'],
                   'residual_spearman': None, 'seed_sd': None}
            raw = reference['residuals']
        else:
            selected = [fits[(model, s)] for s in SEEDS]
            correlations = [residual_spearman(f['residuals'], reference['residuals']) for f in selected]
            row = {'model': model, 'beta': {m: _summary([f['beta'][m] for f in selected]) for m in MASKS},
                   'r2': _summary([f['r2'] for f in selected]),
                   'residual_spearman': _summary([r['rho'] for r in correlations])}
            raw = np.mean([f['residuals'] for f in selected], axis=0).tolist()
            for seed, fit, corr in zip(SEEDS, selected, correlations):
                per_seed.append({'model': model, 'seed': seed, **{k: v for k, v in fit.items() if k != 'residuals'},
                                 'residual_spearman': corr})
            residuals[model] = {str(s): fits[(model, s)]['residuals'] for s in SEEDS}
        summary.append(row)
        seen, top = set(), []
        for index in sorted(range(len(raw)), key=lambda i: (-raw[i], metadata[i]['pair_id'], metadata[i]['draw_id'])):
            pair = metadata[index]
            identity = tuple(sorted((pair['left']['occurrence_id'], pair['right']['occurrence_id'])))
            if raw[index] <= 0 or identity in seen:
                continue
            seen.add(identity)
            if len(top) < 50:
                top.append({**pair, 'resid': raw[index]})
        tops[model] = {'rows': top, 'returned_count': len(top), 'positive_unique_pair_count': len(seen)}
    residuals['dimsim'] = reference['residuals']
    return {'schema_version': 1, 'config': copy.deepcopy(CONFIG), 'bundle_manifest_sha256': bundle_manifest_sha256,
            'provenance': {m: sorted(v, key=lambda p: p['seed']) for m, v in provenance.items()},
            'summary': summary, 'per_seed': per_seed, 'reference_fit': reference,
            'draw_metadata': metadata, 'residuals': residuals, 'top_pairs': tops,
            'diagnostics': {'draw_count': len(canonical), 'unique_pairs': len(pair_identity_map),
                            'mask_counts': reference['counts'], 'observations_independent': False}}


def readable_tables(result):
    """Format the six-row coefficient table and six strictly six-column tables."""
    def fmt(value):
        if isinstance(value, dict):
            return '—' if value['mean'] is None else f"{value['mean']:.6f} ± {value['sample_sd']:.6f}"
        return '—' if value is None else f'{value:.6f}'
    lines = ['| Model | ' + ' | '.join('β' + m for m in MASKS) + ' | R² | Residual Spearman vs DIMSIM |',
             '| ' + ' | '.join(['---'] * 11) + ' |']
    for row in result['summary']:
        cells = [LABELS.get(row['model'], 'DIMSIM')] + [fmt(row['beta'][m]) for m in MASKS]
        cells += [fmt(row['r2']), fmt(row['residual_spearman'])]
        lines.append('| ' + ' | '.join(cells) + ' |')
    tables = {'coefficients.md': '\n'.join(lines) + '\n'}
    def escape(s):
        return str(s).replace('|', '\\|').replace('\n', ' ')
    for model, table in result['top_pairs'].items():
        lines = ['| char1 | char2 | pinyin1 | pinyin2 | mask | resid |', '| --- | --- | --- | --- | --- | --- |']
        for row in table['rows']:
            cells = [row['left']['character'], row['right']['character'], row['left']['pinyin'],
                     row['right']['pinyin'], row['mask'], f"{row['resid']:.8f}"]
            lines.append('| ' + ' | '.join(map(escape, cells)) + ' |')
        tables[f'top-50-{model}.md'] = '\n'.join(lines) + f"\nReturned pairs: {table['returned_count']}.\n"
    return tables
