"""Synthetic formula, identity and reporting checks without model execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from phonetics.evaluation.interpretation import (
    MODELS, MASKS, categorical_fit, residual_spearman, build_interpretation, readable_tables,
)


def fixture(size=4):
    occurrences, sentences, rows = [], [], []
    for mask_index, mask in enumerate(MASKS):
        for j in range(size):
            left, right = f'l{mask}{j}', f'r{mask}{j}'
            for oid, char, components in [(left, '甲', ('a', 'b', '1')),
                    (right, '乙', tuple(v if bit == '1' else v + 'x'
                                       for bit, v in zip(mask, ('a', 'b', '1'))))]:
                occurrences.append(dict(occurrence_id=oid, sample_id=oid, character=char,
                                        character_position=0, pinyin='test1',
                                        **dict(zip(('initial', 'final', 'tone'), components))))
                sentences.append({'sentence_id': oid, 'text': char + '上下文'})
            rows.append(dict(draw_id=f'd{mask}{j}', pair_id=f'p{mask}{j}', left_id=left, right_id=right,
                overlap_mask=mask, dimsim_distance=float(10 + mask_index * 3 + j),
                cosine_similarity=.1 * mask_index - .03 * j,
                analysis_status='ok', distance_status='ok'))
    # Unequal multiplicity deliberately changes the category mean.
    rows.append({**rows[0], 'draw_id': 'repeat'})
    records = []
    for model in MODELS:
        for seed in range(5):
            draws = copy.deepcopy(rows)
            for r in draws:
                r['cosine_similarity'] += .001 * seed
            records.append(dict(model_id=model, seed=seed, run_id='fixed',
                run_manifest_sha256='b' * 64, checkpoint_sha256='c' * 64,
                score=dict(model_id=model, seed=seed, bundle_manifest_sha256='a' * 64,
                           modes={'dimsim-balanced': {'rows': draws}})))
    return records, occurrences, sentences


def build(data=None):
    records, occ, sentences = fixture() if data is None else data
    return build_interpretation(iter(records), occ, sentences, bundle_manifest_sha256='a' * 64)


class InterpretationTests(unittest.TestCase):
    def test_fit_against_independent_least_squares(self):
        masks = ['000', '000', '000', '111', '111', '010']
        y = np.array([10., 12., 12., 20., 22., 30.])
        x = np.array([[m == category for category in MASKS] for m in masks], dtype=float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        fit = categorical_fit(y.tolist(), masks)
        np.testing.assert_allclose(fit['residuals'], y - x @ beta, atol=1e-13)
        self.assertAlmostEqual(fit['beta']['000'], 34 / 3)
        self.assertAlmostEqual(fit['r2'], 1 - np.sum((y - x @ beta)**2) / np.sum((y-y.mean())**2))
        self.assertNotAlmostEqual(fit['r2'], 1 - fit['sse'] / float(y @ y))
        self.assertIsNone(fit['beta']['001'])

    def test_constant_and_invalid_inputs(self):
        fit = categorical_fit([2., 2.], ['000', '000'])
        self.assertIsNone(fit['r2'])
        self.assertEqual(fit['residuals'], [0., 0.])
        self.assertEqual(fit['r2_status'], 'constant_outcome')
        for values, masks in [([], []), ([float('nan')], ['000']), ([1.], ['bad']), ([True], ['000'])]:
            with self.assertRaises(ValueError): categorical_fit(values, masks)

    def test_tied_ranks_and_undefined(self):
        self.assertAlmostEqual(residual_spearman([1, 1, 3, 4], [4, 4, 2, 1])['rho'], -1)
        self.assertIsNone(residual_spearman([0, 0], [1, 2])['rho'])
        self.assertEqual(residual_spearman([1, 2], [0, 0])['status'], 'constant_reference_residual')

    def test_full_report_and_signs(self):
        result = build()
        self.assertEqual(len(result['summary']), 6)
        self.assertEqual(len(result['per_seed']), 25)
        self.assertEqual(result['diagnostics']['draw_count'], 33)
        self.assertEqual(result['diagnostics']['unique_pairs'], 32)
        self.assertAlmostEqual(result['reference_fit']['beta']['000'], -11.2)
        self.assertAlmostEqual(result['summary'][0]['beta']['000']['mean'], -.034)
        self.assertAlmostEqual(result['summary'][0]['beta']['000']['sample_sd'], np.std(np.arange(5)*.001,ddof=1))
        for r in result['per_seed']:
            # Floating subtraction can separate nominal ties at machine precision.
            self.assertGreater(r['residual_spearman']['rho'], .95)
        self.assertIsNone(result['summary'][-1]['residual_spearman'])
        json.dumps(result, allow_nan=False)

    def test_top_pairs_deduplicate_occurrences_only(self):
        result = build()
        top = result['top_pairs']['syllable']
        ids = [r['pair_id'] for r in top['rows']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(len(ids), 1)  # Same characters/Pinyin must not collapse.
        self.assertEqual(ids.count('p0000'), 1)
        self.assertTrue(all(r['resid'] > 0 for r in top['rows']))
        self.assertEqual([r['resid'] for r in top['rows']], sorted([r['resid'] for r in top['rows']], reverse=True))
        self.assertIn('text', top['rows'][0]['left'])
        tables = readable_tables(result)
        self.assertEqual(len(tables), 7)
        for name, table in tables.items():
            if name.startswith('top-'):
                self.assertEqual(table.splitlines()[0], '| char1 | char2 | pinyin1 | pinyin2 | mask | resid |')
                self.assertIn('Returned pairs:', table)

    def test_alignment_and_provenance_fail_closed(self):
        for mutate in (
            lambda d: d[0].pop(),
            lambda d: d[0].append(copy.deepcopy(d[0][0])),
            lambda d: d[0][1].update(run_id='another'),
            lambda d: d[0][1]['score'].update(bundle_manifest_sha256='d'*64),
            lambda d: d[0][1]['score']['modes']['dimsim-balanced']['rows'].reverse(),
            lambda d: d[0][0]['score']['modes']['dimsim-balanced']['rows'][0].update(overlap_mask='111'),
            lambda d: d[0][0]['score']['modes']['dimsim-balanced']['rows'][0].update(cosine_similarity=float('inf')),
            lambda d: d[1][0].update(sample_id='missing'),
            lambda d: d[1][0].update(character_position=10),
        ):
            data = fixture(); mutate(data)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError): build(data)

    def test_raw_mean_ranking_not_standardized(self):
        data = fixture()
        for record in data[0]:
            factor = 1 if record['seed'] < 4 else 3
            for row in record['score']['modes']['dimsim-balanced']['rows']:
                row['cosine_similarity'] *= factor / 3
        result = build(data)
        expected = np.mean([result['residuals']['syllable'][str(s)][0] for s in range(5)])
        pair = next(r for r in result['top_pairs']['syllable']['rows'] if r['pair_id']=='p0000')
        self.assertAlmostEqual(pair['resid'], expected)

    def test_fifty_cap_and_companion_identity(self):
        result = build(fixture(size=20))
        top = result['top_pairs']['syllable']
        self.assertEqual(top['returned_count'], 50)
        self.assertGreater(top['positive_unique_pair_count'], 50)
        self.assertEqual(len({r['pair_id'] for r in top['rows']}), 50)

    def test_empty_category_and_constant_residuals_in_report(self):
        data = fixture()
        for record in data[0]:
            rows = record['score']['modes']['dimsim-balanced']['rows']
            rows[:] = [r for r in rows if r['overlap_mask'] != '111']
            for row in rows:
                row['cosine_similarity'] = .5
        result = build(data)
        self.assertIsNone(result['summary'][0]['beta']['111']['mean'])
        self.assertIsNone(result['summary'][0]['r2']['mean'])
        self.assertIsNone(result['summary'][0]['residual_spearman']['mean'])
        self.assertEqual(result['top_pairs']['syllable']['returned_count'], 0)
        self.assertIn('Returned pairs: 0.', readable_tables(result)['top-50-syllable.md'])

    def test_repeated_pair_requires_identical_score(self):
        data = fixture()
        data[0][0]['score']['modes']['dimsim-balanced']['rows'][-1]['cosine_similarity'] += .1
        with self.assertRaisesRegex(ValueError, 'inconsistent scores'):
            build(data)

    def test_persistence_and_no_overwrite(self):
        from phonetics.evaluation.interpretation_outputs import write_interpretation_outputs
        from phonetics.research.artifacts import file_sha256
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / 'bundle'; bundle.mkdir()
            (bundle / 'manifest.json').write_text('{}', encoding='utf-8')
            data = fixture()
            sha = file_sha256(bundle / 'manifest.json')
            for record in data[0]: record['score']['bundle_manifest_sha256'] = sha
            result = build_interpretation(*data, bundle_manifest_sha256=sha)
            path = write_interpretation_outputs(root, 'synthetic', result,
                input_roots=[bundle], source_hashes={'interpretation.py': 'd'*64})
            manifest = json.loads((path / 'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(list((path / 'tables').glob('*.md'))), 7)
            stored = json.loads((path / 'data/top-50-syllable.json').read_text(encoding='utf-8'))
            self.assertIn('occurrence_id', stored['rows'][0]['left'])
            with self.assertRaises(FileExistsError):
                write_interpretation_outputs(root, 'synthetic', result,
                    input_roots=[bundle], source_hashes={'interpretation.py': 'd'*64})

    def test_failed_write_is_not_complete(self):
        from unittest.mock import patch
        from phonetics.evaluation.interpretation_outputs import write_interpretation_outputs
        from phonetics.research.artifacts import file_sha256
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / 'bundle'; bundle.mkdir()
            (bundle / 'manifest.json').write_text('{}', encoding='utf-8')
            data = fixture()
            sha = file_sha256(bundle / 'manifest.json')
            for record in data[0]: record['score']['bundle_manifest_sha256'] = sha
            result = build_interpretation(*data, bundle_manifest_sha256=sha)
            with patch('phonetics.evaluation.interpretation_outputs._write', side_effect=OSError('synthetic failure')):
                with self.assertRaises(OSError):
                    write_interpretation_outputs(root, 'failed', result,
                        input_roots=[bundle], source_hashes={'source': 'd'*64})
            manifest = json.loads((root/'outputs/contextual-interpretation/failed/manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
