"""Synthetic saved-files to interpretation artifacts integration checks."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_interpretation import fixture
from phonetics.evaluation.interpretation_inputs import iter_dimsim_records, run_saved_interpretation
from phonetics.research.artifacts import file_sha256

PROJECT = Path(__file__).resolve().parents[1]


def write(path, value, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, **kwargs), encoding='utf-8')


def saved_fixture(root):
    records, occurrences, sentences = fixture()
    bundle = root / 'bundle'
    pairs = [{k: r[k] for k in ('draw_id', 'pair_id', 'left_id', 'right_id', 'overlap_mask', 'dimsim_distance')}
             for r in records[0]['score']['modes']['dimsim-balanced']['rows']]
    artifacts = {}
    for name, rows in [('occurrences.jsonl', occurrences), ('sentences.jsonl', sentences),
                       ('pairs/dimsim-balanced.jsonl', pairs)]:
        path = bundle / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        artifacts[name] = file_sha256(path)
    write(bundle / 'manifest.json', {'schema_version': 3, 'artifacts': artifacts})
    sha = file_sha256(bundle / 'manifest.json')
    for record in records:
        record['score']['bundle_manifest_sha256'] = sha
        record['score']['modes']['dimsim-natural'] = {'rows': [{'ignored': 'quoted \\" [ ] { } 中文'}]}
    path = root / 'contextual-dimsim' / 'exact-run'
    batch = {'schema_version': 1, 'bundle_manifest_sha256': sha, 'results': records}
    write(path / 'data/dimsim_batch.json', batch)
    for name in ('config.json', 'data/dimsim_report.json', 'figures/plot_data.json',
                 'tables/tables.json', 'diagnostics/summary.json'):
        write(path / name, {})
    files = [p for p in path.rglob('*') if p.is_file()]
    manifest = {'schema_version': 1, 'status': 'complete', 'analysis': 'contextual-dimsim',
        'run_id': 'exact-run', 'scope': 'numerical_outputs', 'figure_exports': 'pending',
        'python': sys.version, 'dependencies': {'numpy': 'test'}, 'inputs': [],
        'source_hashes': {'source': 'e'*64},
        'scientific_input_pins': {'bundle_manifest_sha256': sha,
            'model_ids': sorted({r['model_id'] for r in records}), 'seeds': list(range(5)),
            'model_seed_provenance': [{k: r[k] for k in ('model_id', 'seed', 'run_id', 'run_manifest_sha256', 'checkpoint_sha256')} for r in records]},
        'required_artifacts': sorted(p.relative_to(path).as_posix() for p in files),
        'artifacts': [{'path': p.relative_to(path).as_posix(), 'bytes': p.stat().st_size, 'sha256': file_sha256(p)} for p in files]}
    write(path / 'manifest.json', manifest)
    return dict(dimsim_root=path, dimsim_manifest_sha256=file_sha256(path/'manifest.json'),
                bundle_path=bundle, bundle_manifest_sha256=sha, project_root=root,
                run_id='interpretation', source_hashes={'source': 'e'*64})


class SavedInterpretationTests(unittest.TestCase):
    def test_stream_parser_whitespace_escapes_and_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'batch.json'
            rows = [{'text': 'a\\b"c\\"[{}]中文\n', 'n': [1, 2.3, None]}]
            for indent in (None, 2):
                write(path, {'results': rows, 'schema_version': 1, 'bundle_manifest_sha256': 'a'*64}, indent=indent)
                for chunk in (1, 2, 7, 1024):
                    self.assertEqual(list(iter_dimsim_records(path, bundle_sha256='a'*64, chunk_size=chunk)), rows)

    def test_stream_rejects_malformed_and_wrong_header(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'batch.json'
            good = json.dumps({'schema_version': 1, 'bundle_manifest_sha256': 'a'*64, 'results': [{}]})
            for text in (good+' junk', good[:-1], good.replace('[{}]', '[{},]'),
                         good.replace('"schema_version": 1', '"schema_version": 2'),
                         good.replace('"results":', '"schema_version": 1, "results":')):
                path.write_text(text)
                with self.assertRaises(ValueError): list(iter_dimsim_records(path, bundle_sha256='a'*64, chunk_size=3))

    def test_end_to_end_without_whole_batch_load_and_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            config = saved_fixture(Path(folder))
            before = {str(p): file_sha256(p) for p in Path(folder).rglob('*') if p.is_file()}
            original = Path.read_text
            def guard(path, *args, **kwargs):
                if path.name == 'dimsim_batch.json': raise AssertionError('whole batch read')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'read_text', guard):
                result = run_saved_interpretation(**config)
            self.assertEqual(json.loads((result/'manifest.json').read_text())['status'], 'complete')
            report = json.loads((result/'data/interpretation.json').read_text(encoding='utf-8'))
            self.assertEqual(len(report['summary']), 6)
            self.assertEqual(len(report['per_seed']), 25)
            self.assertEqual(len(list((result/'tables').glob('*.md'))), 7)
            self.assertTrue(all(file_sha256(Path(p)) == sha for p, sha in before.items()))
            with self.assertRaises(FileExistsError): run_saved_interpretation(**config)

    def test_tampering_and_inconsistent_fixed_sample_rejected_before_output(self):
        for mode in ('manifest', 'artifact', 'sample', 'provenance'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                config = saved_fixture(Path(folder))
                root = config['dimsim_root']
                if mode == 'manifest': config['dimsim_manifest_sha256'] = '0'*64
                elif mode == 'artifact': (root/'tables/tables.json').write_text('tampered')
                else:
                    batch_path = root/'data/dimsim_batch.json'
                    batch = json.loads(batch_path.read_text())
                    if mode == 'sample':
                        for record in batch['results']:
                            record['score']['modes']['dimsim-balanced']['rows'][0]['draw_id'] = 'different'
                    else: batch['results'][0]['checkpoint_sha256'] = 'f'*64
                    write(batch_path, batch)
                    manifest = json.loads((root/'manifest.json').read_text())
                    for artifact in manifest['artifacts']:
                        if artifact['path'] == 'data/dimsim_batch.json':
                            artifact.update(bytes=batch_path.stat().st_size, sha256=file_sha256(batch_path))
                    write(root/'manifest.json', manifest)
                    config['dimsim_manifest_sha256'] = file_sha256(root/'manifest.json')
                with self.assertRaises(ValueError): run_saved_interpretation(**config)
                self.assertFalse((Path(folder)/'outputs/contextual-interpretation/interpretation').exists())



if __name__ == '__main__': unittest.main()
