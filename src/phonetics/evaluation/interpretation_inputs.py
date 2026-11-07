"""Hash-verified, bounded-record loading of saved contextual DIMSIM scores."""
from __future__ import annotations

import json
from pathlib import Path
import re

from .evaluation_outputs import _finite_json, _verify_inventory, _is_sha
from .interpretation import MODELS, SEEDS, build_interpretation
from .interpretation_outputs import write_interpretation_outputs
from ..research.artifacts import file_sha256, safe_run_id, ensure_output_outside_inputs


def _decode(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'non-finite JSON constant: {value}')
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


class _Reader:
    """Read JSON values using structural scanning, independent of whitespace.

    Only one result object is decoded at a time. The enclosing results array
    is never materialized. Natural scores within that object are then released.
    """
    tokens = re.compile(r'[{}\[\]"\\]')

    def __init__(self, stream, chunk_size=1024 * 1024):
        self.stream, self.chunk_size = stream, chunk_size
        self.buffer, self.pos = '', 0

    def peek(self):
        while True:
            if self.pos == len(self.buffer):
                self.buffer, self.pos = self.stream.read(self.chunk_size), 0
                if not self.buffer:
                    return ''
            if self.buffer[self.pos].isspace():
                self.pos += 1
            else:
                return self.buffer[self.pos]

    def expect(self, token):
        if self.peek() != token:
            raise ValueError(f'expected JSON delimiter {token}')
        self.pos += 1

    def value(self):
        first = self.peek()
        if not first:
            raise ValueError('truncated JSON value')
        if first not in '{["':
            parts = []
            while True:
                start = self.pos
                while self.pos < len(self.buffer) and self.buffer[self.pos] not in ',]} \t\r\n':
                    self.pos += 1
                parts.append(self.buffer[start:self.pos])
                if self.pos < len(self.buffer):
                    break
                self.buffer, self.pos = self.stream.read(self.chunk_size), 0
                if not self.buffer:
                    break
            return _decode(''.join(parts))
        depth, quoted, escaped, parts = 0, False, False, []
        start = self.pos
        while True:
            for match in self.tokens.finditer(self.buffer, self.pos):
                index, token = match.start(), match.group()
                if quoted:
                    # An escaped ordinary character is absent from tokens.
                    if escaped and index == self.pos:
                        escaped = False
                        self.pos = index + 1
                        continue
                    escaped = False
                    if token == '\\':
                        escaped = True
                    elif token == '"':
                        quoted = False
                elif token == '"':
                    quoted = True
                elif token in '{[':
                    depth += 1
                elif token in '}]':
                    depth -= 1
                self.pos = index + 1
                if not quoted and depth == 0:
                    parts.append(self.buffer[start:self.pos])
                    return _decode(''.join(parts))
            # Carry escaping only if the final character was the backslash.
            escaped = escaped and self.pos == len(self.buffer)
            parts.append(self.buffer[start:])
            self.buffer, self.pos, start = self.stream.read(self.chunk_size), 0, 0
            if not self.buffer:
                raise ValueError('truncated JSON container')


def iter_dimsim_records(path, *, bundle_sha256, chunk_size=1024 * 1024):
    """Strictly stream the top-level batch object, including its trailing syntax."""
    with Path(path).open(encoding='utf-8') as stream:
        reader = _Reader(stream, chunk_size)
        reader.expect('{')
        keys, header = set(), {}
        while reader.peek() != '}':
            key = reader.value()
            if not isinstance(key, str) or key in keys:
                raise ValueError('invalid or duplicate batch key')
            keys.add(key); reader.expect(':')
            if key == 'results':
                reader.expect('[')
                while reader.peek() != ']':
                    record = reader.value()
                    if not isinstance(record, dict):
                        raise ValueError('result must be an object')
                    yield record
                    del record
                    if reader.peek() == ']':
                        break
                    reader.expect(',')
                    if reader.peek() == ']':
                        raise ValueError('trailing results comma')
                reader.expect(']')
            else:
                header[key] = reader.value()
            if reader.peek() == '}':
                break
            reader.expect(',')
            if reader.peek() == '}':
                raise ValueError('trailing batch comma')
        reader.expect('}')
        if reader.peek():
            raise ValueError('trailing content after batch')
        if (keys != {'results', 'schema_version', 'bundle_manifest_sha256'}
                or type(header.get('schema_version')) is not int or header['schema_version'] != 1
                or header.get('bundle_manifest_sha256') != bundle_sha256):
            raise ValueError('batch header identity mismatch')


def _artifact_path(root, name):
    path = (root / name).resolve()
    if not isinstance(name, str) or path == root or root not in path.parents or not path.is_file():
        raise ValueError('unsafe or missing bundle artifact')
    return path


def load_saved_q_batch(root):
    """Read Q scores once, retaining only fields used by descriptive summaries."""
    from .evaluation_outputs import load_evaluation_summary
    summary = load_evaluation_summary(root)
    manifest = summary['manifest']
    if manifest['analysis'] != 'contextual-anchor-q':
        raise ValueError('Anchor-Q input required')
    sha = manifest['scientific_input_pins']['bundle_manifest_sha256']
    pins = {(p['model_id'], p['seed']): p for p in manifest['scientific_input_pins']['model_seed_provenance']}
    records = []
    for record in iter_dimsim_records(Path(root)/'data/anchor_q_batch.json', bundle_sha256=sha):
        pin = pins.get((record.get('model_id'), record.get('seed')))
        if pin is None or any(record.get(k) != pin.get(k) for k in ('run_id','run_manifest_sha256','checkpoint_sha256')):
            raise ValueError('Q record provenance differs from manifest')
        record['score']['draw_scores'] = [{k:r[k] for k in ('draw_id','triplet_id','edge_id','score')}
                                           for r in record['score']['draw_scores']]
        records.append(record)
    return {'schema_version': 1, 'bundle_manifest_sha256': sha, 'results': records}


def run_saved_interpretation(*, dimsim_root, dimsim_manifest_sha256, bundle_path,
                             bundle_manifest_sha256, project_root, run_id, source_hashes):
    """Verify exact inputs, stream saved scores, analyze, and write a new run.

    No calls to model, extraction, sampling or scoring APIs occur. All parsing
    and identity validation finishes before the output directory is created.
    """
    dimsim_root, bundle_path = (Path(p).resolve(strict=True) for p in (dimsim_root, bundle_path))
    target = Path(project_root).resolve() / 'outputs/contextual-interpretation' / safe_run_id(run_id)
    ensure_output_outside_inputs(target, [dimsim_root, bundle_path])
    if target.exists():
        raise FileExistsError(f'refusing to overwrite existing output: {target}')
    for parent in (target, *target.parents):
        if (parent / 'manifest.json').exists():
            raise ValueError('output would modify an immutable run')
    for sha, path in ((dimsim_manifest_sha256, dimsim_root / 'manifest.json'),
                      (bundle_manifest_sha256, bundle_path / 'manifest.json')):
        if not _is_sha(sha) or file_sha256(path) != sha:
            raise ValueError('explicit input manifest hash mismatch')
    expected = {'config.json', 'data/dimsim_batch.json', 'data/dimsim_report.json',
                'diagnostics/summary.json', 'figures/plot_data.json', 'tables/tables.json'}
    manifest = _verify_inventory(dimsim_root, 'contextual-dimsim', dimsim_root.name, expected)
    pins = manifest['scientific_input_pins']
    if (pins.get('bundle_manifest_sha256') != bundle_manifest_sha256
            or set(pins.get('model_ids', [])) != set(MODELS) or pins.get('seeds') != list(SEEDS)):
        raise ValueError('DIMSIM scientific pins mismatch')
    pin_rows = pins.get('model_seed_provenance', [])
    pin_map = {(r['model_id'], r['seed']): r for r in pin_rows}
    if len(pin_rows) != 25 or set(pin_map) != {(m, s) for m in MODELS for s in SEEDS}:
        raise ValueError('invalid model/seed provenance inventory')
    bundle_manifest = _finite_json(bundle_path / 'manifest.json')
    if bundle_manifest.get('schema_version') != 3:
        raise ValueError('schema-3 bundle required')
    artifacts = bundle_manifest['artifacts']
    for name, sha in artifacts.items():
        if not _is_sha(sha) or file_sha256(_artifact_path(bundle_path, name)) != sha:
            raise ValueError(f'bundle artifact hash mismatch: {name}')
    def jsonl(name):
        if name not in artifacts:
            raise ValueError(f'bundle inventory missing {name}')
        with _artifact_path(bundle_path, name).open(encoding='utf-8') as stream:
            return [_decode(line) for line in stream]
    occurrences, sentences = jsonl('occurrences.jsonl'), jsonl('sentences.jsonl')
    pairs = jsonl('pairs/dimsim-balanced.jsonl')
    identity_keys = ('draw_id', 'pair_id', 'left_id', 'right_id', 'overlap_mask', 'dimsim_distance')
    identities = [tuple(p[k] for k in identity_keys) for p in pairs]
    def records():
        for record in iter_dimsim_records(dimsim_root / 'data/dimsim_batch.json', bundle_sha256=bundle_manifest_sha256):
            pin = pin_map.get((record.get('model_id'), record.get('seed')))
            if pin is None or any(record.get(k) != pin.get(k) for k in
                                  ('run_id', 'run_manifest_sha256', 'checkpoint_sha256')):
                raise ValueError('record provenance differs from manifest')
            rows = record['score']['modes']['dimsim-balanced']['rows']
            if [tuple(r[k] for k in identity_keys) for r in rows] != identities:
                raise ValueError('balanced scores differ from fixed bundle sample')
            # Drop natural data and reports before the consumer retains this record.
            compact = {k: record[k] for k in ('model_id', 'seed', 'run_id', 'run_manifest_sha256', 'checkpoint_sha256')}
            score = record['score']
            compact['score'] = {k: score[k] for k in ('model_id', 'seed', 'bundle_manifest_sha256')}
            compact['score']['modes'] = {'dimsim-balanced': {'rows': rows}}
            del record, score
            yield compact
    result = build_interpretation(records(), occurrences, sentences,
                                  bundle_manifest_sha256=bundle_manifest_sha256)
    result['config']['saved_inputs'] = {'dimsim_root': str(dimsim_root),
        'dimsim_manifest_sha256': dimsim_manifest_sha256, 'bundle_path': str(bundle_path),
        'bundle_manifest_sha256': bundle_manifest_sha256, 'run_id': run_id,
        'loader': 'one JSON result at a time; balanced projection'}
    # Recheck consumed bytes after parsing, including a stable manifest identity.
    batch_artifact = next(a for a in manifest['artifacts'] if a['path'] == 'data/dimsim_batch.json')
    if file_sha256(dimsim_root / batch_artifact['path']) != batch_artifact['sha256']:
        raise ValueError('score bytes changed during analysis')
    for name in ('occurrences.jsonl', 'sentences.jsonl', 'pairs/dimsim-balanced.jsonl'):
        if file_sha256(_artifact_path(bundle_path, name)) != artifacts[name]:
            raise ValueError('bundle bytes changed during analysis')
    if (file_sha256(dimsim_root / 'manifest.json') != dimsim_manifest_sha256
            or file_sha256(bundle_path / 'manifest.json') != bundle_manifest_sha256):
        raise ValueError('input identity changed during analysis')
    return write_interpretation_outputs(project_root, run_id, result,
        input_roots=[bundle_path, dimsim_root], source_hashes=source_hashes)
