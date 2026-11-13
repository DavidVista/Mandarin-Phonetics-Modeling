"""Immutable character experiments with a shared test and per-seed training data."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import math

import numpy as np

from .acoustic import validate_character_data
from .artifacts import read_jsonl, safe_artifact, sha256_file, write_json
from .bundles import content_hash, seal_bundle, staging_directory, validate_bundle
from .data_contract import sample_id, is_han
from .objectives import canonical_model_id
from .pinyin import INITIALS, FINALS, TONES, decompose, validate_canonical_label
from .architecture import default_architecture, normalize_architecture

SCHEMA_VERSION = 4
DEFAULT_MODEL_IDS = ['syllable', 'components', 'acoustic', 'components-acoustic-lambda-1', 'components-acoustic-lambda-10']
_TRAIN = {'optimizer': 'Adam', 'learning_rate': 0.001, 'weight_decay': 0.0, 'batch_size': 32,
          'epochs': 20, 'scheduler': None, 'early_stopping': False, 'gradient_clipping': None, 'cosine_epsilon': 1e-8}


def default_experiment_config(acoustic_dim=1024):
    return {'schema_version': SCHEMA_VERSION, 'sample_size': 100_000, 'seed_count': 5,
            'test_seed': 0, 'test_fraction': 0.1, 'validation_fraction': 0.1,
            'architecture': default_architecture(acoustic_dim),
            'training': copy.deepcopy(_TRAIN), 'model_ids': list(DEFAULT_MODEL_IDS)}


def _validate_config(value, acoustic_dim):
    expected = default_experiment_config(acoustic_dim)
    if not isinstance(value, dict) or set(value) - {'test_size'} != set(expected):
        raise ValueError('invalid character experiment config keys')
    cfg = copy.deepcopy(value)
    for key in ('sample_size', 'seed_count'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if 'test_size' in cfg and (type(cfg['test_size']) is not int or cfg['test_size'] <= 0):
        raise ValueError('test_size must be a positive integer')
    for key in ('schema_version', 'test_seed', 'test_fraction', 'validation_fraction'):
        if content_hash(cfg[key]) != content_hash(expected[key]):
            raise ValueError(f'{key} conflicts with the experiment protocol')
    architecture = normalize_architecture(cfg['architecture'])
    if architecture['acoustic_dim'] != acoustic_dim:
        raise ValueError('architecture acoustic_dim disagrees with acoustic targets')
    training = cfg['training']
    if not isinstance(training, dict) or set(training) != set(_TRAIN):
        raise ValueError('invalid training parameters')
    for key, default in _TRAIN.items():
        value = training[key]
        if key in ('epochs', 'batch_size'):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        elif key == 'learning_rate':
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('learning_rate must be finite and positive')
        elif content_hash(value) != content_hash(default):
            raise ValueError(f'training.{key} conflicts with the experiment protocol')
    ids = cfg['model_ids']
    if not isinstance(ids, list) or not ids or [canonical_model_id(x) for x in ids] != ids or len(set(ids)) != len(ids):
        raise ValueError('model_ids must be unique canonical identifiers')
    if cfg['sample_size'] // 10 < 1:
        raise ValueError('sample_size must allow nonempty validation')
    return cfg


def stream_seed(seed, stream):
    return int(content_hash({'seed': seed, 'stream': stream})[:16], 16)


def make_splits(ids, config):
    ids = sorted(ids)
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate sample IDs')
    n, count = config['sample_size'], config['seed_count']
    test_size = config.get('test_size', n // 10)
    validation_size = n // 10
    if len(ids) < n + test_size:
        raise ValueError('eligible corpus is too small for sample_size plus common test')
    test = sorted(random.Random(stream_seed(config['test_seed'], 'test')).sample(ids, test_size))
    pool = sorted(set(ids) - set(test))
    seeds = {}
    for seed in range(count):
        selected = random.Random(stream_seed(seed, 'sample')).sample(pool, n)
        random.Random(stream_seed(seed, 'validation')).shuffle(selected)
        seeds[str(seed)] = {'train': sorted(selected[validation_size:]),
                            'validation': sorted(selected[:validation_size])}
    return {'test': test, 'seeds': seeds}


def character_vocabulary(rows, train_ids):
    tokens = ['[PAD]', '[UNK]'] + sorted({c for sid in train_ids for c in rows[sid]['text']})
    return {'tokens': tokens, 'stoi': {c: i for i, c in enumerate(tokens)}, 'pad_id': 0, 'unk_id': 1}


def unk_statistics(rows, ids, vocabulary):
    stoi = vocabulary['stoi']
    unknown = [sum(c not in stoi for c in rows[sid]['text']) for sid in ids]
    positions = sum(len(rows[sid]['text']) for sid in ids)
    return {'sentences': len(ids), 'positions': positions, 'unknown_positions': sum(unknown),
            'unknown_sentences': sum(n > 0 for n in unknown),
            'position_rate': sum(unknown) / positions, 'sentence_rate': sum(n > 0 for n in unknown) / len(ids)}


def prepare_experiment(experiment_id, *, prepared_root, config, output_root=None):
    from .source_preparation.archive import WINDOWS_DEVICES, WINDOWS_FORBIDDEN
    if (not isinstance(experiment_id, str) or not experiment_id or experiment_id in {'.', '..'}
            or any(c in WINDOWS_FORBIDDEN for c in experiment_id) or experiment_id.endswith(('.', ' '))
            or experiment_id.split('.')[0].casefold() in WINDOWS_DEVICES):
        raise ValueError('experiment_id must be a safe path component')
    source = Path(prepared_root)
    validate_character_data(source)
    matrix = np.load(source / 'targets.npy', allow_pickle=False)
    cfg = _validate_config(config, matrix.shape[1])
    rows = {r['sample_id']: r for r in read_jsonl(source / 'corpus.jsonl')}
    splits = make_splits(rows, cfg)
    identity = {'experiment_id': experiment_id, 'prepared_manifest_sha256': sha256_file(source / 'manifest.json'),
                'config': cfg, 'contract': 'character-experiment-4', 'training_seeds': list(range(cfg['seed_count']))}
    target = Path(output_root or source.parent.parent / 'experiments').absolute() / experiment_id
    if target.exists() or target.is_symlink():
        validate_experiment(target, identity=identity)
        return target
    with staging_directory(target) as temporary:
        for name in ('corpus.jsonl', 'labels.json', 'targets.npy'):
            shutil.copyfile(source / name, temporary / name)
        shutil.copyfile(source / 'manifest.json', temporary / 'prepared_manifest.json')
        write_json(temporary / 'config.json', cfg)
        write_json(temporary / 'splits.json', splits)
        seed_root = temporary / 'seeds'
        seed_root.mkdir()
        for key, split in splits['seeds'].items():
            root = seed_root / key
            root.mkdir()
            vocabulary = character_vocabulary(rows, split['train'])
            write_json(root / 'vocabulary.json', vocabulary)
            write_json(root / 'diagnostics.json', {name: unk_statistics(rows, ids, vocabulary)
                       for name, ids in (('validation', split['validation']), ('test', splits['test']))})
            write_json(root / 'input_hashes.json', {'train': content_hash(split['train']),
                       'validation': content_hash(split['validation']), 'test': content_hash(splits['test']),
                       'vocabulary': content_hash(vocabulary)})
        seal_bundle(temporary, 'character_experiment', identity)
        validate_experiment(temporary)
    return target


def validate_experiment(path, *, identity=None):
    root = Path(path)
    manifest = validate_bundle(root, 'character_experiment', identity=identity)
    matrix = np.load(root / 'targets.npy', allow_pickle=False)
    if matrix.ndim != 2 or not matrix.shape[1]:
        raise ValueError('invalid acoustic target matrix shape')
    cfg = _validate_config(json.loads((root / 'config.json').read_text(encoding='utf-8')), matrix.shape[1])
    if cfg != manifest['identity']['config']:
        raise ValueError('experiment config identity mismatch')
    if manifest['identity'].get('training_seeds') != list(range(cfg['seed_count'])):
        raise ValueError('experiment seed identities mismatch')
    prepared = json.loads((root / 'prepared_manifest.json').read_text(encoding='utf-8'))
    if sha256_file(root / 'prepared_manifest.json') != manifest['identity']['prepared_manifest_sha256']:
        raise ValueError('prepared provenance pin mismatch')
    for name in ('corpus.jsonl', 'labels.json', 'targets.npy'):
        if prepared['artifacts'].get(name) != sha256_file(root / name):
            raise ValueError('experiment copied source mismatch')
    records = read_jsonl(root / 'corpus.jsonl')
    rows = {r['sample_id']: r for r in records}
    if len(rows) != len(records):
        raise ValueError('duplicate corpus sample IDs')
    labels = json.loads((root / 'labels.json').read_text(encoding='utf-8'))
    from .data_contract import validate_acoustic
    validate_acoustic(dict(zip(labels, matrix)), expected_keys=labels)
    if labels != sorted(set(labels)) or matrix.shape[0] != len(labels) or matrix.dtype != np.float32:
        raise ValueError('invalid target inventory')
    for row in records:
        if (not row['text'] or any(not is_han(c) for c in row['text'])
                or len(row['text']) != len(row['labels']) or row['sentence_length'] != len(row['text'])
                or row['positions'] != list(range(len(row['text'])))
                or not set(row['labels']).issubset(labels)
                or row['sample_id'] != sample_id(row['text'], row['labels'])):
            raise ValueError('sample identity mismatch')
        for label in row['labels']:
            validate_canonical_label(label)
    splits = json.loads((root / 'splits.json').read_text(encoding='utf-8'))
    if splits != make_splits(rows, cfg):
        raise ValueError('seed split reconstruction mismatch')
    for key, split in splits['seeds'].items():
        folder = root / 'seeds' / key
        vocabulary = character_vocabulary(rows, split['train'])
        if json.loads((folder / 'vocabulary.json').read_text(encoding='utf-8')) != vocabulary:
            raise ValueError('vocabulary must be built exclusively from this seed train')
        diagnostics = {name: unk_statistics(rows, ids, vocabulary) for name, ids in
                       (('validation', split['validation']), ('test', splits['test']))}
        if json.loads((folder / 'diagnostics.json').read_text(encoding='utf-8')) != diagnostics:
            raise ValueError('UNK diagnostics mismatch')
        hashes = {'train': content_hash(split['train']), 'validation': content_hash(split['validation']),
                  'test': content_hash(splits['test']), 'vocabulary': content_hash(vocabulary)}
        if json.loads((folder / 'input_hashes.json').read_text(encoding='utf-8')) != hashes:
            raise ValueError('seed input hash mismatch')
    return manifest


def load_experiment(path):
    root = Path(path)
    manifest = validate_experiment(root)
    return {'root': root, 'manifest': manifest, 'config': manifest['identity']['config'],
            'samples': {r['sample_id']: r for r in read_jsonl(root / 'corpus.jsonl')},
            'splits': json.loads((root / 'splits.json').read_text(encoding='utf-8')),
            'acoustic_targets': np.load(root / 'targets.npy', allow_pickle=False),
            'labels': json.loads((root / 'labels.json').read_text(encoding='utf-8'))}


def load_common_test(path, *, expected_manifest_sha256):
    """Return common test samples from the exact bytes bound to the pinned manifest."""
    root = Path(path)
    manifest_bytes = safe_artifact(root, 'manifest.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError('experiment manifest SHA256 mismatch')
    manifest = validate_experiment(root)
    if manifest != json.loads(manifest_bytes):
        raise ValueError('experiment manifest changed during validation')

    splits_bytes = safe_artifact(root, 'splits.json').read_bytes()
    if hashlib.sha256(splits_bytes).hexdigest() != manifest['artifacts']['splits.json']:
        raise ValueError('experiment splits checksum mismatch while reading the common test split')
    test_ids = json.loads(splits_bytes)['test']
    selected_ids = set(test_ids)
    selected = {}
    corpus_digest = hashlib.sha256()
    with safe_artifact(root, 'corpus.jsonl').open('rb') as stream:
        for line in stream:
            corpus_digest.update(line)
            row = json.loads(line)
            if row['sample_id'] in selected_ids:
                selected[row['sample_id']] = row
    if corpus_digest.hexdigest() != manifest['artifacts']['corpus.jsonl']:
        raise ValueError('experiment corpus checksum mismatch while reading the common test split')
    if selected.keys() != selected_ids:
        raise ValueError('common test samples missing from the corpus')
    return test_ids, [selected[sample_id] for sample_id in test_ids]


def select_seed(experiment, seed):
    if type(seed) is not int or str(seed) not in experiment['splits']['seeds']:
        raise ValueError('seed is not present in experiment')
    vocabulary = json.loads((experiment['root'] / 'seeds' / str(seed) / 'vocabulary.json').read_text(encoding='utf-8'))
    labels = experiment['labels']
    label_ids = {label: i for i, label in enumerate(labels)}
    split = experiment['splits']['seeds'][str(seed)]
    samples = {}
    for sid in set(split['train'] + split['validation'] + experiment['splits']['test']):
        row = experiment['samples'][sid]
        samples[sid] = {**row, 'token_ids': [vocabulary['stoi'].get(c, 1) for c in row['text']],
                        'component_labels': [list(decompose(label)) for label in row['labels']],
                        'syllable_ids': [label_ids[label] for label in row['labels']],
                        'acoustic_ids': [label_ids[label] for label in row['labels']]}
    return {**experiment, 'seed': seed, 'samples': samples,
            'splits': {**split, 'test': experiment['splits']['test']},
            'vocab': {'characters': vocabulary, 'syllables': labels,
                      'labels': {'initials': INITIALS, 'finals': FINALS, 'tones': TONES}}}
