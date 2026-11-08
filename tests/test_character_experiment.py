"""Shared-test, train-only vocabulary and seed provenance regressions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from phonetics.artifacts import sha256_file, write_json, write_jsonl
from phonetics.bundles import seal_bundle
from phonetics.data_contract import sample_id
from phonetics.experiment import (default_experiment_config, prepare_experiment, load_experiment,
                                  select_seed, validate_experiment, make_splits)


class CharacterExperimentTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        rows = []
        for i in range(35):
            text = chr(0x4e00 + i)
            rows.append({"text": text, "labels": ["ma1"], "sample_id": sample_id(text, ["ma1"]),
                         "source_ids": [str(i)], "sentence_length": 1, "positions": [0]})
        rows.sort(key=lambda r: r["sample_id"])
        write_jsonl(self.prepared / "corpus.jsonl", rows)
        write_jsonl(self.prepared / "inventory.jsonl", [{"label": "ma1", "sha256": "a" * 64}])
        write_jsonl(self.prepared / "rejections.jsonl", [])
        write_json(self.prepared / "statistics.json", {"input": 35, "selected": 35, "duplicates": 0, "rejected": 0})
        write_json(self.prepared / "labels.json", ["ma1"])
        np.save(self.prepared / "targets.npy", np.array([[1., 2.]], dtype=np.float32), allow_pickle=False)
        write_json(self.prepared / "inference.json", {"fixture": True})
        from phonetics.bundles import content_hash
        for name, kind, artifacts in (
            ('corpus_manifest.json', 'character_corpus', ['corpus.jsonl', 'inventory.jsonl', 'statistics.json', 'rejections.jsonl']),
            ('acoustic_manifest.json', 'acoustic_targets', ['labels.json', 'targets.npy', 'inventory.jsonl', 'inference.json'])):
            identity = {'inventory_manifest_sha256': 'a' * 64}
            write_json(self.prepared / name, {'schema_version': 1, 'bundle_type': kind, 'identity': identity,
                       'identity_sha256': content_hash(identity), 'artifacts': {n: sha256_file(self.prepared / n) for n in artifacts}})
        seal_bundle(self.prepared, 'character_prepared', {
            'corpus_manifest_sha256': sha256_file(self.prepared / 'corpus_manifest.json'),
            'acoustic_manifest_sha256': sha256_file(self.prepared / 'acoustic_manifest.json')})
        self.config = default_experiment_config(2)
        self.config.update(sample_size=20, seed_count=3)

    def build(self, name='experiment'):
        return prepare_experiment(name, prepared_root=self.prepared, output_root=self.root, config=self.config)

    def test_split_sizes_disjointness_and_independent_seeds(self):
        loaded = load_experiment(self.build())
        test = set(loaded['splits']['test'])
        self.assertEqual(len(test), 2)
        train_lists = []
        for seed in range(3):
            selected = select_seed(loaded, seed)
            train, val = set(selected['splits']['train']), set(selected['splits']['validation'])
            self.assertEqual((len(train), len(val)), (18, 2))
            self.assertFalse(train & val or test & train or test & val)
            train_lists.append(train)
            expected = {c for sid in train for c in selected['samples'][sid]['text']}
            self.assertEqual(set(selected['vocab']['characters']['tokens'][2:]), expected)
            self.assertTrue(all(selected['samples'][sid]['token_ids'] == [1] for sid in test | val))
        self.assertNotEqual(train_lists[0], train_lists[1])

    def test_rebuild_reproduces_vocabulary_and_common_test(self):
        first, second = self.build('a'), self.build('b')
        for name in ('splits.json', 'seeds/0/vocabulary.json', 'seeds/1/input_hashes.json'):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_vocabulary_leak_rejected_after_outer_reseal(self):
        root = self.build()
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        path = root / 'seeds/0/vocabulary.json'
        vocabulary = json.loads(path.read_text(encoding='utf-8'))
        vocabulary['tokens'].append('外')
        write_json(path, vocabulary)
        seal_bundle(root, 'character_experiment', manifest['identity'])
        with self.assertRaisesRegex(ValueError, 'exclusively'):
            validate_experiment(root)

    def test_small_pool_and_none_sampling_rejected(self):
        with self.assertRaisesRegex(ValueError, 'too small'):
            make_splits(['a', 'b'], self.config)
        self.config['sample_size'] = None
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            self.build()

    def test_omitted_configuration_fails_before_sampling_or_publication(self):
        with patch('phonetics.experiment.make_splits', side_effect=AssertionError('sampling must not run')):
            with self.assertRaisesRegex(TypeError, 'config'):
                prepare_experiment('omitted', prepared_root=self.prepared, output_root=self.root)
        self.assertFalse((self.root / 'omitted').exists())


if __name__ == '__main__':
    unittest.main()
