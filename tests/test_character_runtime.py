"""CPU fixtures for character batching and shared model initialization."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'CPU Torch is required')
class CharacterRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_objectives_share_encoder_and_batches_ignore_padding(self):
        from phonetics.experiment import default_experiment_config, DEFAULT_MODEL_IDS
        from phonetics.data import collate_experiment
        from phonetics.model import build_encoder, shared_encoder_sha256
        from phonetics.pinyin import INITIALS, FINALS, TONES, decompose
        from phonetics.metrics import batch_objective
        labels = list(decompose('ma1'))
        def sample(text, tokens, sid):
            n = len(text)
            return {'text': text, 'token_ids': tokens, 'sentence_length': n, 'sample_id': sid,
                    'positions': list(range(n)), 'component_labels': [labels] * n,
                    'syllable_ids': [0] * n, 'acoustic_ids': [0] * n}
        batch = collate_experiment([sample('妈', [2], 'a'), sample('妈妈', [2, 2], 'b')])
        self.assertEqual(batch['token_ids'].tolist(), [[2, 0], [2, 2]])
        config = default_experiment_config(2)
        vocabulary = {'characters': {'tokens': ['[PAD]', '[UNK]', '妈']}, 'syllables': ['ma1'],
                      'labels': {'initials': INITIALS, 'finals': FINALS, 'tones': TONES}}
        hashes = []
        for objective in DEFAULT_MODEL_IDS:
            model = build_encoder(config, vocabulary, model_id=objective, seed=1)
            hashes.append(shared_encoder_sha256(model))
            model.eval()
            outputs = model(batch)
            alone = model(collate_experiment([sample('妈', [2], 'a')]))
            self.assertTrue(torch.allclose(outputs['hidden'][0, 0], alone['hidden'][0, 0], atol=1e-6))
            self.assertEqual(outputs['mask'].sum().item(), 3)
            stats = batch_objective(outputs, batch, torch.tensor([[1., 2.]]), objective)
            self.assertTrue(torch.isfinite(stats['objective']))
            stats['objective'].backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertEqual(len(set(hashes)), 1)
        other = build_encoder(config, vocabulary, model_id='syllable', seed=2)
        self.assertNotEqual(shared_encoder_sha256(other), hashes[0])

    def test_sequential_runtime_checkpoints_bind_each_seed_inputs(self):
        from test_character_experiment import CharacterExperimentTests
        from phonetics.training import train_experiment
        from phonetics.experiment import load_experiment, select_seed
        from phonetics.artifacts import sha256_file
        fixture = CharacterExperimentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config['seed_count'] = 2
        root = fixture.build()
        observed = []
        def epoch(model, loader, acoustic, model_id, device, **kwargs):
            observed.append(tuple(loader.dataset.sample_ids))
            return {'objective': 1.0, 'gradient_norms': [], 'gradient_nonfinite_count': 0}, kwargs.get('step_start', 0) + 1
        with patch('phonetics.training._epoch', side_effect=epoch):
            result = train_experiment(root, model_id='syllable', device='cpu',
                                      runs_root=fixture.root / 'runs', run_id='fixture')
        self.assertEqual(result['manifest']['seeds'], [0, 1])
        self.assertEqual(result['manifest']['status'], 'complete')
        loaded = load_experiment(root)
        for seed in (0, 1):
            seed_dir = result['run_dir'] / f'seed-{seed}'
            checkpoint = torch.load(seed_dir / 'best.pt', weights_only=True)
            self.assertEqual(checkpoint['seed'], seed)
            self.assertEqual(checkpoint['vocab_sha256'], sha256_file(root / 'seeds' / str(seed) / 'vocabulary.json'))
            self.assertEqual(checkpoint['seed_input_hashes']['test'], result['manifest']['seed_results'][seed]['seed_input_hashes']['test'])
            self.assertIn(tuple(select_seed(loaded, seed)['splits']['train']), observed)
        self.assertNotEqual(result['manifest']['seed_results'][0]['seed_input_hashes']['train'],
                            result['manifest']['seed_results'][1]['seed_input_hashes']['train'])


if __name__ == '__main__':
    unittest.main()
