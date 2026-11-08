"""Torch dataset and collation for seed-specific character experiment bundles."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from .reproducibility import make_generator


class ExperimentDataset(Dataset):
    def __init__(self, sample_ids, samples):
        self.sample_ids = list(sample_ids)
        self.samples = samples

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        return self.samples[self.sample_ids[index]]


def collate_experiment(samples):
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    max_sentence = max(sample["sentence_length"] for sample in samples)
    token_ids = torch.zeros((batch_size, max_sentence), dtype=torch.long)
    component_labels = torch.full((batch_size, max_sentence, 3), -100, dtype=torch.long)
    syllable_ids = torch.full((batch_size, max_sentence), -100, dtype=torch.long)
    acoustic_ids = torch.full((batch_size, max_sentence), -1, dtype=torch.long)
    sentence_lengths = torch.empty(batch_size, dtype=torch.long)
    for row_index, sample in enumerate(samples):
        n = sample["sentence_length"]
        sentence_lengths[row_index] = n
        sequence = sample["token_ids"]
        if len(sequence) != n or any(type(token) is not int or token <= 0 for token in sequence):
            raise ValueError("invalid character token sequence")
        token_ids[row_index, :n] = torch.as_tensor(sequence, dtype=torch.long)
        component_labels[row_index, :n] = torch.as_tensor(sample["component_labels"], dtype=torch.long)
        syllable_ids[row_index, :n] = torch.as_tensor(sample["syllable_ids"], dtype=torch.long)
        acoustic_ids[row_index, :n] = torch.as_tensor(sample["acoustic_ids"], dtype=torch.long)
    char_mask = token_ids > 0
    return {
        "token_ids": token_ids,
        "sentence_lengths": sentence_lengths, "char_mask": char_mask,
        "component_labels": component_labels, "syllable_ids": syllable_ids,
        "acoustic_ids": acoustic_ids,
        "sample_ids": [sample["sample_id"] for sample in samples],
        "texts": [sample["text"] for sample in samples],
        "positions": [sample["positions"] for sample in samples],
    }


def create_loaders(experiment: dict, seed: int, batch_size: int | None = None, num_workers: int = 0):
    manifest = experiment.get("manifest", {})
    if manifest.get("bundle_type") != "character_experiment" or experiment.get("seed") != seed:
        raise ValueError("training loaders require a validated seed-specific character experiment")
    if type(seed) is not int or seed < 0:
        raise ValueError("loader seed must be a nonnegative integer")
    batch = batch_size or experiment["config"].get("training", {}).get("batch_size", 32)
    if type(batch) is not int or batch <= 0 or type(num_workers) is not int or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    result = {}
    for stream, split in enumerate(("train", "validation", "test"), start=1):
        dataset = ExperimentDataset(experiment["splits"][split], experiment["samples"])
        result[split] = DataLoader(dataset, batch_size=batch, shuffle=split == "train",
                                   generator=make_generator(seed, stream), num_workers=num_workers,
                                   collate_fn=collate_experiment)
    return result
