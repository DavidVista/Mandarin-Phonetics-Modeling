"""Shared IDS and sentence encoders with condition-specific readout heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from .objectives import canonical_model_id, parse_model_id


class IDSCharacterEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.bigru = nn.GRU(64, 32, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        valid = lengths.reshape(-1) > 0
        if not bool(valid.any()):
            return self.embedding.weight.new_zeros((lengths.numel(), 64))
        selected_ids = token_ids.reshape(-1, token_ids.shape[-1])[valid]
        selected_lengths = lengths.reshape(-1)[valid].to(device="cpu", dtype=torch.long)
        embedded = self.embedding(selected_ids)
        packed = pack_padded_sequence(embedded, selected_lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.bigru(packed)
        character_vectors = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        result = self.embedding.weight.new_zeros((lengths.numel(), 64))
        return result.index_copy(0, valid.nonzero(as_tuple=False).flatten(), character_vectors).reshape(*lengths.shape, 64)


class SharedEncoder(nn.Module):
    """Encode variable-length IDS for each character, then full sentence context."""
    context_dim = 128

    def __init__(self, ids_vocab_size: int):
        super().__init__()
        self.ids = IDSCharacterEncoder(ids_vocab_size)
        self.sentence = nn.LSTM(64, 64, num_layers=1, bidirectional=True, batch_first=True)

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = batch["ids_sequences"]
        component_lengths = batch["component_lengths"]
        sentence_lengths = batch["sentence_lengths"].to(dtype=torch.long)
        mask = batch["char_mask"].to(dtype=torch.bool)
        if token_ids.ndim != 3 or component_lengths.shape != token_ids.shape[:2] or mask.shape != token_ids.shape[:2]:
            raise ValueError("invalid padded IDS batch shapes")
        if not torch.equal(mask, component_lengths > 0):
            raise ValueError("character mask and IDS lengths disagree")
        if not bool((sentence_lengths > 0).all()) or not torch.equal(sentence_lengths.cpu(), mask.sum(1).cpu()):
            raise ValueError("sentence lengths disagree with character mask")
        char_vectors = self.ids(token_ids, component_lengths)
        packed = pack_padded_sequence(char_vectors, sentence_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.sentence(packed)
        contextual, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=token_ids.shape[1])
        return contextual, mask

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(batch)


class ClassifierHead(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, output_dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden)


class TrainingModel(nn.Module):
    def __init__(self, encoder: SharedEncoder, model_id: str, vocab: dict, acoustic_dim: int):
        super().__init__()
        model_id = canonical_model_id(model_id)
        family, weight = parse_model_id(model_id)
        self.encoder = encoder
        self.model_id = model_id
        self.lambda_cosine = weight
        self.context_dim = 128
        labels = vocab["labels"]
        self.component_heads = nn.ModuleDict()
        if family in {"components", "components-acoustic"}:
            for name in ("initials", "finals", "tones"):
                self.component_heads[name] = ClassifierHead(len(labels[name]))
        self.syllable_head = ClassifierHead(len(vocab["syllables"])) if family == "syllable" else None
        self.acoustic_head = nn.Linear(128, int(acoustic_dim)) if family in {"acoustic", "components-acoustic"} else None

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder.encode(batch)

    def forward(self, batch: dict) -> dict:
        hidden, mask = self.encoder.encode(batch)
        result = {"hidden": hidden, "mask": mask}
        if self.component_heads:
            result["component_logits"] = {name: head(hidden) for name, head in self.component_heads.items()}
        if self.syllable_head is not None:
            result["syllable_logits"] = self.syllable_head(hidden)
        if self.acoustic_head is not None:
            result["acoustic_prediction"] = self.acoustic_head(hidden)
        return result


def _construct_with_streams(config: dict, vocab: dict, model_id: str, seed: int) -> TrainingModel:
    """Initialize shared weights and heads from independent seed streams."""
    model_id = canonical_model_id(model_id)
    devices = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        encoder = SharedEncoder(len(vocab["ids"]["tokens"]))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed) + 1_000_003)
        model = TrainingModel(encoder, model_id, vocab, int(config["architecture"]["acoustic_dim"]))
    return model


def build_encoder(config: dict, vocab: dict, model_id: str | None = None, seed: int = 0) -> TrainingModel:
    """Build a full model; the returned `.encode(batch)` is the shared export API."""
    architecture = config.get("architecture", {})
    if architecture.get("embedding_dim") != 64 or architecture.get("context_dim") != 128 or architecture.get("acoustic_dim", 0) <= 0:
        raise ValueError("experiment architecture violates encoder dimension contract")
    selected = model_id or config.get("model_id")
    if selected is None:
        raise ValueError("model_id is required to construct objective heads")
    return _construct_with_streams(config, vocab, selected, seed)


def shared_encoder_sha256(model: TrainingModel) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(model.encoder.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii")); digest.update(value.numpy().tobytes())
    return digest.hexdigest()
