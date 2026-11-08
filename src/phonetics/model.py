"""Shared character and sentence encoders with condition-specific readout heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from .objectives import canonical_model_id, parse_model_id
from .architecture import normalize_architecture


class SharedEncoder(nn.Module):
    """Embed characters and encode complete variable-length sentences."""
    context_dim = 128

    def __init__(self, char_vocab_size: int, parameters: dict):
        super().__init__()
        self.context_dim = 2 * parameters['hidden_per_direction']
        self.embedding = nn.Embedding(char_vocab_size, parameters['embedding_dim'], padding_idx=0)
        self.sentence = nn.LSTM(parameters['embedding_dim'], parameters['hidden_per_direction'],
                                num_layers=parameters['layers'], bidirectional=True, batch_first=True)

    def encode(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = batch["token_ids"]
        lengths = batch["sentence_lengths"].to(dtype=torch.long)
        mask = batch["char_mask"].to(dtype=torch.bool)
        if token_ids.ndim != 2 or mask.shape != token_ids.shape:
            raise ValueError("invalid character batch shapes")
        if not torch.equal(mask, token_ids != 0):
            raise ValueError("character mask disagrees with padding")
        if not bool((lengths > 0).all()) or not torch.equal(lengths.cpu(), mask.sum(1).cpu()):
            raise ValueError("sentence lengths disagree with character mask")
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)[None, :]
        if not torch.equal(mask, positions < lengths.to(token_ids.device)[:, None]):
            raise ValueError("character sequences must be right padded")
        packed = pack_padded_sequence(self.embedding(token_ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.sentence(packed)
        hidden, _ = pad_packed_sequence(output, batch_first=True, total_length=token_ids.shape[1])
        return hidden, mask

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(batch)


class ClassifierHead(nn.Module):
    def __init__(self, output_dim: int, context_dim: int, parameters: dict):
        super().__init__()
        hidden = parameters['classification_hidden_dim']
        self.layers = nn.Sequential(nn.Linear(context_dim, hidden), nn.ReLU(),
                                    nn.Dropout(parameters['classification_dropout']), nn.Linear(hidden, output_dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden)


class TrainingModel(nn.Module):
    def __init__(self, encoder: SharedEncoder, model_id: str, vocab: dict, acoustic_dim: int, heads: dict):
        super().__init__()
        model_id = canonical_model_id(model_id)
        family, weight = parse_model_id(model_id)
        self.encoder = encoder
        self.model_id = model_id
        self.lambda_cosine = weight
        self.context_dim = encoder.context_dim
        labels = vocab["labels"]
        self.component_heads = nn.ModuleDict()
        if family in {"components", "components-acoustic"}:
            for name in ("initials", "finals", "tones"):
                self.component_heads[name] = ClassifierHead(len(labels[name]), self.context_dim, heads)
        self.syllable_head = ClassifierHead(len(vocab["syllables"]), self.context_dim, heads) if family == "syllable" else None
        self.acoustic_head = nn.Linear(self.context_dim, int(acoustic_dim)) if family in {"acoustic", "components-acoustic"} else None

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
    architecture = normalize_architecture(config['architecture'])
    devices = []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        encoder = SharedEncoder(len(vocab["characters"]["tokens"]), architecture['encoder'])
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed) + 1_000_003)
        model = TrainingModel(encoder, model_id, vocab, architecture['acoustic_dim'], architecture['heads'])
    return model


def build_encoder(config: dict, vocab: dict, model_id: str | None = None, seed: int = 0) -> TrainingModel:
    """Build a full model; the returned `.encode(batch)` is the shared export API."""
    normalize_architecture(config.get("architecture", {}))
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
