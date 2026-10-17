"""Canonical encoder shared by all training conditions."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class Encoder(nn.Module):
    """Byte-for-byte state-dict-compatible extraction of the pure encoder."""

    def __init__(self, vocab_size_tokens: int, embedding_dim: int, hidden_dim: int,
                 num_initials: int, num_finals: int, num_tones: int,
                 num_layers: int = 4, dropout: float = 0.3, mlp_hidden_dim: int = 256):
        super().__init__()
        self.embedding_bag = nn.EmbeddingBag(vocab_size_tokens, embedding_dim, mode="mean")
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.initial_head = ClassifierHead(hidden_dim * 2, mlp_hidden_dim, num_initials, dropout)
        self.final_head = ClassifierHead(hidden_dim * 2, mlp_hidden_dim, num_finals, dropout)
        self.tone_head = ClassifierHead(hidden_dim * 2, mlp_hidden_dim, num_tones, dropout)

    def forward(self, token_ids, token_offsets, char_lengths, return_hidden=False):
        char_emb = self.embedding_bag(token_ids, token_offsets)
        padded = nn.utils.rnn.pad_sequence(torch.split(char_emb, char_lengths.tolist()), batch_first=True)
        packed = pack_padded_sequence(padded, char_lengths.cpu(), batch_first=True, enforce_sorted=False)
        char_out, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True)
        char_out = self.dropout(char_out)
        result = (self.initial_head(char_out), self.final_head(char_out), self.tone_head(char_out))
        return (*result, char_out) if return_hidden else result


def build_encoder(config: dict, vocab: dict) -> Encoder:
    arch = config["architecture"]
    acoustic_dim = int(config["acoustic_dim"])
    if 2 * int(arch["hidden_dim"]) != acoustic_dim:
        raise ValueError("2 * hidden_dim must equal acoustic_dim for normalized-MSE alignment")
    labels = vocab["labels"]
    return Encoder(len(vocab["wubi_tokens"]), int(arch["embedding_dim"]), int(arch["hidden_dim"]),
                   len(labels["initials"]), len(labels["finals"]), len(labels["tones"]),
                   int(arch["num_layers"]), float(arch["dropout"]), int(arch["mlp_hidden_dim"]))
