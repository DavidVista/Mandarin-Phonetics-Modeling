"""Immutable experiment preparation, validation, and loading."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import random
import shutil
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model import build_encoder
from .reproducibility import make_generator, seed_everything
from .pinyin import INITIALS, FINALS, TONES, PINYIN_CONTRACT_VERSION, PARSER_PROVENANCE, canonical_labels, decompose, validate_canonical_label
from .data_contract import pair_reason, sample_id
from .artifacts import sha256_file
from .preparation import validate_prepared

PREPARATION_VERSION = "3"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _encode(text: str, mapping: dict, stoi: dict[str, int]) -> tuple[list[int], list[list[int]]]:
    ids, spans = [], []
    for char in text:
        code = str(mapping.get(char, char))
        start = len(ids)
        ids.extend(stoi.get(symbol, stoi["[UNK]"]) for symbol in code)
        ids.append(stoi["#"])
        spans.append([start, len(ids)])
    return ids, spans


def _default_config(acoustic_dim: int) -> dict:
    return {"schema_version": 1, "seeds": [42], "split_seed": 42, "split_ratios": [0.8, 0.1, 0.1],
            "architecture": {"embedding_dim": 100, "hidden_dim": 512, "num_layers": 4, "mlp_hidden_dim": 256, "dropout": 0.3},
            "acoustic_dim": acoustic_dim, "training": {"epochs": 20, "early_stopping": False, "batch_size": 32,
            "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "scheduler": "constant",
            "checkpoint_selection": "minimum condition-specific validation objective", "combined_lambda_mse": 1.0,
            "combined_lambda_mse_options": [1.0, 100.0]}}


def prepare_experiment(experiment_id: str, *, datasets_root: str | Path, output_root: str | Path | None = None,
                       config_overrides: dict | None = None) -> Path:
    """Create a complete immutable experiment directory; never overwrites an existing ID."""
    if not isinstance(experiment_id, str) or not experiment_id or Path(experiment_id).name != experiment_id or experiment_id in {".", ".."}:
        raise ValueError("experiment_id must be one non-empty safe path component")
    root = Path(datasets_root).resolve()
    prepared_manifest = validate_prepared(root) if (root / "manifest.json").is_file() else None
    sources = {"corpus": root / "corpus/train.csv", "wubi": root / "wubi/chinese_to_wubi.pkl", "acoustic": root / "embeddings/pinyin_embeddings.pkl"}
    if any(not p.is_file() for p in sources.values()):
        raise FileNotFoundError({k: str(v) for k, v in sources.items() if not v.is_file()})
    output = Path(output_root).resolve() if output_root else (root / "experiments").resolve()
    target = (output / experiment_id).resolve()
    if target.parent != output:
        raise ValueError("experiment path escapes output_root")
    if target.exists():
        raise FileExistsError(f"experiment already exists: {target}")
    with sources["wubi"].open("rb") as f: wubi = pickle.load(f)
    with sources["acoustic"].open("rb") as f: acoustic = pickle.load(f)
    if not isinstance(wubi, dict) or not wubi:
        raise ValueError("Wubi mapping must be a non-empty dict")
    valid_wubi = {str(k): v for k, v in wubi.items() if isinstance(k, str) and len(k) == 1 and isinstance(v, str) and bool(v)}
    keys = sorted(str(k) for k in acoustic)
    matrix = np.stack([np.asarray(acoustic[k], dtype=np.float32) for k in keys])
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("acoustic embeddings must be a finite 2-D matrix with nonzero rows")
    config = _default_config(matrix.shape[1])
    overrides = config_overrides or {}
    if set(overrides) - {"architecture"}:
        raise ValueError("mock overrides may only change architecture")
    config["architecture"].update(overrides.get("architecture", {}))
    config["acoustic_dim"] = int(matrix.shape[1])
    if 2 * int(config["architecture"]["hidden_dim"]) != int(config["acoustic_dim"]):
        raise ValueError("2 * hidden_dim must equal acoustic_dim")
    letters = sorted({c for code in valid_wubi.values() for c in code})
    wubi_tokens = ["[PAD]", "[UNK]", "#", *letters]
    stoi = {v: i for i, v in enumerate(wubi_tokens)}
    acoustic_ids = {v: i for i, v in enumerate(keys)}
    accepted, rejected, seen = [], Counter(), {}
    with sources["corpus"].open(encoding="utf-8-sig", newline="") as f:
        for row_number, row in enumerate(csv.DictReader(f), start=2):
            text_value = row.get("text")
            pinyin_value = row.get("pinyin")
            text = text_value if isinstance(text_value, str) else ""
            raw = pinyin_value.split() if isinstance(pinyin_value, str) else []
            try:
                canonical = canonical_labels(raw, tone_convention="explicit" if prepared_manifest is not None else "neutral_unmarked") if raw else []
            except (ValueError, TypeError, IndexError):
                canonical = []
                normalization_failed = bool(raw)
            else:
                normalization_failed = False
            reason = None
            if prepared_manifest is not None:
                reason = pair_reason(text, canonical, valid_wubi, acoustic)
                if reason: raise ValueError(f"prepared row {row_number} rejected by common contract: {reason}")
                if text in seen: raise ValueError(f"duplicate text in prepared dataset at rows {seen[text][0]} and {row_number}")
                components = [decompose(x) for x in canonical]
            elif not text or not raw: reason = "invalid_row"
            elif normalization_failed: reason = "invalid_decomposition"
            elif len(text) != len(raw): reason = "length_mismatch"
            elif text in seen and seen[text][1] != canonical: raise ValueError(f"conflicting labels for duplicate text at rows {seen[text][0]} and {row_number}")
            elif text in seen: reason = "duplicate_text"
            elif any(ch not in valid_wubi for ch in text): reason = "invalid_wubi_mapping"
            else:
                try: components = [decompose(x) for x in canonical]
                except ValueError: reason = "invalid_decomposition"
                if reason is None and any(x not in acoustic_ids for x in canonical): reason = "missing_acoustic_coverage"
            if reason:
                rejected[reason] += 1
                continue
            seen[text] = (row_number, canonical)
            token_ids, spans = _encode(text, valid_wubi, stoi)
            digest = sample_id(text, canonical)
            accepted.append({"sample_id": digest, "source_row": row_number, "text": text, "token_ids": token_ids,
                             "character_spans": spans, "component_labels": components,
                             "acoustic_ids": [acoustic_ids[x] for x in canonical]})
    if not accepted: raise ValueError("no valid samples")
    ids = sorted(s["sample_id"] for s in accepted)
    random.Random(int(config["split_seed"])).shuffle(ids)
    n = len(ids); train_end = int(n * config["split_ratios"][0]); val_end = train_end + int(n * config["split_ratios"][1])
    splits = {"train": ids[:train_end], "validation": ids[train_end:val_end], "test": ids[val_end:]}
    if any(not members for members in splits.values()):
        raise ValueError("train, validation, and test splits must all be non-empty")
    split_of = {sid: split for split, members in splits.items() for sid in members}
    chars = sorted({ch for s in accepted if split_of[s["sample_id"]] == "train" for ch in s["text"]})
    char_vocab = ["[PAD]", "[UNK]", *chars]; char_stoi = {c:i for i,c in enumerate(char_vocab)}
    for sample in accepted: sample["char_ids"] = [char_stoi.get(c, 1) for c in sample["text"]]
    target.mkdir(parents=True)
    try:
        (target / "initialization").mkdir()
        (target / "samples.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n" for s in sorted(accepted, key=lambda x:x["sample_id"])), encoding="utf-8")
        vocab = {"wubi_tokens": wubi_tokens, "char_tokens": char_vocab, "acoustic_tokens": keys,
                 "labels": {"initials": INITIALS, "finals": FINALS, "tones": TONES}}
        shutil.copyfile(sources["wubi"], target / "wubi_mapping.pkl")
        _json(target / "vocab.json", vocab); _json(target / "splits.json", splits); _json(target / "config.json", config)
        torch.save(torch.from_numpy(matrix), target / "acoustic_targets.pt")
        source_hashes = {k: _sha(v) for k,v in sources.items()}
        for seed in config["seeds"]:
            seed_everything(int(seed)); model = build_encoder(config, vocab)
            torch.save({"state_dict": model.state_dict(), "seed": int(seed), "config_sha256": _sha(target / "config.json"),
                        "vocab_sha256": _sha(target / "vocab.json"), "source_sha256": source_hashes}, target / f"initialization/seed-{seed}.pt")
        artifacts = {str(p.relative_to(target)).replace("\\", "/"): _sha(p) for p in target.rglob("*") if p.is_file()}
        code_files = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("reproducibility.py"),
                      Path(__file__).with_name("pinyin.py"), Path(__file__).with_name("data_contract.py")]
        manifest = {"schema_version": 1, "experiment_id": experiment_id, "preparation_version": PREPARATION_VERSION,
                    "pinyin_contract": PINYIN_CONTRACT_VERSION, "parser_provenance": PARSER_PROVENANCE,
                    "code": {p.name: _sha(p) for p in code_files},
                    "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
                    "sources": {k: {"path": str(v), "sha256": source_hashes[k]} for k,v in sources.items()},
                    **({"prepared_dataset": {"id": prepared_manifest["dataset_id"], "manifest_sha256": sha256_file(root / "manifest.json"),
                                             "contract": prepared_manifest["data_contract"]}} if prepared_manifest else {}),
                    "counts": {"input_rows": len(accepted) + sum(rejected.values()), "accepted": len(accepted), "rejected": dict(sorted(rejected.items())),
                               "splits": {k: len(v) for k,v in splits.items()}, "seeds": config["seeds"]}, "artifacts": artifacts}
        _json(target / "manifest.json", manifest)
    except Exception:
        shutil.rmtree(target)
        raise
    return target


class ExperimentDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, index): return self.samples[index]


def _collate(samples):
    token_ids, offsets, cursor = [], [], 0
    max_chars = max(len(s["character_spans"]) for s in samples)
    component = torch.full((len(samples), max_chars, 3), -100, dtype=torch.long)
    acoustic = torch.full((len(samples), max_chars), -1, dtype=torch.long)
    for row, sample in enumerate(samples):
        token_ids.extend(sample["token_ids"])
        for start, _ in sample["character_spans"]: offsets.append(cursor + start)
        cursor += len(sample["token_ids"])
        n = len(sample["character_spans"]); component[row, :n] = torch.tensor(sample["component_labels"]); acoustic[row, :n] = torch.tensor(sample["acoustic_ids"])
    return {"sample_ids": [s["sample_id"] for s in samples], "token_ids": torch.tensor(token_ids), "token_offsets": torch.tensor(offsets),
            "char_lengths": torch.tensor([len(s["character_spans"]) for s in samples]), "component_labels": component, "acoustic_ids": acoustic}


def load_experiment(path: str | Path) -> dict:
    root = Path(path).resolve(); manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("preparation_version") != PREPARATION_VERSION:
        raise ValueError("unsupported manifest schema/preparation version")
    if manifest.get("pinyin_contract") != PINYIN_CONTRACT_VERSION or manifest.get("parser_provenance") != PARSER_PROVENANCE:
        raise ValueError("unsupported Pinyin parser contract")
    required = {"samples.jsonl", "vocab.json", "wubi_mapping.pkl", "splits.json", "config.json", "acoustic_targets.pt"}
    required |= {f"initialization/seed-{seed}.pt" for seed in manifest.get("counts", {}).get("seeds", [42])}
    if not required.issubset(manifest.get("artifacts", {})):
        raise ValueError("manifest is missing required artifact hashes")
    for relative, digest in manifest["artifacts"].items():
        artifact = (root / relative).resolve()
        if root not in artifact.parents: raise ValueError(f"unsafe artifact path: {relative}")
        if not artifact.is_file() or _sha(artifact) != digest: raise ValueError(f"artifact checksum mismatch: {relative}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8")); vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8")); splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    if vocab.get("labels") != {"initials": INITIALS, "finals": FINALS, "tones": TONES}:
        raise ValueError("experiment component vocabulary violates Pinyin contract")
    with (root / "wubi_mapping.pkl").open("rb") as handle: wubi = pickle.load(handle)
    if config.get("schema_version") != 1 or config.get("seeds") != [42] or config.get("split_seed") != 42 or config.get("split_ratios") != [0.8, 0.1, 0.1] or config.get("training", {}).get("epochs") != 20:
        raise ValueError("config violates fixed experiment protocol")
    if set(splits) != {"train", "validation", "test"} or any(not x for x in splits.values()): raise ValueError("all three splits must be non-empty")
    samples = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {s["sample_id"]: s for s in samples}; members = [x for split in splits.values() for x in split]
    if len(by_id) != len(samples) or len(members) != len(set(members)) or set(members) != set(by_id): raise ValueError("split coverage/disjointness or sample IDs invalid")
    if len({s["text"] for s in samples}) != len(samples): raise ValueError("duplicate exact text in experiment bundle")
    acoustic = torch.load(root / "acoustic_targets.pt", map_location="cpu", weights_only=True)
    norms = torch.linalg.vector_norm(acoustic.double(), dim=1)
    if acoustic.ndim != 2 or acoustic.shape[0] != len(vocab["acoustic_tokens"]) or acoustic.shape[1] != config["acoustic_dim"] or not torch.isfinite(acoustic).all() or not torch.isfinite(norms).all() or torch.any(norms <= 1e-12): raise ValueError("invalid acoustic target matrix")
    for sample in samples:
        n=len(sample["text"])
        if not sample["text"] or not sample["token_ids"]: raise ValueError("empty text or token IDs")
        if not (n == len(sample["character_spans"]) == len(sample["component_labels"]) == len(sample["acoustic_ids"]) == len(sample["char_ids"])): raise ValueError(f"sample shape mismatch: {sample['sample_id']}")
        if any(i < 0 or i >= acoustic.shape[0] for i in sample["acoustic_ids"]): raise ValueError("acoustic ID out of range")
        expected = hashlib.sha256(json.dumps([sample["text"], [vocab["acoustic_tokens"][i] for i in sample["acoustic_ids"]]], ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if sample["sample_id"] != expected: raise ValueError("sample ID hash mismatch")
        if any(i < 0 or i >= len(vocab["wubi_tokens"]) for i in sample["token_ids"]): raise ValueError("token ID out of range")
        if any(i < 0 or i >= len(vocab["char_tokens"]) for i in sample["char_ids"]): raise ValueError("char ID out of range")
        if any(any(v < 0 or v >= len(vocab["labels"][["initials","finals","tones"][j]]) for j,v in enumerate(row)) for row in sample["component_labels"]): raise ValueError("component ID out of range")
        spans=sample["character_spans"]
        if spans[0][0] != 0 or spans[-1][1] != len(sample["token_ids"]) or any(a >= b or (i and a != spans[i-1][1]) for i,(a,b) in enumerate(spans)): raise ValueError("invalid character spans")
        encoded, encoded_spans = _encode(sample["text"], wubi, {v:i for i,v in enumerate(vocab["wubi_tokens"])})
        if encoded != sample["token_ids"] or encoded_spans != spans: raise ValueError("text encoding disagrees with bundled Wubi mapping")
        expected_components = [list(decompose(validate_canonical_label(vocab["acoustic_tokens"][i]))) for i in sample["acoustic_ids"]]
        if sample["component_labels"] != expected_components: raise ValueError("component labels disagree with acoustic labels")
    train_chars = sorted({ch for sid in splits["train"] for ch in by_id[sid]["text"]})
    if vocab["char_tokens"] != ["[PAD]", "[UNK]", *train_chars]: raise ValueError("character vocabulary is not exactly train-derived")
    char_stoi = {char:i for i,char in enumerate(vocab["char_tokens"])}
    for sample in samples:
        if sample["char_ids"] != [char_stoi.get(ch, 1) for ch in sample["text"]]: raise ValueError("character IDs disagree with train-derived vocabulary")
    checkpoints = {}
    for seed in config["seeds"]:
        checkpoint = torch.load(root / f"initialization/seed-{seed}.pt", map_location="cpu", weights_only=True)
        expected_sources = {k:v["sha256"] for k,v in manifest["sources"].items()}
        if checkpoint["seed"] != seed or checkpoint["config_sha256"] != _sha(root / "config.json") or checkpoint["vocab_sha256"] != _sha(root / "vocab.json") or checkpoint.get("source_sha256") != expected_sources: raise ValueError("initialization provenance mismatch")
        model = build_encoder(config, vocab); model.load_state_dict(checkpoint["state_dict"], strict=True); checkpoints[int(seed)] = checkpoint
    return {"root": root, "manifest": manifest, "config": config, "vocab": vocab, "splits": splits, "samples": by_id, "acoustic_targets": acoustic, "initializations": checkpoints}


def create_loaders(experiment: dict, seed: int, batch_size: int | None = None, num_workers: int = 0) -> dict[str, DataLoader]:
    batch = batch_size or int(experiment["config"]["training"]["batch_size"])
    result = {}
    for stream, split in enumerate(("train", "validation", "test"), start=1):
        dataset = ExperimentDataset([experiment["samples"][x] for x in experiment["splits"][split]])
        result[split] = DataLoader(dataset, batch_size=batch, shuffle=split == "train", generator=make_generator(seed, stream), num_workers=num_workers, collate_fn=_collate)
    return result
