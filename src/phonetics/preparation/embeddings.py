"""Full-inventory acoustic embedding generation with deterministic caching."""

from __future__ import annotations

import hashlib
import json
import pickle
import os
import uuid
import wave
from pathlib import Path

from ..artifacts import sha256_file
from ..data_contract import validate_acoustic


def cache_key(entry, model, backend_identity):
    payload = {"audio_sha256": entry["sha256"], "model": model, "backend": backend_identity}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def embed_inventory(inventory, model_config, cache_root, backend=None):
    import numpy as np

    cache_root = Path(cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    result, computed, cached = {}, 0, 0
    backend_identity = getattr(backend, "cache_id", None) if backend is not None else "transformers"
    use_cache = backend is None or backend_identity is not None
    real_backend = backend
    for entry in inventory:
        key = cache_key(entry, model_config, backend_identity or "uncached-stub")
        cache_path = cache_root / f"{key}.pkl"
        if use_cache and cache_path.is_file():
            with cache_path.open("rb") as handle:
                vector = pickle.load(handle)
            cached += 1
        else:
            if real_backend is None:
                real_backend = TransformersBackend(model_config)
            vector = real_backend(entry, model_config)
            vector = np.asarray(vector, dtype=np.float32)
            validate_acoustic({entry["label"]: vector})
            if use_cache:
                temporary = cache_path.with_name(cache_path.name + f".{uuid.uuid4().hex}.tmp")
                with temporary.open("wb") as handle:
                    pickle.dump(vector, handle, protocol=pickle.HIGHEST_PROTOCOL)
                if cache_path.exists():
                    temporary.unlink()
                else:
                    os.rename(temporary, cache_path)
            computed += 1
        result[entry["label"]] = np.asarray(vector, dtype=np.float32)
    validate_acoustic(result, expected_keys=[row["label"] for row in inventory])
    return result, {"computed": computed, "cached": cached, "errors": 0,
                    "backend": "stub" if backend is not None else "transformers"}


class TransformersBackend:
    def __init__(self, config):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for real inference") from exc
        self.torch = torch
        processor_id = config.get("processor") or config["id"]
        self.processor = AutoProcessor.from_pretrained(processor_id, revision=config["revision"])
        self.model = AutoModel.from_pretrained(config["id"], revision=config["revision"]).to(config["device"])
        self.model.eval()

    def __call__(self, entry, config):
        import numpy as np
        samples, sample_rate = _read_audio(Path(entry["path"]), mono=config["mono"])
        if sample_rate != config["sample_rate"]:
            try:
                import librosa
            except ImportError as exc:
                raise RuntimeError("librosa is required for resampling") from exc
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=config["sample_rate"])
        if config["padding"] == "legacy_one_second" and len(samples) < config["sample_rate"]:
            samples = np.pad(samples, (0, config["sample_rate"] - len(samples)))
        inputs = self.processor(samples, sampling_rate=config["sample_rate"], return_tensors="pt",
                                padding=config["padding"] != "none")
        inputs = {key: value.to(config["device"]) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=True)
        hidden = output.hidden_states[int(config["layer"])]
        if config["pooling"] == "mean_all":
            pooled = hidden.mean(dim=1)
        else:
            if "attention_mask" not in inputs or not hasattr(self.model, "_get_feature_vector_attention_mask"):
                raise ValueError("mean_valid requires model feature-vector attention-mask support")
            mask = self.model._get_feature_vector_attention_mask(hidden.shape[1], inputs["attention_mask"])
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return pooled[0].detach().cpu().to(self.torch.float32).numpy()


def _read_audio(path, *, mono):
    import numpy as np
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            channels, width, rate = handle.getnchannels(), handle.getsampwidth(), handle.getframerate()
            raw = handle.readframes(handle.getnframes())
        if width != 2:
            raise ValueError("stdlib WAV reader supports only 16-bit PCM")
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            values = values.reshape(-1, channels)
            if mono:
                values = values.mean(axis=1)
            else:
                raise ValueError("multi-channel input requires mono=true")
        return values, rate
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("librosa is required for non-WAV audio") from exc
    return librosa.load(str(path), sr=None, mono=mono)
