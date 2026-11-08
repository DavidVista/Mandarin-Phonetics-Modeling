"""Configuration-driven stages shared by the CLI and portable notebooks."""
from __future__ import annotations

import json
from pathlib import Path

from .artifacts import safe_artifact, sha256_file


def load_config(path):
    path = Path(path).absolute()
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "paths", "corpus", "model", "acoustic", "experiment_id", "experiment"}
    version = value.get('schema_version') if isinstance(value, dict) else None
    if version == 2:
        required |= {'execution', 'analysis', 'visualization', 'notebooks'}
    if not isinstance(value, dict) or set(value) != required or type(version) is not int or version not in (1, 2):
        raise ValueError("invalid pipeline configuration schema")
    paths = value["paths"]
    names = {"source", "export", "inventory", "corpus", "acoustic", "prepared", "experiments", "cache", "runs"}
    if version == 2:
        names |= {'state', 'generated', 'reports'}
    if not isinstance(paths, dict) or set(paths) != names:
        raise ValueError("pipeline paths must explicitly select all stage destinations")
    def resolve(raw):
        if not isinstance(raw, str) or not raw:
            raise ValueError("paths must be nonempty strings")
        p = Path(raw).expanduser()
        return str((path.parent / p).resolve()) if not p.is_absolute() else str(p.resolve())
    value["paths"] = {key: resolve(raw) for key, raw in paths.items()}
    if len(set(value["paths"].values())) != len(names):
        raise ValueError("stage output directories must be distinct")
    acoustic = value["acoustic"]
    if not isinstance(acoustic, dict) or set(acoustic) != {"existing_bundle", "manifest_sha256"}:
        raise ValueError("acoustic config requires existing_bundle and manifest_sha256, or both null")
    if version == 1 and (acoustic["existing_bundle"] is None) != (acoustic["manifest_sha256"] is None):
        raise ValueError("acoustic bundle and manifest pin must be selected together")
    if acoustic["existing_bundle"] is not None:
        acoustic["existing_bundle"] = resolve(acoustic["existing_bundle"])
    elif acoustic['manifest_sha256'] is not None:
        raise ValueError('manifest pin requires an existing acoustic bundle')
    if not isinstance(value["experiment"], dict) or "sample_size" not in value["experiment"]:
        raise ValueError("explicit experiment.sample_size is required")
    if version == 2:
        from .execution import validate_execution_config
        validate_execution_config(value, resolve)
    from .experiment import _validate_config
    _validate_config(value['experiment'], value['experiment']['architecture']['acoustic_dim'])
    return value


def export_corpus(config):
    from .source_preparation.duyu import export_duyu, _validate_args
    spec = config["corpus"]
    if set(spec) != {"revision", "split", "columns", "tone_convention"}:
        raise ValueError("invalid corpus source parameters")
    _validate_args(**spec)
    target = Path(config["paths"]["export"])
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir():
            raise ValueError("corpus export must be a non-symlink directory")
        provenance = json.loads(safe_artifact(target, "provenance.json").read_text(encoding="utf-8"))
        if any(provenance.get(k) != v for k, v in spec.items()):
            raise ValueError("corpus export parameters changed; use a new output ID")
        required = {"corpus.csv", "source_rows.jsonl", "rejections.jsonl", "metrics.json"}
        if set(provenance.get("files", {})) != required:
            raise ValueError("corpus export is incomplete")
        for name, digest in provenance["files"].items():
            if sha256_file(safe_artifact(target, name)) != digest:
                raise ValueError("corpus export checksum mismatch")
        return target
    return export_duyu(target, **spec, allow_download=True)


def run_stage(config, stage, *, allow_inference=False):
    paths = config["paths"]
    if stage == "export-corpus":
        return export_corpus(config)
    if stage == "download-sounds":
        from .source_preparation.download import download_sounds
        return download_sounds(paths["source"])
    if stage == "prepare-inventory":
        from .sound_inventory import prepare_sound_inventory
        return prepare_sound_inventory(paths["source"], paths["inventory"])
    if stage == "prepare-corpus":
        from .corpus import prepare_character_corpus
        export = export_corpus(config)
        return prepare_character_corpus(export / "corpus.csv", paths["inventory"], paths["corpus"],
                                        expected_sha256=sha256_file(export / "corpus.csv"))
    if stage == "prepare-embeddings":
        spec = dict(config["acoustic"])
        if spec['existing_bundle'] is not None and spec['manifest_sha256'] is None:
            spec['manifest_sha256'] = sha256_file(Path(spec['existing_bundle']) / 'manifest.json')
        if spec["existing_bundle"] is None and not allow_inference:
            raise PermissionError("real acoustic inference requires --allow-inference")
        from .acoustic import prepare_acoustic_targets
        return prepare_acoustic_targets(paths["inventory"], paths["acoustic"], model=config["model"],
                                        cache_root=paths["cache"], existing_bundle=spec["existing_bundle"],
                                        existing_manifest_sha256=spec["manifest_sha256"])
    if stage == "prepare-data":
        from .acoustic import assemble_character_data
        return assemble_character_data(paths["corpus"], paths["acoustic"], paths["prepared"])
    if stage == "prepare-experiment":
        from .experiment import prepare_experiment
        return prepare_experiment(config["experiment_id"], prepared_root=paths["prepared"],
                                  config=config["experiment"], output_root=paths["experiments"])
    if stage == "prepare":
        if config["acoustic"]["existing_bundle"] is None and not allow_inference:
            raise PermissionError("preparation would require inference; select an existing bundle or --allow-inference")
        result = None
        for name in ("export-corpus", "download-sounds", "prepare-inventory", "prepare-corpus",
                     "prepare-embeddings", "prepare-data", "prepare-experiment"):
            result = run_stage(config, name, allow_inference=allow_inference)
        return result
    raise ValueError(f"unknown preparation stage: {stage}")
