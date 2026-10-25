"""Transactional prepared-dataset construction and validation."""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import platform
import shutil
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

from ..artifacts import read_jsonl, safe_artifact, sha256_file, write_json, write_jsonl
from ..data_contract import DATA_CONTRACT_VERSION, FILTERS, pair_reason, sample_id, validate_acoustic, validate_wubi
from ..pinyin import PARSER_PROVENANCE, PINYIN_CONTRACT_VERSION, decompose, validate_canonical_label
from .config import SCHEMA_VERSION, validate_config
from .acoustic_bundle import load_acoustic_bundle
from .embeddings import embed_inventory
from .readers import read_corpus, read_sounds, read_wubi
from .corpus_processing import (CorpusConflictError, coverage as corpus_coverage,
                                deduplicate_rows, filter_rows, normalize_rows, sample_rows)

REQUIRED = {"corpus/train.csv", "embeddings/pinyin_embeddings.pkl", "wubi/chinese_to_wubi.pkl",
            "config.json", "reports/run_statistics.json", "reports/coverage.json", "reports/rejections.jsonl",
            "reports/selection.jsonl", "reports/sound_inventory.jsonl"}
STAGE_ORDER = ["load_corpus", "normalize_identity", "sound_inventory", "validate_wubi",
               *FILTERS, "deduplicate", "sampling", "embeddings", "final_validation"]
STAGE_UNITS = {"load_corpus": "rows", "normalize_identity": "rows", "sound_inventory": "files",
               "validate_wubi": "entries", **{name: "rows" for name in FILTERS},
               "deduplicate": "rows", "sampling": "rows", "embeddings": "syllables",
               "final_validation": "artifacts"}


class StageFailure(ValueError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def prepare_dataset(config: dict, *, embedding_backend=None) -> Path:
    config = validate_config(config)
    if "acoustic_bundle" in config and embedding_backend is not None:
        raise ValueError("embedding_backend is invalid in precomputed mode")
    output = Path(config["output_root"]).resolve(); output.mkdir(parents=True, exist_ok=True)
    target = (output / config["dataset_id"]).resolve()
    if target.parent != output or target.exists():
        raise FileExistsError(f"prepared dataset already exists: {target}")
    temp = Path(tempfile.mkdtemp(prefix=f".{config['dataset_id']}.", dir=output))
    run_id = str(uuid.uuid4())
    write_json(temp / "config.json", {"schema_version": SCHEMA_VERSION, **config})
    config_sha256 = sha256_file(temp / "config.json")
    stages, current = [], None
    try:
        def run(name, unit, function, known_input_count=None):
            nonlocal current
            current = {"name": name, "unit": unit, "input_count": known_input_count,
                       "output_count": None, "rejected": None, "status": "not-run",
                       "counts_complete": False, "duration_seconds": 0.0}
            stages.append(current); started = time.perf_counter()
            try:
                value, counts = function(); current.update(counts); current["status"] = "success"
                current["counts_complete"] = True
                return value
            except Exception as exc:
                current["status"] = "failure"; raise
            except BaseException:
                current["status"] = "failure"; raise
            finally:
                if current["status"] == "failure":
                    exc = sys.exc_info()[1]
                    diagnostics = getattr(exc, "diagnostics", None)
                    if diagnostics:
                        current.update(diagnostics)
                    else:
                        current["counts_reason"] = "operation aborted before complete counts were available"
                current["duration_seconds"] = time.perf_counter() - started
        corpus, corpus_src = run("load_corpus", "rows", lambda: _loaded(read_corpus(config["corpus"])))
        corpus = run("normalize_identity", "rows", lambda: _normalize(corpus, config["corpus"]["tone_convention"]), len(corpus))
        if "acoustic_bundle" in config:
            vectors, sounds, acoustic_manifest = load_acoustic_bundle(config["acoustic_bundle"]["path"],config["acoustic_bundle"]["manifest_sha256"])
            sounds_src={"path":str(Path(config["acoustic_bundle"]["path"]).resolve()),"manifest_sha256":config["acoustic_bundle"]["manifest_sha256"]}
            sounds = run("sound_inventory", "files", lambda: (sounds,{"input_count":len(sounds),"output_count":len(sounds),"rejected":0}))
        else:
            sounds, sounds_src = run("sound_inventory", "files", lambda: _loaded(read_sounds(config["sounds"])))
        wubi, wubi_src = run("validate_wubi", "entries", lambda: _loaded(read_wubi(config["wubi"])))
        filtered, rejections = _filter(corpus, wubi, {row["label"] for row in sounds}, stages)
        deduped = run("deduplicate", "rows", lambda: _deduplicate(filtered), len(filtered))
        selected = run("sampling", "rows", lambda: _sample(deduped, config["sampling"]), len(deduped))
        if "acoustic_bundle" in config:
            origin=acoustic_manifest["model"].get("backend")
            inference={"computed":0,"cached":len(sounds),"errors":0,"backend":"precomputed","origin_backend":origin}
            embeddings=run("embeddings","syllables",lambda:(vectors,{"input_count":len(sounds),"output_count":len(vectors),"rejected":0}),len(sounds))
        else:
            embeddings, inference = run("embeddings", "syllables", lambda: _embeddings(sounds, config, embedding_backend), len(sounds))
        _write_bundle(temp, config, deduped, selected, wubi, embeddings, sounds, rejections)
        stages.append({"name": "final_validation", "unit": "artifacts", "input_count": len(REQUIRED),
                       "output_count": len(REQUIRED), "rejected": 0, "status": "success",
                       "counts_complete": True, "duration_seconds": 0.0})
        stats = {"schema_version": 1, "run_id": run_id, "dataset_id": config["dataset_id"], "status": "success",
                 "config_sha256": config_sha256, "stages": stages,
                 "rejections": dict(sorted(Counter(row["reason"] for row in rejections).items())),
                 "sampling": {**config["sampling"], "eligible": len(deduped), "selected": len(selected)},
                 "inventory": {"sound_files": len(sounds), "unique_labels": len({r['label'] for r in sounds})},
                 "outputs": {"csv_rows": len(selected), "embeddings": len(embeddings), "wubi_entries": len(wubi)},
                 "inference": inference}
        write_json(temp / "reports/run_statistics.json", stats)
        sources = {"corpus": corpus_src, "acoustic_bundle" if "acoustic_bundle" in config else "sounds": sounds_src, "wubi": wubi_src}
        _write_manifest(temp, config, sources, inference, acoustic_manifest if "acoustic_bundle" in config else None)
        started = time.perf_counter()
        try:
            validate_prepared(temp)
        except Exception:
            stages[-1]["status"] = "failure"
            raise
        finally:
            stages[-1]["duration_seconds"] = time.perf_counter() - started
        write_json(temp / "reports/run_statistics.json", stats); _write_manifest(temp, config, sources, inference, acoustic_manifest if "acoustic_bundle" in config else None)
        validate_prepared(temp)
        if target.exists(): raise FileExistsError(f"prepared dataset already exists: {target}")
        os.rename(temp, target)
        return target
    except Exception as exc:
        failure = output / f"{config['dataset_id']}.failed.json"
        recorded={row["name"] for row in stages}
        stages.extend({"name":name,"unit":STAGE_UNITS[name],"input_count":None,"output_count":None,"rejected":None,
                       "status":"not-run","counts_complete":False,"counts_reason":"blocked by earlier failure",
                       "duration_seconds":0.0} for name in STAGE_ORDER if name not in recorded)
        stages.sort(key=lambda row: STAGE_ORDER.index(row["name"]))
        write_json(failure, {"schema_version":1,"run_id":run_id,"dataset_id": config["dataset_id"],
                             "config_sha256":config_sha256,"status":"failure","exception": type(exc).__name__,
                             "message": str(exc), "details":getattr(exc,"diagnostics",{}), "stages": stages})
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _loaded(result):
    value, source = result
    return (value, source), {"input_count": len(value), "output_count": len(value), "rejected": 0}


def _normalize(rows, tone_convention):
    return normalize_rows(rows, tone_convention)


def _filter(rows, wubi, acoustic, stages):
    return filter_rows(rows, wubi, acoustic, stages)


def _deduplicate(rows):
    try:return deduplicate_rows(rows)
    except CorpusConflictError as exc:raise StageFailure(str(exc),exc.diagnostics) from exc


def _sample(rows, spec):
    size = spec["size"]
    if size is not None and size > len(rows):
        raise StageFailure("sampling size exceeds eligible pool", {"input_count":len(rows),"output_count":0,
            "rejected":0,"blocked_count":len(rows),"requested_size":size,"counts_complete":True})
    return sample_rows(rows,spec)


def _embeddings(sounds, config, backend):
    value, counts = embed_inventory(sounds, config["model"], config["cache_root"], backend)
    return (value, counts), {"input_count": len(sounds), "output_count": len(value), "rejected": counts["errors"]}


def _coverage(rows):
    return corpus_coverage(rows)


def _write_bundle(root, config, eligible, rows, wubi, acoustic, inventory, rejections):
    for directory in ("corpus", "embeddings", "wubi", "reports"): (root / directory).mkdir()
    with (root / "corpus/train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=["text","pinyin"]); writer.writeheader()
        writer.writerows({"text":r["text"], "pinyin":" ".join(r["labels"])} for r in rows)
    with (root / "embeddings/pinyin_embeddings.pkl").open("wb") as h: pickle.dump(acoustic,h,pickle.HIGHEST_PROTOCOL)
    with (root / "wubi/chinese_to_wubi.pkl").open("wb") as h: pickle.dump(wubi,h,pickle.HIGHEST_PROTOCOL)
    write_json(root/"config.json", {"schema_version": SCHEMA_VERSION, **config})
    write_jsonl(root/"reports/selection.jsonl", ({k:r[k] for k in ("sample_id","text","labels","source_ids")} for r in rows))
    write_jsonl(root/"reports/rejections.jsonl", rejections); write_jsonl(root/"reports/sound_inventory.jsonl", inventory)
    write_json(root/"reports/coverage.json", {"before_sampling": _coverage(eligible), "after_sampling": _coverage(rows)})


def _write_manifest(root, config, sources, inference, acoustic_manifest=None):
    artifacts={str(p.relative_to(root)).replace("\\","/"):sha256_file(p) for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"}
    code_root=Path(__file__).parents[1]
    code_files=[code_root/"pinyin.py",code_root/"data_contract.py",code_root/"artifacts.py", *sorted(Path(__file__).parent.glob("*.py"))]
    code={str(p.relative_to(code_root)).replace("\\","/"):sha256_file(p) for p in code_files}
    environment={"python":platform.python_version()}
    try:
        import numpy; environment["numpy"] = numpy.__version__
    except ImportError: pass
    try:
        import torch; environment["torch"] = torch.__version__
    except ImportError: pass
    write_json(root/"manifest.json", {"bundle_type":"prepared_dataset","schema_version":SCHEMA_VERSION,"dataset_id":config["dataset_id"],
        "data_contract":DATA_CONTRACT_VERSION,"pinyin_contract":PINYIN_CONTRACT_VERSION,"parser_provenance":PARSER_PROVENANCE,"manifest_self_hash":"excluded",
        "artifacts":artifacts,"sources":sources,"code":code,"environment":environment,
        "model":({**config["model"],"backend":inference["backend"]} if "model" in config else
                 {**acoustic_manifest["model"],"backend":"precomputed","origin_backend":acoustic_manifest["model"].get("backend")}),
        "acoustic_bundle":({"bundle_id":acoustic_manifest["bundle_id"],"manifest_sha256":config["acoustic_bundle"]["manifest_sha256"]} if acoustic_manifest else None)})


def validate_prepared(path: str | Path) -> dict:
    root=Path(path).resolve(); manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("bundle_type")=="prepared_dataset" and manifest.get("schema_version")==2:
        from .assembly import validate_assembled
        return validate_assembled(root,manifest)
    if manifest.get("bundle_type")!="prepared_dataset" or manifest.get("schema_version")!=SCHEMA_VERSION or manifest.get("data_contract")!=DATA_CONTRACT_VERSION or manifest.get("pinyin_contract") != PINYIN_CONTRACT_VERSION or manifest.get("parser_provenance") != PARSER_PROVENANCE:
        raise ValueError("unsupported prepared dataset contract")
    artifacts=manifest.get("artifacts",{})
    if not REQUIRED.issubset(artifacts): raise ValueError("manifest is missing required prepared artifacts")
    for relative,digest in artifacts.items():
        artifact=safe_artifact(root,relative)
        if not artifact.is_file() or sha256_file(artifact)!=digest: raise ValueError(f"artifact checksum mismatch: {relative}")
    config=json.loads((root/"config.json").read_text(encoding="utf-8")); validate_config({k:v for k,v in config.items() if k!="schema_version"})
    selection=read_jsonl(root/"reports/selection.jsonl"); inventory=read_jsonl(root/"reports/sound_inventory.jsonl")
    stats=json.loads((root/"reports/run_statistics.json").read_text(encoding="utf-8"))
    if (stats.get("schema_version") != 1 or not isinstance(stats.get("run_id"),str)
            or stats.get("dataset_id") != manifest.get("dataset_id") or stats.get("status") != "success"):
        raise ValueError("invalid run statistics identity/status")
    if stats.get("config_sha256") != sha256_file(root/"config.json"): raise ValueError("run statistics config hash mismatch")
    if len({row.get("label") for row in inventory}) != len(inventory): raise ValueError("duplicate inventory labels")
    for row in inventory: decompose(validate_canonical_label(row.get("label")))
    with (root/"wubi/chinese_to_wubi.pkl").open("rb") as h: wubi=validate_wubi(pickle.load(h),alphabet=config["wubi"].get("alphabet","abcdefghijklmnopqrstuvwxyz0123456789"))
    with (root/"embeddings/pinyin_embeddings.pkl").open("rb") as h: acoustic=pickle.load(h)
    validate_acoustic(acoustic,expected_keys=[r["label"] for r in inventory])
    outputs=stats.get("outputs",{})
    if outputs.get("embeddings") != len(acoustic) or outputs.get("wubi_entries") != len(wubi):
        raise ValueError("statistics output counts mismatch")
    rejection_rows=read_jsonl(root/"reports/rejections.jsonl")
    if stats.get("rejections") != dict(sorted(Counter(row.get("reason") for row in rejection_rows).items())):
        raise ValueError("statistics rejection counts mismatch")
    with (root/"corpus/train.csv").open(encoding="utf-8",newline="") as h:
        reader=csv.DictReader(h)
        if reader.fieldnames != ["text","pinyin"]: raise ValueError("prepared corpus headers invalid")
        rows=list(reader)
    if len(rows)!=len(selection): raise ValueError("CSV/selection row count mismatch")
    _validate_statistics(stats,config,manifest,len(rows),len(selection),len(inventory),len(wubi),len(acoustic),rejection_rows)
    for row,chosen in zip(rows,selection):
        labels=row["pinyin"].split()
        if row["text"]!=chosen.get("text") or labels!=chosen.get("labels") or sample_id(row["text"],labels)!=chosen.get("sample_id"): raise ValueError("CSV/selection semantic mismatch")
        if pair_reason(row["text"],labels,wubi,acoustic) is not None: raise ValueError("prepared row violates common contract")
    ids=[r["sample_id"] for r in selection]
    if ids!=sorted(ids) or len(ids)!=len(set(ids)) or len({r["text"] for r in selection})!=len(selection): raise ValueError("selection IDs/order/uniqueness invalid")
    return manifest


def _validate_statistics(stats, config, manifest, csv_rows, selection_rows, inventory_rows,
                         wubi_entries, embedding_entries, rejection_rows):
    if config.get("dataset_id") != manifest.get("dataset_id") or stats.get("dataset_id") != config.get("dataset_id"):
        raise ValueError("dataset IDs disagree")
    stages=stats.get("stages")
    if not isinstance(stages,list) or [row.get("name") for row in stages] != STAGE_ORDER:
        raise ValueError("stage list/order is invalid")
    by_name={row["name"]:row for row in stages}
    if len(by_name)!=len(stages): raise ValueError("duplicate stages")
    for name,row in by_name.items():
        if (row.get("unit") != STAGE_UNITS[name] or row.get("status") != "success"
                or ("counts_complete" in row and row.get("counts_complete") is not True)):
            raise ValueError(f"invalid stage metadata: {name}")
        duration=row.get("duration_seconds")
        if not isinstance(duration,(int,float)) or isinstance(duration,bool) or not math.isfinite(duration) or duration < 0:
            raise ValueError(f"invalid stage duration: {name}")
        for field in ("input_count","output_count","rejected"):
            value=row.get(field)
            if not isinstance(value,int) or isinstance(value,bool) or value < 0:
                raise ValueError(f"invalid {field}: {name}")
    for name in (*FILTERS,"deduplicate","sampling"):
        row=by_name[name]
        if row["input_count"] != row["output_count"] + row["rejected"]:
            raise ValueError(f"stage balance invalid: {name}")
    for name in ("load_corpus","normalize_identity","sound_inventory","validate_wubi","embeddings","final_validation"):
        row=by_name[name]
        if row["input_count"] != row["output_count"] or row["rejected"] != 0:
            raise ValueError(f"lossless stage balance invalid: {name}")
    final=by_name["final_validation"]
    if final["input_count"] != len(REQUIRED) or final["output_count"] != len(REQUIRED):
        raise ValueError("final validation artifact count disagrees")
    adjacency=[("load_corpus","normalize_identity"),("normalize_identity","nonempty")]
    adjacency += list(zip(FILTERS,FILTERS[1:]))
    adjacency += [(FILTERS[-1],"deduplicate"),("deduplicate","sampling")]
    for left,right in adjacency:
        if by_name[left]["output_count"] != by_name[right]["input_count"]:
            raise ValueError(f"stage adjacency invalid: {left}/{right}")
    sampling=stats.get("sampling",{}); configured=config["sampling"]
    for field in ("eligible","selected"):
        if not isinstance(sampling.get(field),int) or isinstance(sampling.get(field),bool) or sampling[field] < 0:
            raise ValueError("invalid sampling summary counters")
    if not isinstance(sampling.get("seed"),int) or isinstance(sampling.get("seed"),bool): raise ValueError("invalid sampling seed")
    size=sampling.get("size")
    if size is not None and (not isinstance(size,int) or isinstance(size,bool) or size <= 0): raise ValueError("invalid sampling size")
    if not isinstance(sampling.get("strategy"),str): raise ValueError("invalid sampling strategy")
    if sampling.get("strategy") != configured["strategy"] or sampling.get("seed") != configured["seed"] or sampling.get("size") != configured["size"]:
        raise ValueError("sampling configuration disagrees with statistics")
    eligible=by_name["deduplicate"]["output_count"]; selected=by_name["sampling"]["output_count"]
    if (sampling.get("eligible") != eligible or by_name["sampling"]["input_count"] != eligible
            or sampling.get("selected") != selected or selected > eligible
            or by_name["sampling"]["rejected"] != eligible-selected):
        raise ValueError("sampling counts disagree")
    if configured["size"] is None:
        if selected != eligible: raise ValueError("size=null must select the full eligible pool")
    elif selected != configured["size"] or configured["size"] > eligible:
        raise ValueError("sampling size was not honored")
    if selected != csv_rows or selected != selection_rows or stats.get("outputs",{}).get("csv_rows") != csv_rows:
        raise ValueError("selected/CSV/selection counts disagree")
    inventory=stats.get("inventory",{})
    for value in inventory.values():
        if not isinstance(value,int) or isinstance(value,bool) or value < 0: raise ValueError("invalid inventory counters")
    if inventory.get("sound_files") != inventory_rows or inventory.get("unique_labels") != inventory_rows:
        raise ValueError("inventory statistics disagree")
    if by_name["sound_inventory"]["output_count"] != inventory_rows:
        raise ValueError("sound inventory stage count disagrees")
    outputs=stats.get("outputs",{})
    for value in outputs.values():
        if not isinstance(value,int) or isinstance(value,bool) or value < 0: raise ValueError("invalid output counters")
    if (by_name["validate_wubi"]["output_count"] != wubi_entries or outputs.get("wubi_entries") != wubi_entries
            or by_name["embeddings"]["input_count"] != inventory_rows
            or by_name["embeddings"]["output_count"] != embedding_entries
            or outputs.get("embeddings") != embedding_entries):
        raise ValueError("Wubi/embedding statistics disagree")
    inference=stats.get("inference",{})
    for field in ("computed","cached","errors"):
        value=inference.get(field)
        if not isinstance(value,int) or isinstance(value,bool) or value < 0: raise ValueError("invalid inference counters")
    if inference["computed"]+inference["cached"] != inventory_rows or inference["errors"] != 0:
        raise ValueError("inference coverage disagrees")
    recorded_rejections=stats.get("rejections",{})
    if not isinstance(recorded_rejections,dict) or any(not isinstance(v,int) or isinstance(v,bool) or v < 0 for v in recorded_rejections.values()):
        raise ValueError("invalid rejection counters")
    reasons=Counter(row.get("reason") for row in rejection_rows)
    if stats.get("rejections") != dict(sorted(reasons.items())): raise ValueError("global rejection statistics disagree")
    for name in FILTERS:
        if by_name[name]["rejected"] != reasons.get(name,0): raise ValueError(f"filter rejection count disagrees: {name}")
