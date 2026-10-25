"""Strict assembly of already processed corpus and acoustic artifacts."""

from __future__ import annotations
import json,math,os,pickle,platform,shutil,tempfile,time,uuid
from pathlib import Path
from ..artifacts import read_jsonl,safe_artifact,sha256_file,write_json
from ..data_contract import DATA_CONTRACT_VERSION,pair_reason,validate_acoustic,validate_wubi
from ..pinyin import PARSER_PROVENANCE, PINYIN_CONTRACT_VERSION
from .acoustic_bundle import load_acoustic_bundle
from .corpus_bundle import validate_corpus_bundle
from .readers import read_wubi

ASSEMBLY_CONTRACT_VERSION="assembled-1"
REQUIRED={"corpus/train.csv","embeddings/pinyin_embeddings.pkl","wubi/chinese_to_wubi.pkl","config.json",
          "reports/run_statistics.json","reports/coverage.json","reports/rejections.jsonl","reports/selection.jsonl",
          "reports/sound_inventory.jsonl","reports/processed_corpus_manifest.json","reports/acoustic_bundle_manifest.json"}

def _config(value):
    keys={"dataset_id","output_root","processed_corpus","wubi","acoustic_bundle"}
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError("invalid assembly config keys")
    value=json.loads(json.dumps(value));did=value["dataset_id"]
    if not isinstance(did,str) or not did or Path(did).name!=did:raise ValueError("invalid dataset_id")
    for name in ("processed_corpus","acoustic_bundle"):
        if set(value[name])!={"path","manifest_sha256"}:raise ValueError(f"invalid {name} pin")
    if not {"path","sha256"}.issubset(value["wubi"]) or set(value["wubi"])-{"path","sha256","alphabet"}:raise ValueError("invalid wubi pin")
    return value

def assemble_dataset(config):
    config=_config(config);out=Path(config["output_root"]).resolve();out.mkdir(parents=True,exist_ok=True);target=out/config["dataset_id"]
    if target.exists():raise FileExistsError(target)
    corpus_root=Path(config["processed_corpus"]["path"]).resolve();started=time.perf_counter();cm=validate_corpus_bundle(corpus_root,config["processed_corpus"]["manifest_sha256"]);corpus_duration=time.perf_counter()-started
    started=time.perf_counter();vectors,inventory,am=load_acoustic_bundle(config["acoustic_bundle"]["path"],config["acoustic_bundle"]["manifest_sha256"]);acoustic_duration=time.perf_counter()-started
    started=time.perf_counter();wubi,_=read_wubi(config["wubi"]);wubi_duration=time.perf_counter()-started
    pins=cm["sources"]
    if pins.get("wubi",{}).get("sha256")!=config["wubi"]["sha256"]:raise ValueError("processed corpus Wubi pin mismatch")
    if pins.get("sound_inventory_sha256")!=am.get("inventory_sha256"):raise ValueError("processed corpus sound inventory pin mismatch")
    temp=Path(tempfile.mkdtemp(prefix=f".{target.name}.",dir=out));started=time.perf_counter()
    try:
        for d in ("corpus","embeddings","wubi","reports"):(temp/d).mkdir()
        shutil.copyfile(corpus_root/"corpus.csv",temp/"corpus/train.csv")
        for name in ("selection.jsonl","rejections.jsonl","coverage.json"):shutil.copyfile(corpus_root/f"reports/{name}",temp/f"reports/{name}")
        shutil.copyfile(corpus_root/"manifest.json",temp/"reports/processed_corpus_manifest.json")
        acoustic_root=Path(config["acoustic_bundle"]["path"]).resolve()
        shutil.copyfile(acoustic_root/"pinyin_embeddings.pkl",temp/"embeddings/pinyin_embeddings.pkl")
        shutil.copyfile(acoustic_root/"sound_inventory.jsonl",temp/"reports/sound_inventory.jsonl")
        shutil.copyfile(acoustic_root/"manifest.json",temp/"reports/acoustic_bundle_manifest.json")
        shutil.copyfile(Path(config["wubi"]["path"]),temp/"wubi/chinese_to_wubi.pkl")
        write_json(temp/"config.json",{"schema_version":2,**config})
        config_sha256=sha256_file(temp/"config.json")
        selected=len(read_jsonl(temp/"reports/selection.jsonl"));run_id=str(uuid.uuid4())
        stages=[{"name":"validate_processed_corpus","unit":"rows","input_count":selected,"output_count":selected,"rejected":0,"status":"success","duration_seconds":corpus_duration},
                {"name":"validate_wubi","unit":"entries","input_count":len(wubi),"output_count":len(wubi),"rejected":0,"status":"success","duration_seconds":wubi_duration},
                {"name":"validate_acoustic_bundle","unit":"syllables","input_count":len(inventory),"output_count":len(vectors),"rejected":0,"status":"success","duration_seconds":acoustic_duration},
                {"name":"assemble","unit":"rows","input_count":selected,"output_count":selected,"rejected":0,"status":"success","duration_seconds":time.perf_counter()-started}]
        stats={"schema_version":2,"assembly_contract":ASSEMBLY_CONTRACT_VERSION,"run_id":run_id,"dataset_id":config["dataset_id"],"config_sha256":config_sha256,"status":"success","stages":stages,"outputs":{"csv_rows":selected,"embeddings":len(vectors),"wubi_entries":len(wubi)}}
        write_json(temp/"reports/run_statistics.json",stats)
        artifacts={str(p.relative_to(temp)).replace("\\","/"):sha256_file(p) for p in temp.rglob("*") if p.is_file()}
        write_json(temp/"manifest.json",{"bundle_type":"prepared_dataset","schema_version":2,"assembly_contract":ASSEMBLY_CONTRACT_VERSION,
            "dataset_id":config["dataset_id"],"data_contract":DATA_CONTRACT_VERSION,"pinyin_contract":PINYIN_CONTRACT_VERSION,"parser_provenance":PARSER_PROVENANCE,
            "artifacts":artifacts,"processed_corpus":{"manifest_sha256":config["processed_corpus"]["manifest_sha256"],"corpus_contract":cm["corpus_contract"]},
            "acoustic_bundle":{"manifest_sha256":config["acoustic_bundle"]["manifest_sha256"],"bundle_id":am["bundle_id"]},
            "pins":{"wubi_sha256":config["wubi"]["sha256"],"sound_inventory_sha256":am["inventory_sha256"]},
            "model":{**am["model"],"backend":"precomputed","origin_backend":am["model"].get("backend")},"environment":{"python":platform.python_version()}})
        from .pipeline import validate_prepared
        validate_prepared(temp);os.rename(temp,target);return target
    except Exception:
        shutil.rmtree(temp,ignore_errors=True);raise

def validate_assembled(root,manifest):
    root=Path(root).resolve()
    if (manifest.get("bundle_type")!="prepared_dataset" or manifest.get("schema_version")!=2 or manifest.get("assembly_contract")!=ASSEMBLY_CONTRACT_VERSION
            or manifest.get("data_contract")!=DATA_CONTRACT_VERSION or manifest.get("pinyin_contract") != PINYIN_CONTRACT_VERSION or manifest.get("parser_provenance") != PARSER_PROVENANCE):raise ValueError("unsupported assembly contract")
    arts=manifest.get("artifacts",{})
    if not REQUIRED.issubset(arts):raise ValueError("assembled dataset artifacts missing")
    for rel,digest in arts.items():
        p=safe_artifact(root,rel)
        if not p.is_file() or sha256_file(p)!=digest:raise ValueError(f"assembled artifact mismatch: {rel}")
    config=json.loads((root/"config.json").read_text(encoding="utf8"))
    if config.get("schema_version")!=2 or config.get("dataset_id")!=manifest.get("dataset_id"):raise ValueError("assembly config identity invalid")
    _config({k:v for k,v in config.items() if k!="schema_version"})
    if config.get("processed_corpus",{}).get("manifest_sha256")!=manifest["processed_corpus"]["manifest_sha256"] or config.get("wubi",{}).get("sha256")!=manifest["pins"]["wubi_sha256"] or config.get("acoustic_bundle",{}).get("manifest_sha256")!=manifest["acoustic_bundle"]["manifest_sha256"]:raise ValueError("assembly config pins invalid")
    upstream=json.loads((root/"reports/processed_corpus_manifest.json").read_text(encoding="utf8"))
    if sha256_file(root/"reports/processed_corpus_manifest.json")!=manifest["processed_corpus"]["manifest_sha256"]:raise ValueError("processed corpus manifest pin mismatch")
    if upstream["artifacts"]["corpus.csv"]!=sha256_file(root/"corpus/train.csv"):raise ValueError("assembled corpus is not byte-identical")
    for name in ("selection.jsonl","rejections.jsonl","coverage.json"):
        if upstream["artifacts"][f"reports/{name}"]!=sha256_file(root/f"reports/{name}"):raise ValueError("processed corpus sidecar mismatch")
    if upstream["sources"]["wubi"]["sha256"]!=manifest["pins"]["wubi_sha256"] or sha256_file(root/"wubi/chinese_to_wubi.pkl")!=manifest["pins"]["wubi_sha256"]:raise ValueError("assembled Wubi pin mismatch")
    if upstream["sources"]["sound_inventory_sha256"]!=manifest["pins"]["sound_inventory_sha256"] or sha256_file(root/"reports/sound_inventory.jsonl")!=manifest["pins"]["sound_inventory_sha256"]:raise ValueError("assembled inventory pin mismatch")
    acoustic=json.loads((root/"reports/acoustic_bundle_manifest.json").read_text(encoding="utf8"))
    if sha256_file(root/"reports/acoustic_bundle_manifest.json")!=manifest["acoustic_bundle"]["manifest_sha256"]:raise ValueError("assembled acoustic manifest pin mismatch")
    if (acoustic.get("artifacts",{}).get("pinyin_embeddings.pkl")!=sha256_file(root/"embeddings/pinyin_embeddings.pkl")
            or acoustic.get("artifacts",{}).get("sound_inventory.jsonl")!=sha256_file(root/"reports/sound_inventory.jsonl")):raise ValueError("assembled acoustic artifacts are not byte-identical")
    with (root/"wubi/chinese_to_wubi.pkl").open("rb") as h:wubi=validate_wubi(pickle.load(h),alphabet=config["wubi"].get("alphabet","abcdefghijklmnopqrstuvwxyz0123456789"))
    with (root/"embeddings/pinyin_embeddings.pkl").open("rb") as h:vectors=pickle.load(h)
    inventory=read_jsonl(root/"reports/sound_inventory.jsonl");validate_acoustic(vectors,expected_keys=[x["label"] for x in inventory])
    import csv
    with (root/"corpus/train.csv").open(encoding="utf8",newline="") as h:rows=list(csv.DictReader(h))
    selection=read_jsonl(root/"reports/selection.jsonl")
    if len(rows)!=len(selection):raise ValueError("assembled corpus/selection count mismatch")
    for row,chosen in zip(rows,selection):
        labels=row["pinyin"].split()
        if row["text"]!=chosen.get("text") or labels!=chosen.get("labels") or pair_reason(row["text"],labels,wubi,vectors) is not None:raise ValueError("assembled row contract mismatch")
    stats=json.loads((root/"reports/run_statistics.json").read_text(encoding="utf8"))
    if stats.get("schema_version")!=2 or stats.get("assembly_contract")!=ASSEMBLY_CONTRACT_VERSION or stats.get("status")!="success" or stats.get("dataset_id")!=manifest.get("dataset_id") or stats.get("config_sha256")!=sha256_file(root/"config.json"):raise ValueError("assembly statistics invalid")
    if stats.get("outputs")!={"csv_rows":len(rows),"embeddings":len(vectors),"wubi_entries":len(wubi)}:raise ValueError("assembly output counts invalid")
    expected=["validate_processed_corpus","validate_wubi","validate_acoustic_bundle","assemble"]
    stages=stats.get("stages")
    if not isinstance(stages,list) or [x.get("name") for x in stages]!=expected:raise ValueError("assembly stage order invalid")
    expected_units={"validate_processed_corpus":"rows","validate_wubi":"entries","validate_acoustic_bundle":"syllables","assemble":"rows"}
    for stage in stages:
        if stage.get("status")!="success" or stage.get("unit")!=expected_units[stage["name"]]:raise ValueError("assembly stage metadata invalid")
        if any(not isinstance(stage.get(k),int) or isinstance(stage.get(k),bool) or stage[k]<0 for k in ("input_count","output_count","rejected")):raise ValueError("assembly stage counts invalid")
        if stage["input_count"]!=stage["output_count"] or stage["rejected"]!=0:raise ValueError("assembly stage balance invalid")
        duration=stage.get("duration_seconds")
        if not isinstance(duration,(int,float)) or isinstance(duration,bool) or not math.isfinite(duration) or duration<0:raise ValueError("assembly duration invalid")
    if stages[0]["output_count"]!=len(rows) or stages[1]["output_count"]!=len(wubi) or stages[2]["output_count"]!=len(vectors) or stages[3]["output_count"]!=len(rows):raise ValueError("assembly stage output mismatch")
    return manifest
