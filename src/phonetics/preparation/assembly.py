"""Assemble validated processed IDS records with a pinned acoustic bundle."""
from __future__ import annotations
import json,os,pickle,platform,shutil,tempfile,uuid
from pathlib import Path
from ..artifacts import read_jsonl,safe_artifact,sha256_file,write_json
from ..ids import IDS_RULES_VERSION,build_mapping
from .acoustic_bundle import load_acoustic_bundle
from .corpus_bundle import validate_corpus_bundle

ASSEMBLY_CONTRACT_VERSION="assembled-ids-1"
REQUIRED={"corpus.csv","corpus.jsonl","ids/mapping.json","ids/vocabulary.json","ids/dictionary.jsonl","ids/COPYING","ids/LGPL","acoustic/targets.pkl","config.json","reports/run_statistics.json","reports/processing_statistics.json","reports/coverage.json","reports/rejections.jsonl","reports/selection.jsonl","reports/sound_inventory.jsonl","reports/processed_corpus_manifest.json","reports/processed_corpus_config.json","reports/acoustic_bundle_manifest.json"}

def _config(value):
    keys={"dataset_id","output_root","processed_corpus","acoustic_bundle"}
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError("invalid assembly config keys")
    for name in ("processed_corpus","acoustic_bundle"):
        if not isinstance(value[name],dict) or set(value[name])!={"path","manifest_sha256"}:raise ValueError(f"invalid {name} pin")
    did=value["dataset_id"]
    if not isinstance(did,str) or not did or Path(did).name!=did:raise ValueError("invalid dataset_id")
    return json.loads(json.dumps(value))

def assemble_dataset(config):
    config=_config(config);out=Path(config["output_root"]).resolve();out.mkdir(parents=True,exist_ok=True);target=out/config["dataset_id"]
    if target.exists():raise FileExistsError(target)
    cp=Path(config["processed_corpus"]["path"]).resolve(); cm=validate_corpus_bundle(cp,config["processed_corpus"]["manifest_sha256"])
    vectors,inventory,am=load_acoustic_bundle(config["acoustic_bundle"]["path"],config["acoustic_bundle"]["manifest_sha256"])
    if cm["sources"]["sound_inventory_sha256"]!=am["inventory_sha256"]:raise ValueError("processed corpus sound inventory pin mismatch")
    temp=Path(tempfile.mkdtemp(prefix=f".{target.name}.",dir=out))
    try:
        (temp/"ids").mkdir();(temp/"acoustic").mkdir();(temp/"reports").mkdir()
        for name in ("corpus.csv","corpus.jsonl"):shutil.copyfile(cp/name,temp/name)
        for name in ("mapping.json","vocabulary.json","dictionary.jsonl","COPYING","LGPL"):shutil.copyfile(cp/f"ids/{name}",temp/f"ids/{name}")
        for name in ("selection.jsonl","rejections.jsonl","coverage.json","processing_statistics.json"):shutil.copyfile(cp/f"reports/{name}",temp/f"reports/{name}")
        shutil.copyfile(cp/"manifest.json",temp/"reports/processed_corpus_manifest.json")
        shutil.copyfile(cp/"config.json",temp/"reports/processed_corpus_config.json")
        shutil.copyfile(Path(config["acoustic_bundle"]["path"]).resolve()/"sound_inventory.jsonl",temp/"reports/sound_inventory.jsonl")
        shutil.copyfile(Path(config["acoustic_bundle"]["path"]).resolve()/"manifest.json",temp/"reports/acoustic_bundle_manifest.json")
        with (temp/"acoustic/targets.pkl").open("wb") as h:pickle.dump(vectors,h,pickle.HIGHEST_PROTOCOL)
        write_json(temp/"config.json",{"schema_version":3,**config,"ids":cm["sources"]["ids"]}); records=read_jsonl(temp/"corpus.jsonl")
        write_json(temp/"reports/run_statistics.json",{"schema_version":3,"assembly_contract":ASSEMBLY_CONTRACT_VERSION,"run_id":str(uuid.uuid4()),"dataset_id":config["dataset_id"],"status":"success","config_sha256":sha256_file(temp/"config.json"),"outputs":{"rows":len(records),"ids_records":len(records),"acoustic_targets":len(vectors),"vocabulary_tokens":len(json.loads((temp/"ids/vocabulary.json").read_text(encoding="utf8"))["tokens"])}})
        arts={str(p.relative_to(temp)).replace("\\","/"):sha256_file(p) for p in temp.rglob("*") if p.is_file()}
        code={}
        for source in (Path(__file__),Path(__file__).with_name("corpus_bundle.py"),Path(__file__).with_name("corpus_processing.py"),Path(__file__).parents[1]/"ids.py",Path(__file__).parents[1]/"data_contract.py"):
            code[source.name]=sha256_file(source)
        write_json(temp/"manifest.json",{"bundle_type":"prepared_dataset","schema_version":3,"assembly_contract":ASSEMBLY_CONTRACT_VERSION,"ids_rules":IDS_RULES_VERSION,"dataset_id":config["dataset_id"],"artifacts":arts,"code_sha256":code,"processed_corpus_manifest_sha256":config["processed_corpus"]["manifest_sha256"],"acoustic_bundle_manifest_sha256":config["acoustic_bundle"]["manifest_sha256"],"ids":cm["ids"],"ids_provenance":cm["sources"]["ids"],"acoustic":{"bundle_id":am["bundle_id"],"backend":am["model"]["backend"]},"environment":{"python":platform.python_version()}})
        validate_assembled(temp);os.rename(temp,target);return target
    except Exception:
        shutil.rmtree(temp,ignore_errors=True);raise

def validate_assembled(path,manifest=None):
    root=Path(path).resolve();mp=root/"manifest.json"
    m=manifest or json.loads(mp.read_text(encoding="utf-8"))
    if m.get("bundle_type")!="prepared_dataset" or m.get("schema_version")!=3 or m.get("assembly_contract")!=ASSEMBLY_CONTRACT_VERSION or m.get("ids_rules")!=IDS_RULES_VERSION:raise ValueError("unsupported IDS prepared bundle")
    arts=m.get("artifacts",{})
    if set(arts)!=REQUIRED:raise ValueError("prepared artifact set invalid")
    actual={str(p.relative_to(root)).replace("\\","/") for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"}
    if actual!=set(arts):raise ValueError("prepared bundle contains untracked artifacts")
    for rel,digest in arts.items():
        p=safe_artifact(root,rel)
        if not p.is_file() or sha256_file(p)!=digest:raise ValueError(f"prepared artifact mismatch: {rel}")
    cfg=json.loads((root/"config.json").read_text(encoding="utf8"))
    if cfg.get("schema_version")!=3 or cfg.get("dataset_id")!=m.get("dataset_id") or set(cfg)!={"schema_version","dataset_id","output_root","processed_corpus","acoustic_bundle","ids"}:raise ValueError("prepared config identity invalid")
    processed=json.loads((root/"reports/processed_corpus_manifest.json").read_text(encoding="utf8"))
    if sha256_file(root/"reports/processed_corpus_manifest.json")!=m["processed_corpus_manifest_sha256"]:raise ValueError("processed corpus manifest pin mismatch")
    processing_config=json.loads((root/"reports/processed_corpus_config.json").read_text(encoding="utf8"))
    if sha256_file(root/"reports/processed_corpus_config.json")!=processed.get("artifacts",{}).get("config.json") or processing_config.get("corpus_id")!=processed.get("corpus_id") or processing_config.get("conflict_policy") not in {"exclude","error"}:raise ValueError("processed corpus config pin or identity mismatch")
    acoustic_manifest=json.loads((root/"reports/acoustic_bundle_manifest.json").read_text(encoding="utf8"))
    if sha256_file(root/"reports/acoustic_bundle_manifest.json")!=m.get("acoustic_bundle_manifest_sha256"):raise ValueError("acoustic bundle manifest pin mismatch")
    inventory_path=root/"reports/sound_inventory.jsonl"
    if sha256_file(inventory_path)!=processed.get("sources",{}).get("sound_inventory_sha256") or sha256_file(inventory_path)!=acoustic_manifest.get("inventory_sha256"):raise ValueError("prepared sound inventory pin mismatch")
    copied={"corpus.csv":"corpus.csv","corpus.jsonl":"corpus.jsonl","ids/mapping.json":"ids/mapping.json","ids/vocabulary.json":"ids/vocabulary.json","ids/dictionary.jsonl":"ids/dictionary.jsonl","ids/COPYING":"ids/COPYING","ids/LGPL":"ids/LGPL","reports/selection.jsonl":"reports/selection.jsonl","reports/rejections.jsonl":"reports/rejections.jsonl","reports/coverage.json":"reports/coverage.json","reports/processing_statistics.json":"reports/processing_statistics.json"}
    for dest,source in copied.items():
        if processed.get("artifacts",{}).get(source)!=sha256_file(root/dest):raise ValueError(f"processed corpus artifact pin mismatch: {dest}")
    if processed.get("ids")!=m.get("ids") or processed.get("sources",{}).get("ids")!=m.get("ids_provenance"):raise ValueError("prepared IDS provenance mismatch")
    if cfg.get("ids")!=m.get("ids_provenance"):raise ValueError("prepared config IDS provenance mismatch")
    if m["ids"].get("license_notices_sha256")!={name:sha256_file(root/f"ids/{name}") for name in ("COPYING","LGPL")}:raise ValueError("prepared IDS license notice mismatch")
    mapping=json.loads((root/"ids/mapping.json").read_text(encoding="utf8"));vocab=json.loads((root/"ids/vocabulary.json").read_text(encoding="utf8"))
    rebuilt,rebuilt_vocab,_=build_mapping(processed["sources"]["ids"],dictionary_path=root/"ids/dictionary.jsonl")
    if mapping!=rebuilt or vocab!=rebuilt_vocab:raise ValueError("prepared IDS mapping/vocabulary invalid")
    rows=read_jsonl(root/"corpus.jsonl");selection=read_jsonl(root/"reports/selection.jsonl")
    if rows!=selection:raise ValueError("prepared selection mismatch")
    import csv
    with (root/"corpus.csv").open(encoding="utf-8",newline="") as h:csvrows=list(csv.DictReader(h))
    if len(rows)!=len(csvrows):raise ValueError("prepared row count mismatch")
    sources=set();sample_ids=[];texts=[]
    from ..data_contract import sample_id
    from ..pinyin import decompose,validate_canonical_label
    inventory=read_jsonl(inventory_path);acoustic_labels={entry["label"] for entry in inventory}
    for row,csvrow in zip(rows,csvrows):
        text=row["text"];labels=row["labels"];seq=row["ids_sequences"];lengths=row["component_lengths"]
        positions=row.get("positions")
        if type(row.get("sentence_length")) is not int or row["sentence_length"]!=len(text) or not isinstance(positions,list) or any(type(x) is not int for x in positions) or positions!=list(range(len(text))) or len(text)!=len(labels) or len(text)!=len(seq) or not isinstance(lengths,list) or any(type(n) is not int for n in lengths) or lengths!=[len(s) for s in seq] or any(not s for s in seq):raise ValueError("prepared alignment or length mismatch")
        if seq!=[mapping["characters"][c]["ids"] for c in text] or csvrow!={"text":text,"pinyin":" ".join(labels)}:raise ValueError("prepared encoding/CSV mismatch")
        if any(type(i) is not int or i<=0 or i>=len(vocab["tokens"]) for s in seq for i in s):raise ValueError("prepared token ID outside vocabulary")
        if not isinstance(row.get("source_ids"),list) or not row["source_ids"] or len(row["source_ids"])!=len(set(row["source_ids"])) or any(not isinstance(x,str) or not x or x in sources for x in row["source_ids"]):raise ValueError("prepared source IDs invalid")
        sources.update(row["source_ids"])
        if any(label not in acoustic_labels for label in labels):raise ValueError("prepared labels missing acoustic targets")
        for label in labels:decompose(validate_canonical_label(label))
        expected_id=sample_id(text,labels)
        if row.get("sample_id")!=expected_id:raise ValueError("prepared sample ID mismatch")
        sample_ids.append(expected_id);texts.append(text)
    if sample_ids!=sorted(sample_ids) or len(set(sample_ids))!=len(sample_ids) or len(set(texts))!=len(texts):raise ValueError("prepared sample order or uniqueness invalid")
    with (root/"acoustic/targets.pkl").open("rb") as h:targets=pickle.load(h)
    from ..data_contract import validate_acoustic
    validate_acoustic(targets,expected_keys=[x["label"] for x in read_jsonl(root/"reports/sound_inventory.jsonl")])
    stats=json.loads((root/"reports/run_statistics.json").read_text(encoding="utf8"))
    expected_outputs={"rows":len(rows),"ids_records":len(rows),"acoustic_targets":len(targets),"vocabulary_tokens":len(vocab["tokens"])}
    actual_outputs=stats.get("outputs")
    if (stats.get("schema_version")!=3 or stats.get("assembly_contract")!=ASSEMBLY_CONTRACT_VERSION or stats.get("status")!="success" or not isinstance(stats.get("run_id"),str) or not stats["run_id"] or stats.get("dataset_id")!=m.get("dataset_id") or stats.get("config_sha256")!=sha256_file(root/"config.json") or not isinstance(actual_outputs,dict) or set(actual_outputs)!=set(expected_outputs) or any(type(actual_outputs[k]) is not int or actual_outputs[k]!=v for k,v in expected_outputs.items())):raise ValueError("prepared statistics invalid")
    proc_stats=json.loads((root/"reports/processing_statistics.json").read_text(encoding="utf8"))
    if proc_stats.get("config_sha256")!=sha256_file(root/"reports/processed_corpus_config.json") or proc_stats.get("schema_version")!=2 or proc_stats.get("selected")!=len(rows) or not isinstance(proc_stats.get("run_id"),str) or not proc_stats.get("run_id"):raise ValueError("prepared processing statistics invalid")
    return m
