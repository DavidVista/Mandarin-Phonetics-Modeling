"""Build and validate immutable, model-independent IDS processed-corpus bundles."""
from __future__ import annotations
import csv,json,math,os,platform,shutil,tempfile,time,uuid
from collections import Counter
from pathlib import Path
from ..artifacts import read_jsonl,safe_artifact,sha256_file,write_json,write_jsonl
from ..data_contract import DATA_CONTRACT_VERSION,FILTERS,sample_id
from ..ids import IDS_RULES_VERSION,build_mapping,content_hash,validate_pin
from ..pinyin import decompose,validate_canonical_label
from ..pinyin import PINYIN_CONTRACT_VERSION
from .acoustic_bundle import load_acoustic_bundle
from .readers import read_corpus
from .corpus_processing import normalize_rows,filter_rows,exclude_conflicts,deduplicate_rows,sample_rows,coverage,character_counts
from .corpus_processing import CorpusConflictError

CORPUS_CONTRACT_VERSION="processed-corpus-2"
REQUIRED={"corpus.csv","config.json","ids/mapping.json","ids/vocabulary.json","ids/dictionary.jsonl","ids/COPYING","ids/LGPL","corpus.jsonl","reports/selection.jsonl","reports/rejections.jsonl","reports/coverage.json","reports/processing_statistics.json","compatibility/sound_inventory.jsonl"}

def _config(value):
    required={"corpus_id","output_root","corpus","ids","sampling","acoustic_bundle"}
    if not isinstance(value,dict) or set(value)!=required: raise ValueError("invalid processed corpus config keys")
    value=json.loads(json.dumps(value)); cid=value["corpus_id"]
    if not isinstance(cid,str) or not cid or Path(cid).name!=cid: raise ValueError("invalid corpus_id")
    probe={"dataset_id":cid,"output_root":value["output_root"],"corpus":value["corpus"],"ids":value["ids"],"sampling":value["sampling"],"acoustic_bundle":value["acoustic_bundle"]}
    from .config import validate_config
    validate_config(probe);return value

def prepare_corpus(config,*,conflict_policy="exclude"):
    if conflict_policy not in {"exclude","error"}:raise ValueError("unsupported conflict policy")
    config=_config(config);out=Path(config["output_root"]).resolve();out.mkdir(parents=True,exist_ok=True);target=out/config["corpus_id"]
    if target.exists():raise FileExistsError(target)
    temp=Path(tempfile.mkdtemp(prefix=f".{target.name}.",dir=out));run_id=str(uuid.uuid4());stages=[]
    try:
        write_json(temp/"config.json",{"schema_version":2,**config,"conflict_policy":conflict_policy}); config_hash=sha256_file(temp/"config.json")
        started=time.perf_counter(); rows,corpus_src=read_corpus(config["corpus"]); stages.append(_stage("load_corpus",len(rows),len(rows),0,time.perf_counter()-started,character_counts(rows)))
        if len({r["source_id"] for r in rows})!=len(rows): raise ValueError("corpus source IDs must be unique")
        started=time.perf_counter(); normalized,_=normalize_rows(rows,config["corpus"]["tone_convention"]); stages.append(_stage("normalize_identity",len(rows),len(normalized),0,time.perf_counter()-started,character_counts(normalized)))
        started=time.perf_counter(); ids_mapping,vocabulary,ids_provenance=build_mapping(config["ids"]); stages.append({"name":"validate_ids","unit":"entries","input_count":len(ids_mapping["characters"]),"output_count":len(ids_mapping["characters"]),"rejected":0,"status":"success","duration_seconds":time.perf_counter()-started})
        _vectors,inventory,acoustic_manifest=load_acoustic_bundle(config["acoustic_bundle"]["path"],config["acoustic_bundle"]["manifest_sha256"]); stages.append({"name":"sound_inventory","unit":"syllables","input_count":len(inventory),"output_count":len(inventory),"rejected":0,"status":"success","duration_seconds":0.0})
        acoustic={x["label"] for x in inventory}
        filtered,rejections=filter_rows(normalized,ids_mapping,vocabulary,acoustic,stages,FILTERS[:4])
        id_rows,rejected=filter_rows(filtered,ids_mapping,vocabulary,acoustic,stages,("ids",)); rejections.extend(rejected)
        id_stage=stages[-1]; id_stage["fallback_before"]=mode_counts(filtered,ids_mapping); id_stage["fallback_after"]=mode_counts(id_rows,ids_mapping)
        id_stage["missing_characters"]=missing_counts(filtered,ids_mapping); id_stage["excluded_source_ids"]=[x["source_id"] for x in rejected]
        if not id_rows: raise CorpusConflictError("no rows remain after IDS filtering",{"reasons":dict(sorted(Counter(r["reason"] for r in rejections).items())),"rejections":rejections,"missing_characters":missing_counts(filtered,ids_mapping)})
        id_hash=content_hash(ids_mapping); vocab_hash=content_hash(vocabulary)
        if conflict_policy=="exclude":
            conflict_free,conflict_rejections,conflict_counts=exclude_conflicts(id_rows);rejections.extend(conflict_rejections)
        else:
            conflict_free,conflict_rejections,conflict_counts=id_rows,[],{"input_count":len(id_rows),"output_count":len(id_rows),"rejected":0,"conflict_groups":0,"conflict_rows":0}
        stages.append(_stage("conflicts",len(id_rows),len(conflict_free),len(id_rows)-len(conflict_free),0.0,character_counts(id_rows),character_counts(conflict_free)))
        conflict_free,later=filter_rows(conflict_free,ids_mapping,vocabulary,acoustic,stages,("acoustic",));rejections.extend(later)
        if not conflict_free:
            raise CorpusConflictError("no rows remain after conflicting pronunciations were excluded" if conflict_counts.get("conflict_rows") else "no rows remain after corpus filtering",{"conflict_groups":conflict_counts.get("conflict_groups",0),"conflict_rows":conflict_counts.get("conflict_rows",0),"reasons":dict(sorted(Counter(r["reason"] for r in rejections).items())),"rejections":rejections})
        try:deduped,dc=deduplicate_rows(conflict_free,conflict_policy="error")
        except CorpusConflictError as exc:
            write_json(out/f"{config['corpus_id']}.failed.json",{"schema_version":2,"run_id":run_id,"status":"failure","exception":type(exc).__name__,"message":str(exc),"details":exc.diagnostics,"stages":stages});raise
        stages.append(_stage("deduplicate",dc["input_count"],dc["output_count"],dc["rejected"],0.0,character_counts(conflict_free),character_counts(deduped)))
        selected,sc=sample_rows(deduped,config["sampling"]); stages.append(_stage("sampling",sc["input_count"],sc["output_count"],sc["rejected"],0.0,character_counts(deduped),character_counts(selected)))
        (temp/"reports").mkdir();(temp/"ids").mkdir();(temp/"compatibility").mkdir()
        write_json(temp/"ids/mapping.json",ids_mapping);write_json(temp/"ids/vocabulary.json",vocabulary)
        id_hash=sha256_file(temp/"ids/mapping.json");vocab_hash=sha256_file(temp/"ids/vocabulary.json")
        shutil.copyfile(config["ids"]["path"],temp/"ids/dictionary.jsonl")
        for notice in ("COPYING","LGPL"):
            source=Path(config["ids"]["path"]).resolve().parent/notice
            if not source.is_file():raise FileNotFoundError(f"pinned IDS license notice missing: {source}")
            shutil.copyfile(source,temp/f"ids/{notice}")
        with (temp/"corpus.csv").open("w",encoding="utf-8",newline="") as h:
            w=csv.DictWriter(h,fieldnames=["text","pinyin"]);w.writeheader();w.writerows({"text":r["text"],"pinyin":" ".join(r["labels"])} for r in selected)
        write_jsonl(temp/"corpus.jsonl",(record(r) for r in selected))
        write_jsonl(temp/"reports/selection.jsonl",(record(r) for r in selected));write_jsonl(temp/"reports/rejections.jsonl",rejections)
        write_json(temp/"reports/coverage.json",{"before_sampling":coverage(deduped),"after_sampling":coverage(selected)})
        shutil.copyfile(Path(config["acoustic_bundle"]["path"]).resolve()/"sound_inventory.jsonl",temp/"compatibility/sound_inventory.jsonl")
        stats={"schema_version":2,"run_id":run_id,"status":"success","corpus_id":config["corpus_id"],"config_sha256":config_hash,"stages":stages,
               "rejections":dict(sorted(Counter(r["reason"] for r in rejections).items())),"eligible":len(deduped),"selected":len(selected),"ids":{"mapping_sha256":id_hash,"vocabulary_sha256":vocab_hash,"fallback_before":mode_counts(filtered,ids_mapping),"fallback_after":mode_counts(id_rows,ids_mapping),"missing_characters":missing_counts(filtered,ids_mapping)}}
        write_json(temp/"reports/processing_statistics.json",stats)
        artifacts={str(p.relative_to(temp)).replace("\\","/"):sha256_file(p) for p in temp.rglob("*") if p.is_file()}
        notices={name:sha256_file(temp/f"ids/{name}") for name in ("COPYING","LGPL")}
        write_json(temp/"manifest.json",{"bundle_type":"processed_corpus","schema_version":2,"corpus_contract":CORPUS_CONTRACT_VERSION,"data_contract":DATA_CONTRACT_VERSION,"pinyin_contract":PINYIN_CONTRACT_VERSION,"ids_rules":IDS_RULES_VERSION,"corpus_id":config["corpus_id"],"artifacts":artifacts,"sources":{"corpus":corpus_src,"ids":ids_provenance,"acoustic_bundle_manifest_sha256":config["acoustic_bundle"]["manifest_sha256"],"sound_inventory_sha256":sha256_file(temp/"compatibility/sound_inventory.jsonl")},"ids":{"mapping_sha256":sha256_file(temp/"ids/mapping.json"),"vocabulary_sha256":sha256_file(temp/"ids/vocabulary.json"),"license_notices_sha256":notices},"environment":{"python":platform.python_version()}})
        validate_corpus_bundle(temp);os.rename(temp,target);return target
    except Exception as exc:
        write_json(out/f"{config['corpus_id']}.failed.json",{"schema_version":2,"run_id":run_id,"status":"failure","exception":type(exc).__name__,"message":str(exc),"details":getattr(exc,"diagnostics",{}),"stages":stages})
        shutil.rmtree(temp,ignore_errors=True);raise

def _stage(name,ins,outs,rejected,duration,input_chars,output_chars=None):return {"name":name,"unit":"rows","input_count":ins,"output_count":outs,"rejected":rejected,"status":"success","duration_seconds":duration,"input_characters":input_chars,"output_characters":output_chars if output_chars is not None else input_chars}
def record(row):return {**{k:row[k] for k in ("sample_id","text","labels","source_ids","ids_sequences","component_lengths")},"sentence_length":len(row["text"]),"positions":list(range(len(row["text"])))}
def mode_counts(rows,mapping):
    counts=Counter(ch for row in rows for ch in row["text"] if ch in mapping["characters"])
    by_mode={mode:{"types":0,"occurrences":0} for mode in ("decomposition","character_fallback")}
    for ch,count in counts.items():
        target=by_mode[mapping["characters"][ch]["mode"]];target["types"]+=1;target["occurrences"]+=count
    return {"types":sum(x["types"] for x in by_mode.values()),"occurrences":sum(x["occurrences"] for x in by_mode.values()),"by_mode":by_mode}
def missing_counts(rows,mapping):
    counts=Counter(ch for row in rows for ch in row["text"] if ch not in mapping["characters"])
    return {"characters":dict(sorted(counts.items(),key=lambda x:ord(x[0]))),"occurrences":sum(counts.values())}

def validate_corpus_bundle(path,expected_manifest_sha256=None):
    root=Path(path).resolve();mp=root/"manifest.json"
    if expected_manifest_sha256 and sha256_file(mp)!=expected_manifest_sha256:raise ValueError("processed corpus manifest checksum mismatch")
    m=json.loads(mp.read_text(encoding="utf-8"))
    if m.get("bundle_type")!="processed_corpus" or m.get("schema_version")!=2 or m.get("corpus_contract")!=CORPUS_CONTRACT_VERSION or m.get("ids_rules")!=IDS_RULES_VERSION:raise ValueError("unsupported processed corpus bundle")
    arts=m.get("artifacts",{})
    if set(arts)!=REQUIRED:raise ValueError("processed corpus artifact set invalid")
    actual={str(p.relative_to(root)).replace("\\","/") for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"}
    if actual!=set(arts):raise ValueError("processed corpus contains untracked artifacts")
    for rel,digest in arts.items():
        file=safe_artifact(root,rel)
        if not file.is_file() or sha256_file(file)!=digest:raise ValueError(f"processed corpus artifact mismatch: {rel}")
    cfg=json.loads((root/"config.json").read_text(encoding="utf-8"));stats=json.loads((root/"reports/processing_statistics.json").read_text(encoding="utf-8"))
    if cfg.get("schema_version")!=2 or cfg.get("conflict_policy") not in {"exclude","error"} or cfg.get("corpus_id")!=m.get("corpus_id") or stats.get("config_sha256")!=sha256_file(root/"config.json"):raise ValueError("processed corpus identity mismatch")
    mapping=json.loads((root/"ids/mapping.json").read_text(encoding="utf-8"));vocab=json.loads((root/"ids/vocabulary.json").read_text(encoding="utf-8"))
    rebuilt,rebuilt_vocab,provenance=build_mapping(cfg["ids"],dictionary_path=root/"ids/dictionary.jsonl")
    if mapping!=rebuilt or vocab!=rebuilt_vocab:raise ValueError("processed corpus IDS mapping or vocabulary mismatch")
    if m["sources"].get("ids")!=provenance or m["ids"].get("mapping_sha256")!=sha256_file(root/"ids/mapping.json") or m["ids"].get("vocabulary_sha256")!=sha256_file(root/"ids/vocabulary.json"):raise ValueError("processed corpus IDS provenance mismatch")
    if m["ids"].get("license_notices_sha256")!={name:sha256_file(root/f"ids/{name}") for name in ("COPYING","LGPL")}:raise ValueError("processed corpus IDS license notice mismatch")
    rows=read_jsonl(root/"corpus.jsonl");selection=read_jsonl(root/"reports/selection.jsonl")
    if rows!=selection:raise ValueError("processed corpus record/selection mismatch")
    with (root/"corpus.csv").open(encoding="utf-8",newline="") as h:csv_rows=list(csv.DictReader(h))
    if len(rows)!=len(csv_rows):raise ValueError("processed corpus CSV record count mismatch")
    seen_ids=set(); seen_sources=set()
    for row,csv_row in zip(rows,csv_rows):
        text=row.get("text"); labels=row.get("labels");seq=row.get("ids_sequences");lengths=row.get("component_lengths")
        positions=row.get("positions")
        if (not isinstance(text,str) or not text or type(row.get("sentence_length")) is not int or row["sentence_length"]!=len(text) or not isinstance(positions,list) or any(type(pos) is not int for pos in positions) or positions!=list(range(len(text))) or not isinstance(labels,list) or len(text)!=len(labels) or len(seq)!=len(text) or not isinstance(lengths,list) or any(type(n) is not int for n in lengths) or lengths!=[len(x) for x in seq] or any(not x for x in seq)):
            raise ValueError("invalid unpadded IDS record")
        expected=[]
        for ch in text:
            if ch not in mapping["characters"]:raise ValueError("prepared text character missing from IDS mapping")
            expected.append(mapping["characters"][ch]["ids"])
        if seq!=expected or csv_row!={"text":text,"pinyin":" ".join(labels)}:raise ValueError("IDS sequence or CSV alignment mismatch")
        if any(type(i) is not int or i<=0 or i>=len(vocab["tokens"]) for seqrow in seq for i in seqrow):raise ValueError("IDS token ID outside fixed vocabulary")
        for label in labels:decompose(validate_canonical_label(label))
        sid=sample_id(text,labels)
        if row.get("sample_id")!=sid or sid in seen_ids:raise ValueError("sample ID invalid or duplicated")
        seen_ids.add(sid)
        source_ids=row.get("source_ids")
        if not isinstance(source_ids,list) or not source_ids or len(source_ids)!=len(set(source_ids)) or any(not isinstance(x,str) or not x or x in seen_sources for x in source_ids):raise ValueError("source IDs invalid or duplicated")
        seen_sources.update(source_ids)
    if (stats.get("schema_version")!=2 or not isinstance(stats.get("run_id"),str) or not stats.get("run_id") or stats.get("status")!="success" or stats.get("selected")!=len(rows) or stats.get("config_sha256")!=sha256_file(root/"config.json")):
        raise ValueError("processed corpus statistics invalid")
    if stats.get("rejections")!=dict(sorted(Counter(item.get("reason") for item in read_jsonl(root/"reports/rejections.jsonl")).items())):raise ValueError("processed corpus rejection statistics invalid")
    if stats.get("ids",{}).get("mapping_sha256")!=sha256_file(root/"ids/mapping.json") or stats.get("ids",{}).get("vocabulary_sha256")!=sha256_file(root/"ids/vocabulary.json"):raise ValueError("processed corpus IDS statistics invalid")
    stages=stats.get("stages")
    expected_order=["load_corpus","normalize_identity","validate_ids","sound_inventory",*FILTERS[:4],"ids","conflicts","acoustic","deduplicate","sampling"]
    if not isinstance(stages,list) or [stage.get("name") for stage in stages]!=expected_order:raise ValueError("processed corpus stage order invalid")
    for stage in stages:
        if stage.get("status")!="success":raise ValueError("processed corpus stage status invalid")
        for key in ("input_count","output_count","rejected"):
            if type(stage.get(key)) is not int or stage[key]<0:raise ValueError("processed corpus stage count invalid")
        if stage["input_count"]!=stage["output_count"]+stage["rejected"]:raise ValueError("processed corpus stage balance invalid")
        duration=stage.get("duration_seconds")
        if not isinstance(duration,(int,float)) or isinstance(duration,bool) or not math.isfinite(duration) or duration<0:raise ValueError("processed corpus stage duration invalid")
    if stages[-1]["output_count"]!=len(rows) or stages[-1]["input_count"]!=stats.get("eligible"):raise ValueError("processed corpus sampling counts invalid")
    inventory=read_jsonl(root/"compatibility/sound_inventory.jsonl")
    if stages[3]["output_count"]!=len(inventory):raise ValueError("processed corpus sound inventory counts invalid")
    labels={entry.get("label") for entry in inventory}
    if any(label not in labels for row in rows for label in row["labels"]):raise ValueError("processed corpus label missing from sound inventory")
    row_stages=[stage for stage in stages if stage["unit"]=="rows"]
    if any(left["output_count"]!=right["input_count"] for left,right in zip(row_stages,row_stages[1:])):raise ValueError("processed corpus stage adjacency invalid")
    sampling=cfg["sampling"]
    if (type(stats.get("eligible")) is not int or type(stats.get("selected")) is not int or stats["eligible"]!=stages[-2]["output_count"] or
            (sampling["size"] is not None and stats["selected"]!=sampling["size"]) or (sampling["size"] is None and stats["selected"]!=stats["eligible"])):
        raise ValueError("processed corpus sampling configuration mismatch")
    id_stage=next(stage for stage in stages if stage["name"]=="ids")
    for name in ("fallback_before","fallback_after"):
        if stats["ids"].get(name)!=id_stage.get(name):raise ValueError("processed corpus IDS exposure statistics mismatch")
    reasons=stats["rejections"]
    for name in FILTERS:
        if next(stage for stage in stages if stage["name"]==name)["rejected"]!=reasons.get(name,0):raise ValueError("processed corpus filter rejection count mismatch")
    cov=json.loads((root/"reports/coverage.json").read_text(encoding="utf-8"))
    selected_rows=[{"text":row["text"],"labels":row["labels"]} for row in rows]
    expected_coverage=json.loads(json.dumps(coverage(selected_rows)))
    if cov.get("after_sampling")!=expected_coverage:raise ValueError("processed corpus coverage mismatch")
    return m
