"""Build and validate immutable, model-independent processed-corpus bundles."""

from __future__ import annotations
import csv,json,math,os,platform,shutil,tempfile,time,uuid
from collections import Counter
from pathlib import Path
from ..artifacts import read_jsonl,safe_artifact,sha256_file,write_json,write_jsonl
from ..data_contract import DATA_CONTRACT_VERSION,FILTERS
from ..pinyin import PARSER_PROVENANCE, PINYIN_CONTRACT_VERSION
from .acoustic_bundle import load_acoustic_bundle
from .readers import read_corpus,read_wubi
from .corpus_processing import CorpusConflictError,normalize_rows,filter_rows,exclude_conflicts,deduplicate_rows,sample_rows,coverage

CORPUS_CONTRACT_VERSION="processed-corpus-1"
REQUIRED={"corpus.csv","config.json","reports/selection.jsonl","reports/rejections.jsonl","reports/coverage.json","reports/processing_statistics.json",
          "compatibility/sound_inventory.jsonl","compatibility/wubi.pkl"}

def _config(value):
    required={"corpus_id","output_root","corpus","wubi","sampling","acoustic_bundle"}
    if not isinstance(value,dict) or set(value)!=required:raise ValueError("invalid processed corpus config keys")
    value=json.loads(json.dumps(value));cid=value["corpus_id"]
    if not isinstance(cid,str) or not cid or Path(cid).name!=cid:raise ValueError("invalid corpus_id")
    from .config import validate_config
    probe={"dataset_id":cid,"output_root":value["output_root"],"corpus":value["corpus"],"wubi":value["wubi"],"sampling":value["sampling"],"acoustic_bundle":value["acoustic_bundle"]}
    validate_config(probe);return value

def prepare_corpus(config):
    config=_config(config);out=Path(config["output_root"]).resolve();out.mkdir(parents=True,exist_ok=True);target=out/config["corpus_id"]
    if target.exists():raise FileExistsError(target)
    temp=Path(tempfile.mkdtemp(prefix=f".{target.name}.",dir=out));stages=[];run_id=str(uuid.uuid4())
    try:
        write_json(temp/"config.json",{"schema_version":1,**config});config_sha256=sha256_file(temp/"config.json")
        started=time.perf_counter();rows,corpus_src=read_corpus(config["corpus"]);stages.append(_stage("load_corpus",len(rows),len(rows),0,time.perf_counter()-started))
        source_ids=[r["source_id"] for r in rows]
        if len(source_ids)!=len(set(source_ids)):raise ValueError("corpus source IDs must be unique")
        started=time.perf_counter();normalized,counts=normalize_rows(rows,config["corpus"]["tone_convention"]);stages.append(_stage("normalize_identity",**_counts(counts),duration=time.perf_counter()-started))
        started=time.perf_counter();_vectors,inventory,acoustic_manifest=load_acoustic_bundle(config["acoustic_bundle"]["path"],config["acoustic_bundle"]["manifest_sha256"]);stages.append({**_stage("sound_inventory",len(inventory),len(inventory),0,time.perf_counter()-started),"unit":"syllables"})
        started=time.perf_counter();wubi,wubi_src=read_wubi(config["wubi"]);stages.append({**_stage("validate_wubi",len(wubi),len(wubi),0,time.perf_counter()-started),"unit":"entries"})
        acoustic={x["label"] for x in inventory}
        filtered,rejections=filter_rows(normalized,wubi,acoustic,stages,FILTERS[:4])
        started=time.perf_counter();conflict_free,conflict_rejections,counts=exclude_conflicts(filtered)
        stages.append(_stage("conflicts",**_counts(counts),duration=time.perf_counter()-started));stages[-1].update({k:counts[k] for k in ("conflict_groups","conflict_rows")})
        rejections.extend(conflict_rejections)
        conflict_free,later_rejections=filter_rows(conflict_free,wubi,acoustic,stages,FILTERS[4:]);rejections.extend(later_rejections)
        reasons=dict(sorted(Counter(r["reason"] for r in rejections).items()))
        if not conflict_free:
            message="no rows remain after conflicting pronunciations were excluded" if counts["conflict_rows"] else "no rows remain after corpus filtering"
            stages.append({"name":"validate_nonempty","unit":"rows","input_count":0,"output_count":0,"rejected":0,
                           "status":"failure","counts_complete":True,"duration_seconds":0.0})
            raise CorpusConflictError(message,{"conflict_groups":counts["conflict_groups"],"conflict_rows":counts["conflict_rows"],
                "reasons":reasons,"rejections":rejections})
        stages.append(_stage("validate_nonempty",len(conflict_free),len(conflict_free),0))
        started=time.perf_counter()
        try:deduped,counts=deduplicate_rows(conflict_free)
        except Exception as exc:
            detail=getattr(exc,"diagnostics",{});stages.append({"name":"deduplicate","unit":"rows","status":"failure","duration_seconds":time.perf_counter()-started,**detail});raise
        stages.append(_stage("deduplicate",**_counts(counts),duration=time.perf_counter()-started))
        started=time.perf_counter()
        try:selected,counts=sample_rows(deduped,config["sampling"])
        except Exception:
            stages.append({"name":"sampling","unit":"rows","input_count":len(deduped),"output_count":0,"rejected":0,
                           "blocked_count":len(deduped),"status":"failure","duration_seconds":time.perf_counter()-started});raise
        stages.append(_stage("sampling",**_counts(counts),duration=time.perf_counter()-started))
        (temp/"reports").mkdir();(temp/"compatibility").mkdir()
        with (temp/"corpus.csv").open("w",encoding="utf-8",newline="") as h:
            w=csv.DictWriter(h,fieldnames=["text","pinyin"]);w.writeheader();w.writerows({"text":r["text"],"pinyin":" ".join(r["labels"])} for r in selected)
        write_jsonl(temp/"reports/selection.jsonl",({k:r[k] for k in ("sample_id","text","labels","source_ids")} for r in selected))
        write_jsonl(temp/"reports/rejections.jsonl",rejections);write_json(temp/"reports/coverage.json",{"before_sampling":coverage(deduped),"after_sampling":coverage(selected)})
        acoustic_root=Path(config["acoustic_bundle"]["path"]).resolve()
        shutil.copyfile(acoustic_root/"sound_inventory.jsonl",temp/"compatibility/sound_inventory.jsonl")
        shutil.copyfile(Path(config["wubi"]["path"]),temp/"compatibility/wubi.pkl")
        stats={"schema_version":1,"run_id":run_id,"status":"success","corpus_id":config["corpus_id"],"config_sha256":config_sha256,"stages":stages,
               "rejections":dict(sorted(Counter(r["reason"] for r in rejections).items())),"eligible":len(deduped),"selected":len(selected)}
        write_json(temp/"reports/processing_statistics.json",stats)
        invhash=acoustic_manifest["artifacts"]["sound_inventory.jsonl"]
        artifacts={str(p.relative_to(temp)).replace("\\","/"):sha256_file(p) for p in temp.rglob("*") if p.is_file()}
        write_json(temp/"manifest.json",{"bundle_type":"processed_corpus","schema_version":1,"corpus_contract":CORPUS_CONTRACT_VERSION,
            "data_contract":DATA_CONTRACT_VERSION,"pinyin_contract":PINYIN_CONTRACT_VERSION,"parser_provenance":PARSER_PROVENANCE,"corpus_id":config["corpus_id"],"artifacts":artifacts,
            "sources":{"corpus":corpus_src,"wubi":wubi_src,"acoustic_bundle_manifest_sha256":config["acoustic_bundle"]["manifest_sha256"],
                       "sound_inventory_sha256":invhash},"environment":{"python":platform.python_version()}})
        validate_corpus_bundle(temp);os.rename(temp,target);return target
    except Exception as exc:
        config_sha256=sha256_file(temp/"config.json") if (temp/"config.json").is_file() else None
        order=["load_corpus","normalize_identity","sound_inventory","validate_wubi",*FILTERS[:4],"conflicts",*FILTERS[4:],"validate_nonempty","deduplicate","sampling"]
        recorded={row["name"] for row in stages}
        if not any(row.get("status")=="failure" for row in stages):
            failed=next((name for name in order if name not in recorded),order[-1]);units={"sound_inventory":"syllables","validate_wubi":"entries"}
            stages.append({"name":failed,"unit":units.get(failed,"rows"),"input_count":None,"output_count":None,"rejected":None,
                           "status":"failure","counts_complete":False,"counts_reason":"operation aborted before complete counts were available","duration_seconds":0.0});recorded.add(failed)
        units={"sound_inventory":"syllables","validate_wubi":"entries"}
        stages.extend({"name":name,"unit":units.get(name,"rows"),"input_count":None,"output_count":None,"rejected":None,
                       "status":"not-run","counts_complete":False,"counts_reason":"blocked by earlier failure","duration_seconds":0.0}
                      for name in order if name not in {row["name"] for row in stages})
        stages.sort(key=lambda row:order.index(row["name"]))
        write_json(out/f"{config['corpus_id']}.failed.json",{"schema_version":1,"run_id":run_id,"status":"failure","config_sha256":config_sha256,
            "exception":type(exc).__name__,"message":str(exc),"details":getattr(exc,"diagnostics",{}),"stages":stages})
        shutil.rmtree(temp,ignore_errors=True);raise

def _counts(c):return {"input_count":c["input_count"],"output_count":c["output_count"],"rejected":c["rejected"]}
def _stage(name,input_count,output_count,rejected,duration=0.0):return {"name":name,"unit":"rows","input_count":input_count,"output_count":output_count,"rejected":rejected,"status":"success","duration_seconds":duration}

def validate_corpus_bundle(path,expected_manifest_sha256=None):
    root=Path(path).resolve();mp=root/"manifest.json"
    if expected_manifest_sha256 and sha256_file(mp)!=expected_manifest_sha256:raise ValueError("processed corpus manifest checksum mismatch")
    m=json.loads(mp.read_text(encoding="utf8"))
    if (m.get("bundle_type")!="processed_corpus" or m.get("schema_version")!=1 or m.get("corpus_contract")!=CORPUS_CONTRACT_VERSION
            or m.get("data_contract")!=DATA_CONTRACT_VERSION or m.get("pinyin_contract") != PINYIN_CONTRACT_VERSION or m.get("parser_provenance") != PARSER_PROVENANCE):raise ValueError("unsupported processed corpus bundle")
    if not REQUIRED.issubset(m.get("artifacts",{})):raise ValueError("processed corpus artifacts missing")
    for rel,digest in m["artifacts"].items():
        p=safe_artifact(root,rel)
        if not p.is_file() or sha256_file(p)!=digest:raise ValueError(f"processed corpus artifact mismatch: {rel}")
    config=json.loads((root/"config.json").read_text(encoding="utf8"));stats=json.loads((root/"reports/processing_statistics.json").read_text(encoding="utf8"))
    if config.get("schema_version")!=1 or config.get("corpus_id")!=m.get("corpus_id"):raise ValueError("processed corpus config identity mismatch")
    _config({k:v for k,v in config.items() if k!="schema_version"})
    if (stats.get("schema_version")!=1 or stats.get("status")!="success" or stats.get("corpus_id")!=m.get("corpus_id")
            or not isinstance(stats.get("run_id"),str) or not stats["run_id"]
            or stats.get("config_sha256")!=sha256_file(root/"config.json")):raise ValueError("processed corpus statistics identity mismatch")
    with (root/"corpus.csv").open(encoding="utf8",newline="") as h:
        reader=csv.DictReader(h)
        if reader.fieldnames!=["text","pinyin"]:raise ValueError("processed corpus CSV headers invalid")
        rows=list(reader)
    selection=read_jsonl(root/"reports/selection.jsonl")
    if len(rows)!=len(selection) or any(r["text"]!=s.get("text") or r["pinyin"].split()!=s.get("labels") for r,s in zip(rows,selection)):raise ValueError("processed corpus selection mismatch")
    from ..data_contract import sample_id
    from ..pinyin import decompose, validate_canonical_label
    ids=[];texts=[]
    for row,selected in zip(rows,selection):
        labels=row["pinyin"].split()
        if not row["text"] or len(row["text"])!=len(labels) or not isinstance(selected.get("source_ids"),list) or not selected["source_ids"]:raise ValueError("processed corpus row invalid")
        for label in labels:decompose(validate_canonical_label(label))
        expected=sample_id(row["text"],labels)
        if selected.get("sample_id")!=expected:raise ValueError("processed corpus sample ID mismatch")
        ids.append(expected);texts.append(row["text"])
    if ids!=sorted(ids) or len(ids)!=len(set(ids)) or len(texts)!=len(set(texts)):raise ValueError("processed corpus IDs/order/uniqueness invalid")
    all_sources=[source for row in selection for source in row["source_ids"]]
    if len(all_sources)!=len(set(all_sources)):raise ValueError("processed corpus source IDs are not unique")
    order=["load_corpus","normalize_identity","sound_inventory","validate_wubi",*FILTERS[:4],"conflicts",*FILTERS[4:],"validate_nonempty","deduplicate","sampling"]
    stages=stats.get("stages")
    if not isinstance(stages,list) or [x.get("name") for x in stages]!=order:raise ValueError("processed corpus stage order invalid")
    by={x["name"]:x for x in stages}
    for name,row in by.items():
        units={"sound_inventory":"syllables","validate_wubi":"entries"}
        if row.get("status")!="success" or row.get("unit")!=units.get(name,"rows"):raise ValueError("processed corpus stage status invalid")
        if any(not isinstance(row.get(k),int) or isinstance(row.get(k),bool) or row[k]<0 for k in ("input_count","output_count","rejected")):raise ValueError("processed corpus stage counts invalid")
        duration=row.get("duration_seconds")
        if not isinstance(duration,(int,float)) or isinstance(duration,bool) or not math.isfinite(duration) or duration<0:raise ValueError("processed corpus duration invalid")
        if name in {"load_corpus","normalize_identity","sound_inventory","validate_wubi","validate_nonempty"}:
            if row["input_count"]!=row["output_count"] or row["rejected"]!=0:raise ValueError("processed corpus lossless stage balance invalid")
        elif row["input_count"]!=row["output_count"]+row["rejected"]:raise ValueError("processed corpus stage balance invalid")
    reducing=[*FILTERS[:4],"conflicts",*FILTERS[4:],"validate_nonempty","deduplicate","sampling"]
    adjacency=[("load_corpus","normalize_identity"),("normalize_identity",reducing[0]),*zip(reducing,reducing[1:])]
    for left,right in adjacency:
        if by[left]["output_count"]!=by[right]["input_count"]:raise ValueError("processed corpus stage adjacency invalid")
    sampling=config["sampling"]
    if stats.get("eligible")!=by["deduplicate"]["output_count"] or stats.get("selected")!=len(rows) or by["sampling"]["output_count"]!=len(rows):raise ValueError("processed corpus selection counts invalid")
    if sampling["size"] is None and len(rows)!=stats["eligible"]:raise ValueError("processed corpus full sampling invalid")
    if sampling["size"] is not None and len(rows)!=sampling["size"]:raise ValueError("processed corpus sampling size invalid")
    rejections=read_jsonl(root/"reports/rejections.jsonl")
    if stats.get("rejections")!=dict(sorted(Counter(x.get("reason") for x in rejections).items())):raise ValueError("processed corpus rejection counts invalid")
    if by["conflicts"].get("rejected")!=sum(x.get("reason")=="conflicting_pronunciation" for x in rejections):raise ValueError("processed corpus conflict counts invalid")
    cov=json.loads((root/"reports/coverage.json").read_text(encoding="utf8"))
    selected_rows=[{"text":r["text"],"labels":r["pinyin"].split()} for r in rows]
    expected_coverage=json.loads(json.dumps(coverage(selected_rows)))
    if cov.get("after_sampling")!=expected_coverage:raise ValueError("processed corpus coverage invalid")
    inventory=read_jsonl(root/"compatibility/sound_inventory.jsonl")
    if (sha256_file(root/"compatibility/sound_inventory.jsonl")!=m["sources"]["sound_inventory_sha256"]
            or by["sound_inventory"]["output_count"]!=len(inventory)):raise ValueError("processed corpus inventory counts invalid")
    import pickle
    from ..data_contract import validate_wubi
    with (root/"compatibility/wubi.pkl").open("rb") as h:
        wubi=validate_wubi(pickle.load(h),alphabet=config["wubi"].get("alphabet","abcdefghijklmnopqrstuvwxyz0123456789"))
    if (sha256_file(root/"compatibility/wubi.pkl")!=m["sources"]["wubi"]["sha256"]
            or by["validate_wubi"]["output_count"]!=len(wubi)):raise ValueError("processed corpus Wubi counts invalid")
    return m
