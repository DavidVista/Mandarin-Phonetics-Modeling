"""Strict Wav2Vec2 graduality bundle loading and population contracts."""
from __future__ import annotations
import hashlib,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd

EMBEDDING_FILES=("config.json","sound_inventory.jsonl","pinyin_embeddings.pkl","reports/inference.json")
EVALUATION_FILES=("config.json","coverage.json","sentences.jsonl","occurrences.jsonl","groups.jsonl","pairs/contextual.jsonl","pairs/character_pinyin.jsonl","pairs/syllable.jsonl")
EXTRACTION={"layer":-1,"pooling":"mean_all","padding":"none","sample_rate":16000,"mono":True,"dtype":"float32"}
def sha256_file(path,chunk=1024*1024):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for data in iter(lambda:f.read(chunk),b""):h.update(data)
 return h.hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def read_jsonl(path):
 rows=[]
 with Path(path).open(encoding="utf-8") as f:
  for number,line in enumerate(f,1):
   if line.strip():
    try:rows.append(json.loads(line))
    except json.JSONDecodeError as exc:raise ValueError(f"Invalid JSONL {path}:{number}: {exc}") from exc
 return rows
def contained(root,relative):
 root=Path(root).resolve();path=(root/relative).resolve()
 if path!=root and root not in path.parents:raise ValueError(f"Artifact escapes bundle root: {relative!r}")
 return path
def verified_artifacts(root,manifest):
 artifacts=manifest.get("artifacts")
 if not isinstance(artifacts,dict) or not artifacts:raise ValueError(f"Missing artifacts map in {root}")
 verified={}
 for relative,want in sorted(artifacts.items()):
  path=contained(root,relative)
  if not path.is_file():raise FileNotFoundError(path)
  got=sha256_file(path)
  if got!=want:raise ValueError(f"SHA-256 mismatch for {path}: {got} != {want}")
  verified[relative]=got
 return verified
def comparable_model(value):
 if isinstance(value,str):return value
 if isinstance(value,dict):return {k:comparable_model(v) for k,v in value.items() if k!="backend"}
 if isinstance(value,list):return [comparable_model(v) for v in value]
 return value
def model_identifier(value):
 if isinstance(value,str):return value
 if isinstance(value,dict):
  for key in ("id","model_id","name","pretrained_model_name_or_path"):
   if isinstance(value.get(key),str):return value[key]
 return None
def load_embedding_bundle(name,root,expected_manifest_sha256,expected_model,expected_extraction=EXTRACTION):
 root=Path(root).resolve();mp=root/"manifest.json";actual=sha256_file(mp)
 if actual!=expected_manifest_sha256:raise ValueError(f"Unexpected {name} manifest SHA-256: {actual}")
 manifest=read_json(mp)
 if manifest.get("bundle_type")!="acoustic_embeddings" or manifest.get("schema_version")!=1:raise ValueError(f"Unsupported embedding manifest contract for {name}")
 verified=verified_artifacts(root,manifest)
 for relative in EMBEDDING_FILES:
  if relative not in verified or not contained(root,relative).is_file():raise ValueError(f"Required verified embedding artifact is missing: {relative}")
 config=read_json(root/"config.json");inference=read_json(root/"reports/inference.json");inventory=read_jsonl(root/"sound_inventory.jsonl");labels=[x.get("label") for x in inventory]
 if any(not isinstance(x,str) or not x for x in labels) or len(labels)!=len(set(labels)):raise ValueError(f"Invalid or duplicate inventory labels for {name}")
 inventory_hash=sha256_file(root/"sound_inventory.jsonl")
 if manifest.get("inventory_sha256")!=inventory_hash:raise ValueError(f"inventory_sha256 mismatch for {name}")
 if inference.get("errors",0)!=0:raise ValueError(f"Inference report contains errors for {name}")
 mm=manifest.get("model");cm=config.get("model")
 if comparable_model(mm)!=comparable_model(cm):raise ValueError(f"Manifest/config model mismatch for {name}")
 if model_identifier(mm)!=expected_model:raise ValueError(f"Unexpected model id for {name}")
 if not isinstance(mm,dict) or not isinstance(cm,dict):raise ValueError(f"Model configuration must be an object for {name}")
 for field,want in expected_extraction.items():
  if cm.get(field,mm.get(field))!=want:raise ValueError(f"Unexpected extraction setting {name}/{field}")
 with (root/"pinyin_embeddings.pkl").open("rb") as f:raw=pickle.load(f)
 if not isinstance(raw,dict) or set(raw)!=set(labels):raise ValueError(f"Embedding labels do not exactly match inventory for {name}")
 vectors={};dims=set()
 for label in labels:
  v=np.asarray(raw[label])
  if v.ndim!=1 or not np.issubdtype(v.dtype,np.number):raise ValueError(f"Vector for {name}/{label} is not numeric 1-D")
  v=v.astype(np.float64,copy=False)
  if not np.isfinite(v).all() or np.linalg.norm(v)==0:raise ValueError(f"Vector for {name}/{label} is non-finite or zero")
  vectors[label]=v;dims.add(v.size)
 if len(dims)!=1:raise ValueError(f"Inconsistent dimensions for {name}")
 dimension=dims.pop()
 if manifest.get("count")!=len(vectors) or manifest.get("dimension")!=dimension:raise ValueError(f"Manifest count/dimension mismatch for {name}")
 return {"name":name,"root":root,"manifest":manifest,"config":config,"inference":inference,"inventory":inventory,"vectors":vectors,"manifest_sha256":actual,"artifact_sha256":verified,"inventory_sha256":inventory_hash,"dimension":dimension}
def load_evaluation_bundle(root,expected_manifest_sha256,pairing_policy="nonzero_nonzero"):
 root=Path(root).resolve();mp=root/"manifest.json";actual=sha256_file(mp)
 if actual!=expected_manifest_sha256:raise ValueError(f"Unexpected evaluation manifest SHA-256: {actual}")
 manifest=read_json(mp)
 if manifest.get("schema_version")!=2 or manifest.get("pairing_policy")!=pairing_policy:raise ValueError("Unsupported evaluation contract")
 verified=verified_artifacts(root,manifest)
 for relative in EVALUATION_FILES:
  if relative not in verified or not contained(root,relative).is_file():raise ValueError(f"Required verified evaluation artifact is missing: {relative}")
 config=read_json(root/"config.json");coverage=read_json(root/"coverage.json")
 for field in ("schema_version","evaluation_id","pairing_policy"):
  if config.get(field)!=manifest.get(field):raise ValueError(f"Evaluation config/manifest mismatch: {field}")
 return {"root":root,"manifest":manifest,"config":config,"coverage":coverage,"groups":read_jsonl(root/"groups.jsonl"),"syllable_pairs":read_jsonl(root/"pairs/syllable.jsonl"),"manifest_sha256":actual,"artifact_sha256":verified}
def normalized_component(value,field,label,allow_empty=False,coerce_string=False):
 if value is None:raise ValueError(f"Missing {field} for {label}")
 if not isinstance(value,str):
  if coerce_string:value=str(value)
  else:raise ValueError(f"{field} must be a string for {label}")
 value=value.strip()
 if not value and not allow_empty:raise ValueError(f"Empty {field} for {label}")
 return value
def validate_registered_pairs(rows,eligible):
 required=("pair_id","left_id","right_id","left_pinyin","right_pinyin","equal_initial","equal_final","equal_tone","overlap_mask","left_zero_initial","right_zero_initial","no_zero_initial_diagnostic");metadata={x["group_id"]:x for x in eligible};seen_ids=set();seen=set()
 for number,row in enumerate(rows,1):
  if not isinstance(row,dict):raise ValueError(f"Syllable pair row {number} must be an object")
  missing=[x for x in required if x not in row]
  if missing:raise ValueError(f"Syllable pair row {number} is missing required fields: {missing}")
  left,right=row["left_id"],row["right_id"]
  if not isinstance(left,str) or not isinstance(right,str) or left not in metadata or right not in metadata or left>=right or (left,right) in seen:raise ValueError(f"Invalid stored pair at row {number}")
  pair_id=hashlib.sha256(json.dumps([left,right],ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
  if row["pair_id"]!=pair_id or pair_id in seen_ids:raise ValueError(f"Invalid canonical pair_id at row {number}")
  a,b=metadata[left],metadata[right];expected={"left_pinyin":a["pinyin"],"right_pinyin":b["pinyin"],"equal_initial":a["initial"]==b["initial"],"equal_final":a["final"]==b["final"],"equal_tone":a["tone"]==b["tone"],"left_zero_initial":a["initial"]=="","right_zero_initial":b["initial"]=="","no_zero_initial_diagnostic":a["initial"]!="" and b["initial"]!=""};expected["overlap_mask"]="".join("1" if expected[x] else "0" for x in ("equal_initial","equal_final","equal_tone"))
  for field in ("equal_initial","equal_final","equal_tone","left_zero_initial","right_zero_initial","no_zero_initial_diagnostic"):
   if type(row[field]) is not bool:raise ValueError(f"{field} must be bool at row {number}")
  for field,want in expected.items():
   if row[field]!=want:raise ValueError(f"Recorded {field} mismatch at row {number}")
  seen.add((left,right));seen_ids.add(pair_id)
def prepare_population(evaluation,bundles):
 groups=evaluation["groups"];ids=[x.get("group_id") for x in groups]
 if any(not isinstance(x,str) or not x for x in ids) or len(ids)!=len(set(ids)):raise ValueError("Invalid or duplicate group_id values")
 rows=[x for x in groups if x.get("group_kind")=="syllable"];labels=[x.get("pinyin") for x in rows]
 if any(not isinstance(x,str) or not x.strip() for x in labels) or len(labels)!=len(set(labels)):raise ValueError("Invalid or duplicate pinyin labels")
 eligible=[];excluded=[]
 for row in rows:
  label=row["pinyin"].strip();initial=normalized_component(row.get("initial"),"initial",label,allow_empty=True);final=normalized_component(row.get("final"),"final",label);tone=normalized_component(row.get("tone"),"tone",label,coerce_string=True)
  if initial=="":excluded.append({"group_id":row["group_id"],"pinyin":label,"reason":"zero_initial"});continue
  eligible.append({"group_id":row["group_id"],"pinyin":label,"initial":initial,"final":final,"tone":tone})
 eligible.sort(key=lambda x:x["group_id"]);triples=[(x["initial"],x["final"],x["tone"]) for x in eligible]
 if len(triples)!=len(set(triples)):raise ValueError("Distinct eligible labels share component triples")
 required={x["pinyin"] for x in eligible};missing={name:sorted(required-set(bundle["vectors"])) for name,bundle in bundles.items()};missing={k:v for k,v in missing.items() if v}
 if missing:raise ValueError(f"Eligible syllables missing from embeddings: {missing}")
 validate_registered_pairs(evaluation["syllable_pairs"],eligible)
 coverage={"evaluation_groups_all":len(groups),"evaluation_syllable_groups":len(rows),"eligible_syllables":len(eligible),"excluded_zero_initial":len(excluded),"stored_syllable_pairs":len(evaluation["syllable_pairs"]),"full_population_pairs":len(eligible)*(len(eligible)-1)//2,"acoustic_union_labels":len(set().union(*(set(x["vectors"]) for x in bundles.values())))}
 for name,bundle in bundles.items():coverage[f"{name}_acoustic_labels"]=len(bundle["vectors"]);coverage[f"{name}_acoustic_extra"]=len(set(bundle["vectors"])-required)
 contracts={}
 for source_name,source in (("evaluation_manifest",evaluation["manifest"]),("evaluation_config",evaluation["config"]),("evaluation_coverage",evaluation["coverage"])):contracts[source_name]={k:source.get(k) for k in ("data_contract","pinyin_contract","component_contract") if k in source}
 for name,bundle in bundles.items():contracts[f"{name}_manifest"]={k:bundle["manifest"].get(k) for k in ("data_contract","pinyin_contract","component_contract") if k in bundle["manifest"]}
 flags=[]
 for key in ("data_contract","pinyin_contract","component_contract"):
  observed={src:v[key] for src,v in contracts.items() if key in v}
  if len({json.dumps(v,sort_keys=True) for v in observed.values()})>1:flags.append({"field":key,"values":observed,"status":"mismatch_review_required" if key in ("pinyin_contract","component_contract") else "different_domain_contracts_recorded"})
 if not any("pinyin_contract" in v for k,v in contracts.items() if k.startswith("evaluation_")):flags.append({"field":"pinyin_contract","values":{},"status":"absent_in_evaluation_recorded_limitation"})
 mismatch=[x for x in flags if x["status"]=="mismatch_review_required"]
 if mismatch:raise ValueError(f"Explicit contract mismatch requires review: {mismatch}")
 return eligible,excluded,coverage,contracts,flags
