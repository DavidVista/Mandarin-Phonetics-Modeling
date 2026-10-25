"""Immutable, portable full-inventory acoustic embedding bundles."""
from __future__ import annotations
import importlib.metadata,json,os,pickle,platform,re,shutil,tempfile
from pathlib import Path
from ..artifacts import read_jsonl,safe_artifact,sha256_file,write_json,write_jsonl
from ..data_contract import DATA_CONTRACT_VERSION,validate_acoustic
from ..pinyin import PARSER_PROVENANCE, PINYIN_CONTRACT_VERSION,canonical_labels,decompose
from .config import SCHEMA_VERSION,validate_model_config,_keys
from .embeddings import embed_inventory
from .readers import read_sounds

REQUIRED={"pinyin_embeddings.pkl","sound_inventory.jsonl","config.json","reports/inference.json"}

def prepare_acoustic_bundle(config,*,embedding_backend=None):
    if not isinstance(config,dict) or set(config)!={"bundle_id","output_root","cache_root","sounds","model"}: raise ValueError("invalid acoustic bundle config keys")
    cfg=dict(config); _keys(cfg["sounds"],{"path","tone_convention"},{"sha256","manifest"},"sounds");validate_model_config(cfg["model"])
    if cfg["sounds"]["tone_convention"] not in {"explicit","neutral_unmarked"}: raise ValueError("invalid tone convention")
    bid=cfg["bundle_id"]
    if not isinstance(bid,str) or not bid or Path(bid).name!=bid: raise ValueError("invalid bundle_id")
    out=Path(cfg["output_root"]).resolve();out.mkdir(parents=True,exist_ok=True);target=out/bid
    if target.exists(): raise FileExistsError(target)
    inventory,source=read_sounds(cfg["sounds"])
    for row in inventory: decompose(row["label"])
    vectors,inference=embed_inventory(inventory,cfg["model"],cfg["cache_root"],embedding_backend)
    temp=Path(tempfile.mkdtemp(prefix=f".{bid}.",dir=out))
    try:
        (temp/"reports").mkdir();
        with (temp/"pinyin_embeddings.pkl").open("wb") as h: pickle.dump(vectors,h,pickle.HIGHEST_PROTOCOL)
        portable=[{k:v for k,v in row.items() if k!="path"} for row in inventory]
        write_jsonl(temp/"sound_inventory.jsonl",portable);write_json(temp/"config.json",{"schema_version":1,**cfg})
        write_json(temp/"reports/inference.json",inference)
        artifacts={str(p.relative_to(temp)).replace("\\","/"):sha256_file(p) for p in temp.rglob("*") if p.is_file()}
        env={"python":platform.python_version()}
        for package in ("numpy","torch","transformers"):
            try: env[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError: pass
        code_root=Path(__file__).parents[1]
        code_files=[code_root/"pinyin.py",code_root/"data_contract.py",code_root/"artifacts.py",*sorted(Path(__file__).parent.glob("*.py"))]
        code={str(p.relative_to(code_root)).replace("\\","/"):sha256_file(p) for p in code_files}
        model={**cfg["model"],"backend":inference["backend"]}
        write_json(temp/"manifest.json",{"bundle_type":"acoustic_embeddings","schema_version":SCHEMA_VERSION,"data_contract":DATA_CONTRACT_VERSION,
          "pinyin_contract":PINYIN_CONTRACT_VERSION,"parser_provenance":PARSER_PROVENANCE,"bundle_id":bid,"manifest_self_hash":"excluded","artifacts":artifacts,"source":source,
          "inventory_sha256":sha256_file(temp/"sound_inventory.jsonl"),"count":len(inventory),"dimension":len(next(iter(vectors.values()))),
          "model":model,"inference":inference,"environment":env,"code":code})
        validate_acoustic_bundle(temp);os.rename(temp,target);return target
    except Exception: shutil.rmtree(temp,ignore_errors=True);raise

def validate_acoustic_bundle(path,*,expected_manifest_sha256=None):
    root=Path(path).resolve();mp=root/"manifest.json"
    if expected_manifest_sha256 and sha256_file(mp)!=expected_manifest_sha256: raise ValueError("acoustic bundle manifest checksum mismatch")
    manifest=json.loads(mp.read_text(encoding="utf-8"))
    if manifest.get("bundle_type")!="acoustic_embeddings" or manifest.get("schema_version")!=1 or manifest.get("data_contract")!=DATA_CONTRACT_VERSION or manifest.get("pinyin_contract") != PINYIN_CONTRACT_VERSION or manifest.get("parser_provenance") != PARSER_PROVENANCE: raise ValueError("unsupported acoustic bundle")
    arts=manifest.get("artifacts",{});
    if not REQUIRED.issubset(arts): raise ValueError("missing acoustic bundle artifacts")
    for rel,digest in arts.items():
        p=safe_artifact(root,rel)
        if not p.is_file() or sha256_file(p)!=digest: raise ValueError(f"artifact checksum mismatch: {rel}")
    inv=read_jsonl(root/"sound_inventory.jsonl")
    if not inv or len({r.get("label") for r in inv})!=len(inv): raise ValueError("invalid acoustic inventory")
    for row in inv:
        label=row.get("label");relative=row.get("relative_path");digest=row.get("sha256")
        if ("path" in row or not isinstance(relative,str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts
                or not isinstance(label,str) or canonical_labels([label],tone_convention="explicit")[0]!=label
                or not isinstance(digest,str) or re.fullmatch(r"[0-9a-f]{64}",digest) is None): raise ValueError("inventory is not portable or canonical")
        decompose(label)
    with (root/"pinyin_embeddings.pkl").open("rb") as h:vectors=pickle.load(h)
    dim=validate_acoustic(vectors,expected_keys=[r["label"] for r in inv])
    cfg=json.loads((root/"config.json").read_text(encoding="utf-8"));model=validate_model_config(cfg["model"])
    if cfg.get("bundle_id")!=manifest.get("bundle_id"): raise ValueError("bundle identity mismatch")
    if {**model,"backend":manifest["model"].get("backend")}!=manifest.get("model"): raise ValueError("bundle model metadata mismatch")
    inf=json.loads((root/"reports/inference.json").read_text(encoding="utf-8"))
    for field in ("computed","cached","errors"):
        if not isinstance(inf.get(field),int) or isinstance(inf.get(field),bool) or inf[field]<0: raise ValueError("invalid inference counts")
    if inf.get("backend") not in {"stub","transformers"} or manifest.get("model",{}).get("backend")!=inf.get("backend"): raise ValueError("invalid inference backend")
    if inf!=manifest.get("inference") or inf["computed"]+inf["cached"]!=len(inv) or inf["errors"]!=0: raise ValueError("invalid inference counts")
    for field in ("count","dimension"):
        if not isinstance(manifest.get(field),int) or isinstance(manifest.get(field),bool) or manifest[field]<=0: raise ValueError("bundle counts mismatch")
    if manifest["count"]!=len(inv) or manifest["dimension"]!=dim or manifest.get("inventory_sha256")!=sha256_file(root/"sound_inventory.jsonl"): raise ValueError("bundle counts mismatch")
    return manifest

def load_acoustic_bundle(path,manifest_sha256):
    manifest=validate_acoustic_bundle(path,expected_manifest_sha256=manifest_sha256);root=Path(path)
    with (root/"pinyin_embeddings.pkl").open("rb") as h:vectors=pickle.load(h)
    return vectors,read_jsonl(root/"sound_inventory.jsonl"),manifest
