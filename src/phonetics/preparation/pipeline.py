"""Direct prepared-dataset API backed by the reusable staged IDS flow."""
from __future__ import annotations
import json,shutil,tempfile
from pathlib import Path
from ..artifacts import sha256_file,write_json
from .config import validate_config
from .acoustic_bundle import prepare_acoustic_bundle
from .assembly import assemble_dataset,validate_assembled
from .corpus_bundle import prepare_corpus


def prepare_dataset(config:dict,*,embedding_backend=None)->Path:
    cfg=validate_config(config)
    output=Path(cfg["output_root"]).resolve();output.mkdir(parents=True,exist_ok=True)
    target=output/cfg["dataset_id"]
    if target.exists():raise FileExistsError(f"prepared dataset already exists: {target}")
    if "acoustic_bundle" in cfg and embedding_backend is not None:raise ValueError("embedding_backend is invalid in precomputed mode")
    temp=Path(tempfile.mkdtemp(prefix=f".{cfg['dataset_id']}.ids.",dir=output))
    write_json(temp/"requested-config.json",cfg);config_hash=sha256_file(temp/"requested-config.json")
    try:
        if "acoustic_bundle" in cfg:
            acoustic=cfg["acoustic_bundle"]
        else:
            acoustic_path=prepare_acoustic_bundle({"bundle_id":f"{cfg['dataset_id']}.acoustic","output_root":str(temp/"acoustic"),"cache_root":cfg["cache_root"],"sounds":cfg["sounds"],"model":cfg["model"]},embedding_backend=embedding_backend)
            acoustic={"path":str(acoustic_path),"manifest_sha256":sha256_file(acoustic_path/"manifest.json")}
        processed=prepare_corpus({"corpus_id":f"{cfg['dataset_id']}.processed","output_root":str(temp/"processed"),"corpus":cfg["corpus"],"ids":cfg["ids"],"sampling":cfg["sampling"],"acoustic_bundle":acoustic},conflict_policy="error")
        return assemble_dataset({"dataset_id":cfg["dataset_id"],"output_root":str(output),"processed_corpus":{"path":str(processed),"manifest_sha256":sha256_file(processed/"manifest.json")},"acoustic_bundle":acoustic})
    except Exception as exc:
        details=getattr(exc,"diagnostics",{})
        inner=temp/"processed"/f"{cfg['dataset_id']}.processed.failed.json"
        if inner.is_file():details=json.loads(inner.read_text(encoding="utf-8")).get("details",details)
        write_json(output/f"{cfg['dataset_id']}.failed.json",{"schema_version":2,"dataset_id":cfg["dataset_id"],"run_id":str(__import__("uuid").uuid4()),"config_sha256":config_hash,"status":"failure","exception":type(exc).__name__,"message":str(exc),"details":details})
        raise
    finally:
        shutil.rmtree(temp,ignore_errors=True)


def validate_prepared(path:str|Path)->dict:
    return validate_assembled(path)
