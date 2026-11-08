"""Reusable atomic research-run artifact management."""
from __future__ import annotations
from contextlib import contextmanager
from hashlib import sha256
import json,os,platform,re,subprocess
from importlib.metadata import PackageNotFoundError,version
from pathlib import Path

def file_sha256(path,chunk=1024*1024):
    h=sha256()
    with Path(path).open("rb") as f:
        while data:=f.read(chunk):h.update(data)
    return h.hexdigest()
def safe_run_id(value):
    if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",value) or value in {".",".."}:raise ValueError("run_id must be a non-empty safe filename component")
    return value
def safe_analysis(value):
    if not isinstance(value,str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}",value):raise ValueError("analysis must be a safe lowercase name")
    return value
def ensure_output_outside_inputs(output,input_roots):
    output=Path(output).resolve()
    for root in map(lambda p:Path(p).resolve(),input_roots):
        if output==root or root in output.parents:raise ValueError(f"Output must not be inside input: {root}")
def write_json_new(path,value):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
def git_sha(root):
    try:return subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
    except Exception:return None

def source_name(path,project_root):
    path=Path(path).resolve()
    try:return path.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:return 'sdk/'+path.relative_to(Path(__file__).resolve().parents[2]).as_posix()
def _expected_paths(values):
    if not values:raise ValueError("expected_files must be a non-empty collection")
    result=set()
    for value in values:
        path=Path(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {".",""}:raise ValueError(f"Unsafe expected artifact path: {value}")
        result.add(path.as_posix())
    if "manifest.json" in result:raise ValueError("manifest.json cannot self-hash")
    return result
@contextmanager
def research_run(project_root,analysis,run_id,input_roots,config,source_paths,expected_files):
    expected=_expected_paths(expected_files)
    run_id=safe_run_id(run_id);analysis=safe_analysis(analysis);root=Path(project_root).resolve()/"outputs"/analysis/run_id;ensure_output_outside_inputs(root,input_roots);root.mkdir(parents=True,exist_ok=False)
    manifest={"schema_version":1,"analysis":analysis,"run_id":run_id,"status":"running","required_artifacts":sorted(expected),"artifacts":[]};write_json_new(root/"manifest.json",manifest)
    try:
        for name in ("data","tables","figures","diagnostics"):(root/name).mkdir()
        dependencies={}
        for name in ("numpy","pandas","matplotlib","scipy"):
            try:dependencies[name]=version(name)
            except PackageNotFoundError:dependencies[name]=None
        inputs=[]
        for p in input_roots:
            p=Path(p).resolve();identity=p if p.is_file() else p/"manifest.json"
            if not identity.is_file():raise FileNotFoundError(f"Input identity file is missing: {identity}")
            inputs.append({"path":str(p),"identity_path":str(identity),"sha256":file_sha256(identity)})
        write_json_new(root/"config.json",config);manifest.update({"git_sha":git_sha(project_root),"python":platform.python_version(),"dependencies":dependencies,"source_hashes":{source_name(p,project_root):file_sha256(p) for p in source_paths},"inputs":inputs})
        yield root,manifest
        actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name!="manifest.json"}
        missing=expected-actual;unexpected=actual-expected
        if missing or unexpected:raise RuntimeError(f"Artifact contract mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        empty=[name for name in expected if (root/name).stat().st_size<=0]
        if empty:raise RuntimeError(f"Required artifacts are empty: {sorted(empty)}")
        artifacts=[]
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.name!="manifest.json":artifacts.append({"path":p.relative_to(root).as_posix(),"sha256":file_sha256(p),"bytes":p.stat().st_size})
        manifest["artifacts"]=artifacts;manifest["status"]="complete"
    except Exception as exc:
        manifest["status"]="failed";manifest["error"]={"type":type(exc).__name__,"message":str(exc)};raise
    finally:
        temp=root/"manifest.json.tmp";temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8");os.replace(temp,root/"manifest.json")
