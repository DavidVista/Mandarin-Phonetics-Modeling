"""Thin orchestration helpers for the generated five-model evaluation notebooks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_ORDER = ("syllable", "components", "acoustic", "components-acoustic-lambda-1", "components-acoustic-lambda-10")
SEEDS = (0, 1, 2, 3, 4)

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _build_source_digest(build_metadata: dict) -> str:
    hashes = build_metadata.get("source_sha256") if isinstance(build_metadata, dict) else None
    if not isinstance(hashes, dict) or not hashes or any(not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v) for v in hashes.values()):
        raise ValueError("BUILD_METADATA requires the complete source SHA-256 mapping")
    payload=json.dumps(hashes,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _reject_immutable_ancestors(target: Path) -> None:
    for ancestor in (target, *target.parents):
        manifest=ancestor/"manifest.json"
        if not manifest.is_file(): continue
        try: record=json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError,ValueError): continue
        if (str(record.get("analysis", "")).startswith("contextual-")
                and record.get("status") in {"complete", "failed"}):
            raise ValueError(f"Refusing output beneath immutable evaluation run: {ancestor}")

def preflight_evaluation(config: dict, build_metadata: dict) -> dict[str, Any]:
    """Validate explicit inputs and every 25-seed/output target without creating files."""
    from .bundle import load_evaluation_bundle
    from .seed_runs import discover_seed_runs
    from ..research.artifacts import safe_run_id, ensure_output_outside_inputs
    from .evaluation_outputs import load_contextual_extraction_outputs
    from .extraction_batch import resolve_encoder_source_hash
    required=("bundle_path","bundle_manifest_sha256","experiment_path","experiment_manifest_sha256","model_run_paths",
              "extraction_mode","extraction_prefix","existing_representation_runs","project_root","reference_model","run_ids",
              "device","batch_size","dimsim_boundary_start","dimsim_boundary_stop","dimsim_boundary_step",
              "dimsim_min_side_count","dimsim_min_unique_pairs_per_bin")
    if not isinstance(config,dict) or set(required)-set(config): raise ValueError("The single parameter cell must define every evaluation input and output")
    def path_value(key):
        value=config[key]
        if not isinstance(value,(str,Path)) or not str(value).strip() or "EDIT_ME" in str(value).upper(): raise ValueError(f"Set explicit {key} in the parameter cell")
        return Path(value).resolve(strict=True)
    bundle_path=path_value("bundle_path"); experiment_path=path_value("experiment_path")
    project_root=Path(config["project_root"]).expanduser().resolve()
    if not str(config["project_root"]).strip() or "EDIT_ME" in str(config["project_root"]).upper(): raise ValueError("Set explicit project_root in the parameter cell")
    bundle_manifest=bundle_path/"manifest.json"; experiment_manifest=experiment_path/"manifest.json"
    if _sha(bundle_manifest)!=config["bundle_manifest_sha256"]: raise ValueError("Evaluation bundle manifest SHA-256 mismatch")
    if _sha(experiment_manifest)!=config["experiment_manifest_sha256"]: raise ValueError("Experiment manifest SHA-256 mismatch")
    bundle=load_evaluation_bundle(bundle_path)
    if bundle["manifest"].get("schema_version")!=3: raise ValueError("A schema-3 contextual evaluation bundle is required")
    model_paths=config["model_run_paths"]
    if not isinstance(model_paths,dict) or set(model_paths)!=set(MODEL_ORDER): raise ValueError("Provide exactly the five canonical model run paths")
    if (config["reference_model"]!="syllable" or not isinstance(config["device"],str) or not config["device"].strip()
            or type(config["batch_size"]) is not int or config["batch_size"]<=0):
        raise ValueError("Set the syllable reference, a nonempty device, and a positive integer batch size")
    if any(type(config[k]) is not int for k in ("dimsim_boundary_start","dimsim_boundary_stop","dimsim_boundary_step","dimsim_min_side_count","dimsim_min_unique_pairs_per_bin")):
        raise ValueError("DIMSIM thresholds and support counts must be integers")
    if (config["dimsim_boundary_start"]<10 or config["dimsim_boundary_stop"]>110 or config["dimsim_boundary_stop"]<config["dimsim_boundary_start"]
            or config["dimsim_boundary_step"]<=0 or config["dimsim_min_side_count"]<1 or config["dimsim_min_unique_pairs_per_bin"]<1):
        raise ValueError("DIMSIM thresholds must lie in 10..110 with positive step and support counts")
    if config["reference_model"]!="syllable": raise ValueError("reference_model must be syllable")
    runs={}; rep_ids={}; discovered={}
    encoder_sha=resolve_encoder_source_hash(build_metadata)
    from .extraction_batch import _exp_hashes
    experiment_hashes=_exp_hashes(experiment_path,config["experiment_manifest_sha256"])
    if bundle["manifest"].get("source_experiment",{}).get("hashes")!={k:v for k,v in experiment_hashes.items() if k!="acoustic_targets.npy"}:
        raise ValueError("Selected experiment artifacts do not match the bundle's complete source inventory")
    if config["extraction_mode"] not in {"extract","reuse"}: raise ValueError("extraction_mode must be extract or reuse")
    if config["extraction_mode"]=="extract":
        if config["existing_representation_runs"] is not None: raise ValueError("extract mode requires existing_representation_runs=None")
        prefix=safe_run_id(config["extraction_prefix"])
    else:
        prefix=None
        existing=config["existing_representation_runs"]
        if not isinstance(existing,dict) or set(existing)!=set(MODEL_ORDER): raise ValueError("reuse mode requires exact model -> seed representation pins")
        for model in MODEL_ORDER:
            if not isinstance(existing[model],dict) or set(existing[model])!=set(SEEDS): raise ValueError(f"reuse mode requires exactly seeds 0..4 for {model}")
            for seed,item in existing[model].items():
                if not isinstance(item,dict) or set(item)!={"path","manifest_sha256"}: raise ValueError(f"invalid representation pin for {model} seed {seed}")
    for key in ("bundle_manifest_sha256","experiment_manifest_sha256"):
        if not isinstance(config[key],str) or len(config[key])!=64 or any(c not in "0123456789abcdef" for c in config[key]): raise ValueError(f"Set explicit lowercase SHA-256 for {key}")
    for model in MODEL_ORDER:
        run_path=Path(model_paths[model]).resolve(strict=True)
        found=discover_seed_runs(run_path,model,SEEDS);runs[model]=run_path;discovered[model]=found
        if found["experiment_hashes"]!=experiment_hashes:
            raise ValueError(f"Training source inventory does not match selected bundle: {model}")
        seed_records={row["seed"]:row for row in found["seeds"]}
        found["seeds_by_id"]=seed_records
        rep_ids[model]={seed:(safe_run_id(f"{prefix}-{model}-seed-{seed}") if prefix else None) for seed in SEEDS}
    ids=[rep_ids[m][s] for m in MODEL_ORDER for s in SEEDS]
    if prefix and len(set(ids))!=25: raise ValueError("Expanded extraction IDs collide")
    run_ids=config["run_ids"]
    expected_run_names={"contextual-anchor-q","contextual-dimsim","contextual-random-control"}
    if not isinstance(run_ids,dict) or set(run_ids)!=expected_run_names: raise ValueError("Explicit Q, DIMSIM, and random run IDs are required")
    for value in run_ids.values(): safe_run_id(value)
    numerical_targets=[project_root/"outputs"/name/run_ids[name] for name in expected_run_names]
    targets=list(numerical_targets)
    if config["extraction_mode"]=="extract": targets.extend(project_root/"outputs"/"contextual-representations"/run_id for run_id in ids)
    # Validate all immutable destinations before any model extraction or scoring.
    for target in targets:
        ensure_output_outside_inputs(target,[bundle_path,experiment_path,*runs.values()])
        _reject_immutable_ancestors(target)
        if target.exists(): raise FileExistsError(f"Evaluation output target already exists: {target}")
    reps={m:{} for m in MODEL_ORDER}
    if config["extraction_mode"]=="reuse":
        for model in MODEL_ORDER:
            for seed in SEEDS:
                item=config["existing_representation_runs"][model][seed]
                rep_root=Path(item["path"]).resolve(strict=True)
                if not isinstance(item.get("manifest_sha256"),str) or _sha(rep_root/"manifest.json")!=item["manifest_sha256"]:
                    raise ValueError(f"Explicit representation manifest pin mismatch: {model} seed {seed}")
                safe_run_id(rep_root.name)
                export=load_contextual_extraction_outputs(rep_root)
                pin=export["manifest"]["scientific_input_pins"];record=discovered[model]["seeds_by_id"][seed]
                expected={"bundle_manifest_sha256":config["bundle_manifest_sha256"],"experiment_manifest_sha256":config["experiment_manifest_sha256"],
                          "model_id":model,"seed":seed,"checkpoint_sha256":record["checkpoint_sha256"],
                          "training_manifest_sha256":discovered[model]["manifest_sha256"]}
                if any(pin.get(k)!=v for k,v in expected.items()) or export["manifest"]["source_hashes"].get("evaluation/encoders.py")!=encoder_sha:
                    raise ValueError(f"Reused contextual extraction pins mismatch for {model} seed {seed}")
                rep_ids[model][seed]=export["manifest"]["run_id"]
                reps[model][seed]=rep_root/"data"
        ids=[rep_ids[m][s] for m in MODEL_ORDER for s in SEEDS]
        if len(set(ids))!=25: raise ValueError("Reused representation run IDs must be distinct")
    if config["extraction_mode"]=="extract":
        from .extraction_batch import plan_contextual_extraction
        context_plan=plan_contextual_extraction(bundle_path=bundle_path,bundle_manifest_sha256=config["bundle_manifest_sha256"],
            experiment_path=experiment_path,experiment_manifest_sha256=config["experiment_manifest_sha256"],training_runs=runs,
            run_ids=rep_ids,output_root=project_root/"outputs",device=config["device"],batch_size=config["batch_size"],source_sha256=encoder_sha)
    else: context_plan=None
    return {"config":config,"build_metadata":build_metadata,"bundle":bundle,"bundle_path":bundle_path,
            "experiment_path":experiment_path,"project_root":project_root,"model_runs":runs,"discovered":discovered,
            "representation_ids":rep_ids,"representation_dirs":reps,"source_hashes":dict(build_metadata["source_sha256"]),
            "source_digest":_build_source_digest(build_metadata),"encoder_sha256":encoder_sha,
            "extraction_plan":context_plan}

def extract_or_reuse(context: dict, *, extractor=None) -> dict:
    """Load all 25 pinned reusable exports, or execute one complete explicit extraction plan."""
    if context["config"]["extraction_mode"]=="reuse": return context["representation_dirs"]
    from .evaluation_outputs import run_contextual_extraction_outputs
    c=context["config"]
    run_ids=context["representation_ids"]
    plan=context["extraction_plan"]
    result=run_contextual_extraction_outputs(plan,extractor=extractor)
    context["extraction_result"]=result
    return result["representation_dirs"]

def score_anchor_q(context: dict, representation_dirs: dict) -> dict:
    from .anchor_q_batch import score_anchor_q_batch
    return score_anchor_q_batch(context["bundle_path"],context["model_runs"],representation_dirs)

def score_dimsim(context: dict, representation_dirs: dict) -> dict:
    from .dimsim_reporting import score_contextual_dimsim_batch
    c=context["config"]
    analysis={"boundary_thresholds":list(range(c["dimsim_boundary_start"],c["dimsim_boundary_stop"]+1,c["dimsim_boundary_step"])),
              "boundary_min_side_count":c["dimsim_min_side_count"],"min_unique_pairs_per_bin":c["dimsim_min_unique_pairs_per_bin"]}
    return score_contextual_dimsim_batch(context["bundle_path"],context["model_runs"],representation_dirs,analysis)

def score_random_control(context: dict, representation_dirs: dict) -> dict:
    from .random_control import resolve_contextual_dimension,generate_random_control,evaluate_random_control
    resolution=resolve_contextual_dimension(representation_dirs,bundle_manifest_sha256=context["config"]["bundle_manifest_sha256"],bundle=context["bundle"])
    generated=generate_random_control(sorted(row["occurrence_id"] for row in context["bundle"]["occurrences"]),dimension=resolution["dimension"])
    c=context["config"]
    analysis={"boundary_thresholds":list(range(c["dimsim_boundary_start"],c["dimsim_boundary_stop"]+1,c["dimsim_boundary_step"])),
              "boundary_min_side_count":c["dimsim_min_side_count"],"min_unique_pairs_per_bin":c["dimsim_min_unique_pairs_per_bin"]}
    random_sha=context["source_hashes"].get("evaluation/random_control.py")
    if not isinstance(random_sha,str): raise ValueError("build metadata lacks random-control source SHA-256")
    return evaluate_random_control(context["bundle"],generated,bundle_manifest_sha256=c["bundle_manifest_sha256"],
        dimension_resolution=resolution,source_sha256=random_sha,analysis_config=analysis)

def persist_results(context: dict, q_batch: dict, dimsim_batch: dict, random_result: dict) -> dict:
    """Save numerical results only, with no plotting or presentation dependencies."""
    from .report_contract import build_evaluation_report
    from .evaluation_outputs import write_evaluation_outputs, write_random_control_outputs
    from ..research.artifacts import ensure_output_outside_inputs, safe_run_id

    c = context["config"]
    run_ids = c["run_ids"]
    analyses = ("contextual-anchor-q", "contextual-dimsim", "contextual-random-control")
    if not isinstance(run_ids, dict) or set(run_ids) != set(analyses):
        raise ValueError("Explicit Q, DIMSIM, and random run IDs are required")
    inputs = [context["bundle_path"], context["experiment_path"], *context["model_runs"].values()]
    targets = {
        name: Path(context["project_root"]).resolve() / "outputs" / name / safe_run_id(run_ids[name])
        for name in analyses
    }
    # Check all three destinations before the first numerical run is written.
    for target in targets.values():
        ensure_output_outside_inputs(target, inputs)
        _reject_immutable_ancestors(target)
        if target.exists():
            raise FileExistsError(f"Evaluation output target already exists: {target}")
    numerical_ids = {name: run_ids[name] for name in analyses[:2]}
    contract = build_evaluation_report(q_batch, dimsim_batch, c["reference_model"], numerical_ids)

    def jsonable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(v) for v in value]
        return value

    # Older callers may still supply these settings; they do not affect numerical outputs.
    data_config = {k: v for k, v in c.items() if k not in {"presentation_id", "export_formats", "export_dpi"}}
    resolved = jsonable({
        **data_config,
        "expanded_representation_ids": context["representation_ids"],
        "model_ids": list(MODEL_ORDER),
        "seeds": list(SEEDS),
        "source_digest": context["source_digest"],
        "analysis_config": {
            "dimsim_boundary_thresholds": list(range(
                c["dimsim_boundary_start"], c["dimsim_boundary_stop"] + 1, c["dimsim_boundary_step"]
            )),
        },
    })
    numerical = write_evaluation_outputs(
        context["project_root"], contract, run_ids=numerical_ids,
        input_roots=inputs, source_hashes=context["source_hashes"], resolved_config=resolved,
    )
    random_path = write_random_control_outputs(
        context["project_root"], run_ids["contextual-random-control"], random_result,
        input_roots=inputs, source_hashes=context["source_hashes"], resolved_config=resolved,
    )
    return {"numerical": numerical, "random_control": random_path}
