"""Wav2Vec2 graduality research orchestration and CLI."""
from __future__ import annotations
from dataclasses import asdict,dataclass
from hashlib import sha256
import argparse,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import file_sha256,research_run
from ..evaluation.graduality import (EDGE_SPECS,MASK_ORDER,build_analysis_arrays,full_mask_summary,gap_rows,prepare_anchor_cache,q_scores_for_weights,run_stimulus_bootstrap,summarize_bootstrap)
from ..evaluation import graduality_io as gio
from ..visualization.graduality import render_graduality

@dataclass(frozen=True)
class GradualityConfig:
    project_root:Path
    run_id:str
    base_root:Path
    zh_cn_root:Path
    evaluation_root:Path
    base_manifest_sha256:str
    zh_cn_manifest_sha256:str
    evaluation_manifest_sha256:str
    bootstrap_repeats:int=1000
    bootstrap_seed:int=20261024
    def resolved(self):return {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in asdict(self).items()}

def _json(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def _jsonl(path):return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
def _verified(root,expected):
    root=Path(root).resolve();mp=root/"manifest.json"
    if file_sha256(mp)!=expected:raise ValueError(f"Unexpected manifest SHA-256: {mp}")
    manifest=_json(mp);artifacts=manifest.get("artifacts")
    if not isinstance(artifacts,dict) or not artifacts:raise ValueError(f"Missing artifacts map: {mp}")
    for relative,want in artifacts.items():
        path=(root/relative).resolve()
        if root not in path.parents or not path.is_file() or file_sha256(path)!=want:raise ValueError(f"Invalid artifact: {relative}")
    return manifest
def _embedding(name,root,expected,model_id):
    manifest=_verified(root,expected);config=_json(Path(root)/"config.json");inventory=_jsonl(Path(root)/"sound_inventory.jsonl")
    if manifest.get("schema_version")!=1 or manifest.get("bundle_type")!="acoustic_embeddings":raise ValueError(f"Unsupported embedding contract: {name}")
    model=manifest.get("model",{});observed=model.get("id") or model.get("model_id") or model.get("name") or model.get("pretrained_model_name_or_path")
    if observed!=model_id:raise ValueError(f"Unexpected model: {name}")
    expected_settings={"layer":-1,"pooling":"mean_all","padding":"none","sample_rate":16000,"mono":True,"dtype":"float32"}
    for key,want in expected_settings.items():
        if config.get("model",{}).get(key,model.get(key))!=want:raise ValueError(f"Unexpected {name}/{key}")
    labels=[x.get("label") for x in inventory]
    if any(not isinstance(x,str) or not x for x in labels) or len(labels)!=len(set(labels)):raise ValueError(f"Invalid inventory: {name}")
    with (Path(root)/"pinyin_embeddings.pkl").open("rb") as f:raw=pickle.load(f)
    if set(raw)!=set(labels):raise ValueError(f"Embedding inventory mismatch: {name}")
    vectors={k:np.asarray(v,dtype=float) for k,v in raw.items()}
    if any(v.ndim!=1 or not np.isfinite(v).all() or np.linalg.norm(v)==0 for v in vectors.values()):raise ValueError(f"Invalid vectors: {name}")
    return {"root":Path(root).resolve(),"manifest":manifest,"config":config,"inventory":inventory,"inventory_sha256":file_sha256(Path(root)/"sound_inventory.jsonl"),"vectors":vectors}
def _pair_plan(rows,eligible):
    metadata={x["group_id"]:x for x in eligible};seen=set()
    for number,row in enumerate(rows,1):
        required=("pair_id","left_id","right_id","left_pinyin","right_pinyin","equal_initial","equal_final","equal_tone","overlap_mask","left_zero_initial","right_zero_initial","no_zero_initial_diagnostic")
        if any(x not in row for x in required):raise ValueError(f"Missing pair-plan field at row {number}")
        left,right=row["left_id"],row["right_id"]
        if left not in metadata or right not in metadata or left>=right or (left,right) in seen:raise ValueError(f"Invalid pair endpoints at row {number}")
        expected_id=sha256(json.dumps([left,right],ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        if row["pair_id"]!=expected_id:raise ValueError(f"Invalid pair_id at row {number}")
        a,b=metadata[left],metadata[right];expected={"left_pinyin":a["pinyin"],"right_pinyin":b["pinyin"],"equal_initial":a["initial"]==b["initial"],"equal_final":a["final"]==b["final"],"equal_tone":a["tone"]==b["tone"],"left_zero_initial":False,"right_zero_initial":False,"no_zero_initial_diagnostic":True};expected["overlap_mask"]="".join("1" if expected[x] else "0" for x in ("equal_initial","equal_final","equal_tone"))
        for key,value in expected.items():
            if row[key]!=value or (key.startswith("equal_") or key.endswith("initial") or key=="no_zero_initial_diagnostic") and type(row[key]) is not bool:raise ValueError(f"Pair-plan mismatch {key} at row {number}")
        seen.add((left,right))
def _evaluation(root,expected):
    manifest=_verified(root,expected)
    if manifest.get("schema_version")!=2 or manifest.get("pairing_policy")!="nonzero_nonzero":raise ValueError("Unsupported evaluation contract")
    return {"root":Path(root).resolve(),"manifest":manifest,"config":_json(Path(root)/"config.json"),"coverage":_json(Path(root)/"coverage.json"),"groups":_jsonl(Path(root)/"groups.jsonl"),"pairs":_jsonl(Path(root)/"pairs/syllable.jsonl")}
def _population(evaluation,bundles):
    eligible=[];excluded=[]
    for row in evaluation["groups"]:
        if row.get("group_kind")!="syllable":continue
        label=row.get("pinyin");initial=row.get("initial");final=row.get("final");tone=str(row.get("tone")).strip()
        if not isinstance(label,str) or not label.strip() or not isinstance(initial,str) or not isinstance(final,str) or not final.strip() or not tone:raise ValueError("Invalid syllable components")
        if not initial.strip():excluded.append(row);continue
        eligible.append({"group_id":row["group_id"],"pinyin":label.strip(),"initial":initial.strip(),"final":final.strip(),"tone":tone})
    eligible.sort(key=lambda x:x["group_id"]);triples=[(x["initial"],x["final"],x["tone"]) for x in eligible]
    if len(triples)!=len(set(triples)):raise ValueError("Off-diagonal 111 component triples")
    required={x["pinyin"] for x in eligible}
    if any(required-set(x["vectors"]) for x in bundles.values()):raise ValueError("Eligible syllables missing from embeddings")
    _pair_plan(evaluation["pairs"],eligible);return eligible,excluded

def run_graduality(config:GradualityConfig):
    """Execute one complete load, analysis, artifact, and render run."""
    if config.bootstrap_repeats<1:raise ValueError("bootstrap_repeats must be positive")
    package=Path(__file__).parents[1];viz=package/"visualization";sources=[Path(__file__),Path(__file__).with_name("artifacts.py"),Path(__file__).with_name("__init__.py"),package/"__init__.py",package/"evaluation"/"__init__.py",package/"evaluation"/"anchor.py",package/"evaluation"/"graduality.py",package/"evaluation"/"graduality_io.py",viz/"__init__.py",viz/"core.py",viz/"graduality.py",viz/"style.toml",viz/"registry.toml",package/"pinyin.py"]
    inputs=[config.base_root,config.zh_cn_root,config.evaluation_root]
    expected={"config.json","data/pairs.npz","data/bootstrap_replicates.csv","data/bootstrap_plot.npz","data/plot_spec.json","tables/q_metrics.csv","tables/gap_metrics.csv","tables/mask_summary.csv","tables/mask_extrema.csv","tables/bootstrap_summary.csv","diagnostics/bootstrap_coverage.csv","diagnostics/q_edge_coverage.csv","diagnostics/coverage.csv","diagnostics/contracts.json","diagnostics/population.json","figures/cosine-distributions.png","figures/cosine-distributions.eps","figures/cosine-distributions.metadata.json","figures/cosine-density.png","figures/cosine-density.eps","figures/cosine-density.metadata.json","figures/bootstrap-differences.png","figures/bootstrap-differences.eps","figures/bootstrap-differences.metadata.json"}
    with research_run(config.project_root,"wav2vec2-graduality",config.run_id,inputs,config.resolved(),sources,expected) as (root,manifest):
        bundles={"base":gio.load_embedding_bundle("base",config.base_root,config.base_manifest_sha256,"facebook/wav2vec2-base"),"zh-cn":gio.load_embedding_bundle("zh-cn",config.zh_cn_root,config.zh_cn_manifest_sha256,"jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")}
        if bundles["base"]["inventory_sha256"]!=bundles["zh-cn"]["inventory_sha256"]:raise ValueError("Embedding inventories differ")
        for a,b in zip(bundles["base"]["inventory"],bundles["zh-cn"]["inventory"]):
            for field in ("label","sha256","relative_path","sample_rate","channels","frames"):
                if a.get(field)!=b.get(field):raise ValueError(f"Inventory mismatch: {field}")
        evaluation=gio.load_evaluation_bundle(config.evaluation_root,config.evaluation_manifest_sha256);population,excluded,pop_coverage,contracts,contract_flags=gio.prepare_population(evaluation,bundles);labels,components,pair_i,pair_j,masks,matrices,cosine=build_analysis_arrays(population,{k:v["vectors"] for k,v in bundles.items()});cache,edge_coverage=prepare_anchor_cache(components,matrices,EDGE_SPECS);full_q=q_scores_for_weights(cache,np.ones(len(labels),dtype=np.int64));mask_summary=full_mask_summary(cosine,masks);mask_means={m:mask_summary[mask_summary.model==m].set_index("mask")["mean"].to_dict() for m in ("base","zh-cn")};gaps=gap_rows(mask_means);replicates,coverage=run_stimulus_bootstrap(config.bootstrap_repeats,config.bootstrap_seed,labels,cache,cosine,masks,pair_i,pair_j);summary=summarize_bootstrap(replicates,config.bootstrap_repeats)
        q_tidy=full_q.melt(id_vars=["level","component","low_mask","high_mask","valid_anchors"],value_vars=["base","zh-cn","delta"],var_name="model",value_name="full_estimate");q_tidy["model"]=q_tidy.model.replace({"delta":"delta_zh_minus_base"});q_tidy["metric"]=q_tidy.apply(lambda r:f"Q_edge:{r.component}:{r.low_mask}->{r.high_mask}" if r.level=="edge" else f"Q_{r.level}:{r.component}",axis=1);q_report=q_tidy.merge(summary[summary.family=="Q"],on=["metric","model"],how="left")
        gap_tidy=gaps.melt(id_vars=["kind","component","low_mask","high_mask"],value_vars=["base","zh-cn","delta"],var_name="model",value_name="full_estimate");gap_tidy["model"]=gap_tidy.model.replace({"delta":"delta_zh_minus_base"});gap_tidy["metric"]=gap_tidy.apply(lambda r:f"gap:{r.component}:{r.low_mask}->{r.high_mask}" if r.kind=="edge" else r.kind,axis=1);gap_report=gap_tidy.merge(summary[summary.family=="gap"],on=["metric","model"],how="left")
        extrema=pd.concat([mask_summary.loc[mask_summary.groupby("model")["mean"].idxmin(),["model","mask","mean"]].assign(extremum="observed_min"),mask_summary.loc[mask_summary.groupby("model")["mean"].idxmax(),["model","mask","mean"]].assign(extremum="observed_max")],ignore_index=True)
        qdelta=replicates[(replicates.metric=="Q_overall9:all")&(replicates.model=="delta_zh_minus_base")].value.to_numpy(float);ddelta=replicates[(replicates.metric=="D")&(replicates.model=="delta_zh_minus_base")].value.to_numpy(float);qedges=np.histogram_bin_edges(qdelta[np.isfinite(qdelta)],bins=40);dedges=np.histogram_bin_edges(ddelta[np.isfinite(ddelta)],bins=40)
        np.savez_compressed(root/"data/pairs.npz",labels=np.asarray(labels),pair_i=pair_i,pair_j=pair_j,masks=masks,base=cosine["base"],zh_cn=cosine["zh-cn"],cosine_edges=np.linspace(-1,1,61));np.savez_compressed(root/"data/bootstrap_plot.npz",q_values=qdelta,d_values=ddelta,q_edges=qedges,d_edges=dedges);replicates.to_csv(root/"data/bootstrap_replicates.csv",index=False,float_format="%.17g");plot_spec={"mask_order":list(MASK_ORDER),"series":{"base":"Base","zh-cn":"Zh-cn","delta_zh_minus_base":"Zh-cn − Base"},"cosine_edges":np.linspace(-1,1,61).tolist(),"bootstrap":{"bins":40,"density":False,"canonical":"bootstrap_plot.npz","Q_overall9:all":{"array":"q_values","edges":"q_edges","valid":int(np.isfinite(qdelta).sum())},"D":{"array":"d_values","edges":"d_edges","valid":int(np.isfinite(ddelta).sum())}},"figures":["Cosine distributions","Cosine density","Bootstrap differences"]};(root/"data/plot_spec.json").write_text(json.dumps(plot_spec,indent=2),encoding="utf-8");q_report.to_csv(root/"tables/q_metrics.csv",index=False);gap_report.to_csv(root/"tables/gap_metrics.csv",index=False);mask_summary.to_csv(root/"tables/mask_summary.csv",index=False);extrema.to_csv(root/"tables/mask_extrema.csv",index=False);summary.to_csv(root/"tables/bootstrap_summary.csv",index=False);coverage.to_csv(root/"diagnostics/bootstrap_coverage.csv",index=False);pd.DataFrame(edge_coverage).to_csv(root/"diagnostics/q_edge_coverage.csv",index=False);pd.DataFrame([pop_coverage]).to_csv(root/"diagnostics/coverage.csv",index=False);(root/"diagnostics/contracts.json").write_text(json.dumps({"contracts":contracts,"flags":contract_flags},indent=2),encoding="utf-8");(root/"diagnostics/population.json").write_text(json.dumps({"labels":labels,"excluded":excluded,"full_population_pairs":len(pair_i)},indent=2),encoding="utf-8")
        data_hash=file_sha256(root/"data/pairs.npz");code_hash=file_sha256(Path(__file__).parents[1]/"visualization"/"graduality.py");render_graduality(root/"data",root/"figures",code_hash);manifest["scientific_provenance"]={"models":{k:{"id":gio.model_identifier(v["manifest"]["model"]),"manifest_sha256":v["manifest_sha256"],"extraction":{x:v["config"]["model"].get(x) for x in gio.EXTRACTION}} for k,v in bundles.items()},"evaluation_manifest_sha256":evaluation["manifest_sha256"],"edge_specs":EDGE_SPECS,"mask_order":MASK_ORDER,"bootstrap":{"seed":config.bootstrap_seed,"repeats":config.bootstrap_repeats},"histograms":{"cosine_edges":61,"bootstrap_bins":40},"plot_data_sha256":data_hash,"figure_source_sha256":code_hash};return root

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--base-root",type=Path,required=True);p.add_argument("--zh-cn-root",type=Path,required=True);p.add_argument("--evaluation-root",type=Path,required=True);a=p.parse_args(argv);run_graduality(GradualityConfig(a.project_root,a.run_id,a.base_root,a.zh_cn_root,a.evaluation_root))
if __name__=="__main__":main()
