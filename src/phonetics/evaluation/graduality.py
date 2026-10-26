"""Reusable graduality statistics preserving the Wav2Vec2 analysis estimands."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .anchor import edge_scores,prepare_anchor_cache

MODEL_ORDER=("base","zh-cn");MASK_ORDER=("000","001","010","011","100","101","110")
EDGE_SPECS=(("initial","000","100"),("initial","001","101"),("initial","010","110"),("final","000","010"),("final","001","011"),("final","100","110"),("tone","000","001"),("tone","010","011"),("tone","100","101"))
SECONDARY_EDGE_KEYS=(("tone","000","001"),("initial","000","100"),("initial","001","101"),("tone","100","101"))

def build_analysis_arrays(population,vectors,model_order=MODEL_ORDER,mask_order=MASK_ORDER):
    labels=[x["pinyin"] for x in population];components=np.asarray([[x["initial"],x["final"],x["tone"]] for x in population],object);pair_i,pair_j=np.triu_indices(len(labels),k=1);masks=np.asarray(["".join(x) for x in np.where(components[pair_i]==components[pair_j],"1","0")])
    if np.any(masks=="111"):raise ValueError("Off-diagonal mask 111 is forbidden")
    if set(masks)-set(mask_order):raise ValueError("Unexpected masks")
    matrices={};cosine={}
    for name in model_order:
        matrix=np.stack([vectors[name][label] for label in labels]);norm=np.linalg.norm(matrix,axis=1,keepdims=True)
        if np.any(norm==0):raise ValueError(f"Zero vector for {name}")
        matrix=matrix/norm;sim=matrix@matrix.T
        if not np.isfinite(sim).all() or not np.allclose(sim,sim.T,atol=1e-10) or sim.min() < -1-1e-8 or sim.max()>1+1e-8:raise ValueError(f"Invalid cosine matrix for {name}")
        matrices[name]=sim;cosine[name]=sim[pair_i,pair_j]
    expected=len(labels)*(len(labels)-1)//2
    if len(pair_i)!=expected or len(set(zip(pair_i.tolist(),pair_j.tolist())))!=expected:raise AssertionError("Full pair population incomplete")
    return labels,components,pair_i,pair_j,masks,matrices,cosine

def q_scores_for_weights(cache,weights,edge_specs=EDGE_SPECS,model_order=MODEL_ORDER):
    edge=pd.DataFrame(edge_scores(cache,edge_specs,weights,model_order));edge.insert(0,"level","edge");edge["delta"]=edge[model_order[1]]-edge[model_order[0]];rows=[]
    for component in ("initial","final","tone"):
        b=edge[edge.component==component];valid=len(b)==3 and b[list(model_order)].notna().all().all();rows.append({"level":"component","component":component,"low_mask":None,"high_mask":None,model_order[0]:b[model_order[0]].mean() if valid else np.nan,model_order[1]:b[model_order[1]].mean() if valid else np.nan,"delta":b.delta.mean() if valid else np.nan,"valid_anchors":np.nan})
    valid=len(edge)==9 and edge[list(model_order)].notna().all().all();rows.append({"level":"overall9","component":"all","low_mask":None,"high_mask":None,model_order[0]:edge[model_order[0]].mean() if valid else np.nan,model_order[1]:edge[model_order[1]].mean() if valid else np.nan,"delta":edge.delta.mean() if valid else np.nan,"valid_anchors":np.nan})
    secondary=edge.set_index(["component","low_mask","high_mask"]).reindex(SECONDARY_EDGE_KEYS);valid=secondary[list(model_order)].notna().all().all();rows.append({"level":"secondary4","component":"selected4","low_mask":None,"high_mask":None,model_order[0]:secondary[model_order[0]].mean() if valid else np.nan,model_order[1]:secondary[model_order[1]].mean() if valid else np.nan,"delta":secondary.delta.mean() if valid else np.nan,"valid_anchors":np.nan});return pd.concat([edge,pd.DataFrame(rows)],ignore_index=True)

def weighted_mask_means(values,masks,pair_i,pair_j,weights,mask_order=MASK_ORDER):
    pw=weights[pair_i]*weights[pair_j];means={};counts={}
    for mask in mask_order:
        selected=masks==mask;positive=pw[selected]>0;counts[mask]=int(positive.sum());means[mask]=np.average(values[selected][positive],weights=pw[selected][positive]) if positive.any() else np.nan
    return means,counts
def full_mask_summary(cosine,masks,model_order=MODEL_ORDER,mask_order=MASK_ORDER):
    rows=[]
    for model in model_order:
        for mask in mask_order:
            values=np.asarray(cosine[model])[masks==mask];rows.append({"model":model,"mask":mask,"n_pairs":len(values),"mean":np.mean(values) if len(values) else np.nan,"sd":np.std(values,ddof=1) if len(values)>1 else np.nan,"median":np.median(values) if len(values) else np.nan,"q025":np.quantile(values,.025) if len(values) else np.nan,"q25":np.quantile(values,.25) if len(values) else np.nan,"q75":np.quantile(values,.75) if len(values) else np.nan,"q975":np.quantile(values,.975) if len(values) else np.nan})
    return pd.DataFrame(rows)
def gap_rows(means,model_order=MODEL_ORDER,edge_specs=EDGE_SPECS):
    rows=[]
    for component,low,high in edge_specs:
        row={"kind":"edge","component":component,"low_mask":low,"high_mask":high};row.update({m:means[m][high]-means[m][low] for m in model_order});row["delta"]=row[model_order[1]]-row[model_order[0]];rows.append(row)
    for kind,low,high in (("D","000","110"),("D101_supplemental","000","101")):
        row={"kind":kind,"component":None,"low_mask":low,"high_mask":high};row.update({m:means[m][high]-means[m][low] for m in model_order});row["delta"]=row[model_order[1]]-row[model_order[0]];rows.append(row)
    return pd.DataFrame(rows)
def run_stimulus_bootstrap(repeats,seed,labels,cache,cosine,masks,pair_i,pair_j,model_order=MODEL_ORDER):
    rng=np.random.default_rng(seed);records=[];coverage=[];n=len(labels)
    for repeat in range(repeats):
        weights=np.bincount(rng.integers(0,n,size=n),minlength=n).astype(np.int64);q=q_scores_for_weights(cache,weights,model_order=model_order)
        for row in q.to_dict("records"):
            metric=f"Q_edge:{row['component']}:{row['low_mask']}->{row['high_mask']}" if row["level"]=="edge" else f"Q_{row['level']}:{row['component']}"
            for model in model_order:records.append({"repeat":repeat,"family":"Q","metric":metric,"model":model,"value":row[model]})
            records.append({"repeat":repeat,"family":"Q","metric":metric,"model":"delta_zh_minus_base","value":row["delta"]})
        mm={};counts={}
        for model in model_order:
            mm[model],counts[model]=weighted_mask_means(cosine[model],masks,pair_i,pair_j,weights)
            for mask in MASK_ORDER:records.append({"repeat":repeat,"family":"mask_mean","metric":mask,"model":model,"value":mm[model][mask]})
        for row in gap_rows(mm,model_order).to_dict("records"):
            metric=f"gap:{row['component']}:{row['low_mask']}->{row['high_mask']}" if row["kind"]=="edge" else row["kind"]
            for model in model_order:records.append({"repeat":repeat,"family":"gap","metric":metric,"model":model,"value":row[model]})
            records.append({"repeat":repeat,"family":"gap","metric":metric,"model":"delta_zh_minus_base","value":row["delta"]})
        coverage.append({"repeat":repeat,"unique_syllables":int(np.count_nonzero(weights)),**{f"pairs_{m}_{k}":counts[m][k] for m in model_order for k in MASK_ORDER}})
        edge_rows=q[q.level=="edge"].to_dict("records");coverage[-1].update({f"valid_Q_{r['component']}_{r['low_mask']}_to_{r['high_mask']}":int(np.isfinite(r["delta"])) for r in edge_rows});coverage[-1].update({f"anchors_Q_{r['component']}_{r['low_mask']}_to_{r['high_mask']}":int(r["valid_anchors"]) for r in edge_rows});coverage[-1]["valid_Q_overall9"]=int(np.isfinite(q[q.level=="overall9"].iloc[0]["delta"]))
    return pd.DataFrame(records),pd.DataFrame(coverage)
def summarize_bootstrap(replicates,repeats):
    rows=[]
    for (family,metric,model),block in replicates.groupby(["family","metric","model"],sort=True):
        values=block.value.to_numpy(float);values=values[np.isfinite(values)];rows.append({"family":family,"metric":metric,"model":model,"bootstrap_median":np.median(values) if len(values) else np.nan,"ci025":np.quantile(values,.025) if len(values) else np.nan,"ci975":np.quantile(values,.975) if len(values) else np.nan,"n_valid":len(values),"n_requested":repeats,"share_positive_sign_stability":np.mean(values>0) if len(values) and model=="delta_zh_minus_base" else np.nan})
    return pd.DataFrame(rows)

__all__=["MODEL_ORDER","MASK_ORDER","EDGE_SPECS","SECONDARY_EDGE_KEYS","build_analysis_arrays","prepare_anchor_cache","q_scores_for_weights","weighted_mask_means","full_mask_summary","gap_rows","run_stimulus_bootstrap","summarize_bootstrap"]
