"""Generic weighted anchor-comparison evaluation."""
from __future__ import annotations
import numpy as np

def masks_from_components(components, anchor):
    values=np.asarray(components,dtype=object)
    return np.asarray(["".join(bits) for bits in np.where(values==values[anchor],"1","0")])

def prepare_anchor_cache(components, similarities, edge_specs):
    components=np.asarray(components,dtype=object);n=len(components);cache={name:{} for name in similarities};coverage=[]
    for component,low_mask,high_mask in edge_specs:
        edge=(component,low_mask,high_mask);candidates={}
        for anchor in range(n):
            masks=masks_from_components(components,anchor);low=np.flatnonzero(masks==low_mask);high=np.flatnonzero(masks==high_mask);low=low[low!=anchor];high=high[high!=anchor]
            if len(low) and len(high):candidates[anchor]=(low,high)
        coverage.append({"component":component,"low_mask":low_mask,"high_mask":high_mask,"anchors":len(candidates),"n_low_min":min((len(x[0]) for x in candidates.values()),default=0),"n_low_max":max((len(x[0]) for x in candidates.values()),default=0),"n_high_min":min((len(x[1]) for x in candidates.values()),default=0),"n_high_max":max((len(x[1]) for x in candidates.values()),default=0)})
        for name,sim in similarities.items():
            sim=np.asarray(sim,float);model={}
            for anchor,(low,high) in candidates.items():
                order=np.argsort(sim[anchor,low],kind="mergesort");model[anchor]={"low":low[order],"low_scores":sim[anchor,low][order],"high":high,"high_scores":sim[anchor,high]}
            cache[name][edge]=model
    return cache,coverage

def weighted_anchor_score(entry,stimulus_weights):
    weights=np.asarray(stimulus_weights,float);lw=weights[entry["low"]];hw=weights[entry["high"]];tl=lw.sum();th=hw.sum()
    if tl<=0 or th<=0:return np.nan
    prefix=np.concatenate(([0.],np.cumsum(lw,dtype=float)));left=np.searchsorted(entry["low_scores"],entry["high_scores"],side="left");right=np.searchsorted(entry["low_scores"],entry["high_scores"],side="right");wins=prefix[left]+.5*(prefix[right]-prefix[left]);return float(np.dot(hw,wins)/(tl*th))

def edge_scores(cache,edge_specs,stimulus_weights,model_order):
    rows=[];weights=np.asarray(stimulus_weights)
    for edge in edge_specs:
        values={};counts={}
        for model in model_order:
            q=[];aw=[]
            for anchor,entry in cache[model][edge].items():
                if weights[anchor]<=0:continue
                value=weighted_anchor_score(entry,weights)
                if np.isfinite(value):q.append(value);aw.append(weights[anchor])
            values[model]=np.average(q,weights=aw) if aw else np.nan;counts[model]=len(aw)
        if len(set(counts.values()))!=1:raise AssertionError("Model anchor eligibility diverged")
        rows.append({"component":edge[0],"low_mask":edge[1],"high_mask":edge[2],**values,"valid_anchors":next(iter(counts.values()))})
    return rows
