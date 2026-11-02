"""Render approved evaluation summaries from immutable numerical plot data."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
from . import apply_style, export_figure, layout, load_protocol, shared_legend

MODEL_LABELS={"syllable":"Syllable","components":"Components","acoustic":"Acoustic","components-acoustic-lambda-1":"Combined λ1","components-acoustic-lambda-10":"Combined λ10"}
FIGURE_STEMS = ("anchor-q-transitions", "transition-shares", "component-shares", "dimsim-geometry", "dimsim-correlation")
def _semantic(p,model): return p.registry["semantics"][model]

def _digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _json(path): return json.loads(Path(path).read_text(encoding="utf-8"))

def _guard_not_in_immutable_run(target):
    path=Path(target).resolve()
    for ancestor in (path,*path.parents):
        manifest=ancestor/"manifest.json"
        if not manifest.is_file(): continue
        try: data=_json(manifest)
        except (OSError,ValueError): continue
        if (str(data.get("analysis", "")).startswith("contextual-")
                and data.get("status") in {"complete","failed"}):
            raise ValueError(f"Refusing to write inside immutable evaluation run: {ancestor}")

def _check_paired_outputs(aq,ds):
    am,dm=aq["manifest"],ds["manifest"]
    if am.get("analysis")!="contextual-anchor-q" or dm.get("analysis")!="contextual-dimsim":
        raise ValueError("output roles must be anchor-Q and DIMSIM respectively")
    ap,dp=am.get("scientific_input_pins",{}),dm.get("scientific_input_pins",{})
    if ap.get("analysis")!="contextual-anchor-q" or dp.get("analysis")!="contextual-dimsim":
        raise ValueError("scientific output roles do not match manifests")
    for key in ("bundle_manifest_sha256","model_ids","seeds"):
        if ap.get(key)!=dp.get(key): raise ValueError(f"paired outputs differ in {key}")
    def checkpoints(pins):
        rows=pins.get("model_seed_provenance")
        if not isinstance(rows,list): raise ValueError("model/seed checkpoint provenance is required")
        result={(x.get("model_id"),x.get("seed")):x for x in rows if isinstance(x,dict)}
        if len(result)!=len(rows) or not result or any(not re.fullmatch(r"[0-9a-f]{64}",str(x.get("checkpoint_sha256",""))) for x in result.values()):
            raise ValueError("invalid model/seed checkpoint provenance")
        return result
    if checkpoints(ap)!=checkpoints(dp): raise ValueError("paired outputs use different model/seed checkpoint provenance")

def build_figures(anchor_plot, dimsim_plot, protocol=None):
    """Build figures from the two persisted report plot_data objects."""
    p=protocol or load_protocol(); figs=[]
    models=list(MODEL_LABELS)
    try:
        # Q transition edge panels, with seed values retained as points.
        fig,axs=layout(3,p,columns=3,legend_rows=2,width_mm=180);figs.append(fig)
        components=("initial","final","tone");labels=None
        rows=anchor_plot["edges"]
        for ci,(ax,component) in enumerate(zip(axs,components)):
            subset=[r for r in rows if r["edge_id"].startswith(component+":")]
            labels=[r["edge_id"].split(":",1)[1] for r in subset if r["model_id"]==models[0]]
            for mi,model in enumerate(models):
                rr=[r for r in subset if r["model_id"]==model]
                x=np.arange(len(rr),dtype=float)
                s=_semantic(p,model)
                ax.scatter(x+(mi-2)*.055,[r["mean"] for r in rr],marker=s["marker"],c=s["color"],s=p.style["marker"]["size_pt"]**2,label=MODEL_LABELS[model],zorder=3)
                for j,r in enumerate(rr):
                    vals=[v["q"] for v in r["seed_points"]]
                    ax.scatter(x[j]+(mi-2)*.055+np.linspace(-.018,.018,len(vals)),vals,marker="o",facecolors="white",edgecolors="#777777",linewidths=.45,s=10,zorder=2)
            ax.set(title=component.title(),xticks=np.arange(4),xticklabels=labels,ylim=(0,1),ylabel="Anchor-Q");ax.tick_params(axis="x",labelrotation=35)
        shared_legend(fig,axs,p,rows=2);fig._phonetics_resolution={"kind":"anchor_q_transitions","model_order":models,"seed_points":5}

        # Q normalized share bars use values exactly as reported.
        fig,axs=layout(1,p,columns=1,legend_rows=2,width_mm=180);figs.append(fig);ax=axs[0]
        key,field,n=("transition_shares","edge_id",12);rs=anchor_plot[key]
        for ax in axs:
            for ci,comp in enumerate(components):
                values=[]
            first_valid=next((m for m in models if all(r["share"] is not None for r in rs if r["model_id"]==m)),None)
            for mi,model in enumerate(models):
                names=[r[field] for r in rs if r["model_id"]==model]
                vals=[next(r["share"] for r in rs if r["model_id"]==model and r[field]==name) for name in names]
                if len(vals)!=n or any(v is None for v in vals): continue
                if n==12:
                    bottom=0.0
                    for ei,(name,val) in enumerate(zip(names,vals)):
                        component=name.split(":",1)[0]
                        shade=("#555555","#999999","#cccccc")[("initial","final","tone").index(component)]
                        hatch=("","/","\\","x")[ei%4]
                        ax.bar(mi,val,bottom=bottom,width=.68,color=shade,hatch=hatch,edgecolor="black",linewidth=.25,label=f"{component.title()} {name.split(':',1)[1]}" if model==first_valid else None)
                        bottom+=val
                else:
                    b=0
                    for ci,val in enumerate(vals):
                        ax.bar(mi,val,bottom=b,width=.65,color=("#444444","#888888","#c5c5c5")[ci],hatch=("","//","xx")[ci],edgecolor="black",linewidth=.3,label=components[ci] if mi==0 else None);b+=val
            ax.set_ylim(0,1);ax.set_ylabel("Q share");ax.set_xticks(np.arange(5));ax.set_xticklabels([MODEL_LABELS[m] for m in models],rotation=25,ha="right")
            if n==12:
                for mi,model in enumerate(models):
                    if any(r["model_id"]==model and r.get("status")=="undefined_zero_total" for r in rs): ax.text(mi,.5,"undefined",rotation=90,ha="center",va="center")
        shared_legend(fig,axs,p,rows=2)
        fig._phonetics_resolution={"kind":"anchor_q_transition_shares","undefined_zero_total":"shown as absent bars"}

        fig,axs=layout(1,p,columns=1,legend_rows=1,width_mm=180);figs.append(fig);ax=axs[0]
        rs=anchor_plot["component_shares"]
        first_component_valid=next((m for m in models if all(r["share"] is not None for r in rs if r["model_id"]==m)),None)
        for mi,model in enumerate(models):
            vals=[next(r["share"] for r in rs if r["model_id"]==model and r["component"]==comp) for comp in components]
            if any(v is None for v in vals):
                ax.text(mi,.5,"undefined",rotation=90,ha="center",va="center");continue
            bottom=0.0
            for ci,(comp,val) in enumerate(zip(components,vals)):
                ax.bar(mi,val,bottom=bottom,width=.65,color=("#444444","#888888","#c5c5c5")[ci],hatch=("","//","xx")[ci],edgecolor="black",linewidth=.3,label=comp.title() if model==first_component_valid else None);bottom+=val
        ax.set(ylim=(0,1),ylabel="Q share",xticks=range(5),xticklabels=[MODEL_LABELS[m] for m in models]);shared_legend(fig,axs,p)
        fig._phonetics_resolution={"kind":"anchor_q_component_shares","undefined_zero_total":"shown as absent bars"}

        # Geometry panels with per-panel draw axes and strict support gaps.
        curves=dimsim_plot["curves"];bounds=dimsim_plot["boundaries"]
        fig,axs=layout(10,p,columns=2,legend_rows=3,width_mm=180);figs.append(fig)
        # Reserve physical room for the second y-axis on every panel.
        for host in axs:
            pos=host.get_position();host.set_position([pos.x0,pos.y0,pos.width-.06,pos.height])
        count_rows=dimsim_plot.get("mode_counts",[])
        max_bin_draws=max((pt["draw_count"] for row in curves for pt in row["seed_points"] if pt["draw_count"] is not None),default=1)
        for mi,model in enumerate(models):
            for col,mode in enumerate(("dimsim-natural","dimsim-balanced")):
                ax=axs[mi*2+col];cx=ax.twinx();ax.set_zorder(cx.get_zorder()+1);ax.patch.set_visible(False)
                s=_semantic(p,model)
                cur=[r for r in curves if r["model_id"]==model and r["mode"]==mode]
                x=np.arange(13);bar=next((r for r in count_rows if r["mode"]==mode),{})
                draws=bar.get("draw_count",0);unique=bar.get("unique_pair_count",0)
                bin_draws=[r["seed_points"][0]["draw_count"] for r in cur]
                bin_unique=[r["seed_points"][0]["unique_pair_count"] for r in cur]
                cx.bar(np.arange(len(cur)),bin_draws,width=.76,color="#e6e6e6",edgecolor="none",zorder=0)
                cx.bar(np.arange(len(cur)),bin_unique,width=.76,color="#a0a0a0",edgecolor="none",zorder=1)
                for j,r in enumerate(cur):
                    if r["mean_cosine"] is not None:
                        ax.plot([j],[r["mean_cosine"]],marker=s["marker"] if r["supported"] else "o",markerfacecolor=s["color"] if r["supported"] else "none",markeredgecolor=s["color"],linestyle="none",markersize=p.style["marker"]["size_pt"])
                # Five seed curves preserve the same unsupported gaps as the mean.
                for si in range(5):
                    for j in range(len(cur)-1):
                        a,b=cur[j]["seed_points"][si],cur[j+1]["seed_points"][si]
                        if cur[j]["supported"] and cur[j+1]["supported"] and a["mean_cosine"] is not None and b["mean_cosine"] is not None:
                            ax.plot([j,j+1],[a["mean_cosine"],b["mean_cosine"]],color="#aaaaaa",linewidth=p.style["line"]["reference_pt"])
                bd=next(r for r in bounds if r["model_id"]==model and r["mode"]==mode)
                bx=(bd["median"]-5)/10 if bd["median"] is not None else None
                for j in range(len(cur)-1):
                    a,b=cur[j],cur[j+1]
                    if a["supported"] and b["supported"] and a["mean_cosine"] is not None and b["mean_cosine"] is not None:
                        if bx is not None and j<bx<j+1:
                            by=np.interp(bx,[j,j+1],[a["mean_cosine"],b["mean_cosine"]])
                            ax.plot([j,bx],[a["mean_cosine"],by],color="#171717",linewidth=p.style["line"]["data_pt"],linestyle="-")
                            ax.plot([bx,j+1],[by,b["mean_cosine"]],color="#171717",linewidth=p.style["line"]["data_pt"],linestyle="--")
                        else:
                            solid=bx is None or bx>=j+1
                            ax.plot([j,j+1],[a["mean_cosine"],b["mean_cosine"]],color="#171717",linewidth=p.style["line"]["data_pt"],linestyle="-" if solid else "--")
                if bx is not None:
                    ax.axvline(bx,color="#171717",linewidth=p.style["line"]["reference_pt"])
                    j=int(np.floor(bx))
                    if 0<=j<len(cur)-1 and cur[j]["supported"] and cur[j+1]["supported"] and cur[j]["mean_cosine"] is not None and cur[j+1]["mean_cosine"] is not None:
                        by=np.interp(bx,[j,j+1],[cur[j]["mean_cosine"],cur[j+1]["mean_cosine"]]);ax.plot([bx],[by],marker="D",color="#171717",markersize=p.style["marker"]["size_pt"])
                cx.set_ylim(0,max_bin_draws*1.05);cx.set_yticks([0,max_bin_draws]);cx.set_yticklabels(["0",str(max_bin_draws)]);ax.set_ylim(-1,1);ax.set_xlim(-.5,12.5);ax.set_xticks(range(13));ax.set_xticklabels([str(x) for x in range(5,120,10)]+["[120,∞)"],rotation=45)
                ax.set_title(f"{MODEL_LABELS[model]} · {mode}");ax.set_ylabel("Cosine similarity" if col==0 else "");cx.set_ylabel("Pair draws" if col==1 else "")
                if mi==4: ax.set_xlabel("DIMSIM distance (bin centers)",labelpad=-2)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        fig.legend(handles=[Line2D([],[],color="#171717",label="Mean cosine"),Line2D([],[],color="#aaaaaa",label="Seed curves"),Patch(facecolor="#e6e6e6",label="All draws"),Patch(facecolor="#a0a0a0",label="Unique pairs"),Line2D([],[],color="#171717",linestyle="--",marker="D",label="Exploratory boundary")],labels=["Mean cosine","Seed curves","All draws","Unique pairs","Exploratory boundary"],loc="upper center",bbox_to_anchor=(.5,.995),ncol=5,frameon=False)
        fig._phonetics_resolution={"kind":"dimsim_geometry","support_gaps_not_bridged":True,"boundary":"exploratory median of valid seed thresholds"}

        # Full range Spearman rho per mode.
        fig,axs=layout(2,p,columns=2,legend_rows=2,width_mm=180);figs.append(fig)
        rho=dimsim_plot.get("rho",[])
        for ax,mode in zip(axs,("dimsim-natural","dimsim-balanced")):
            rr=[r for r in rho if r["mode"]==mode]
            for mi,model in enumerate(models):
                r=next((v for v in rr if v["model_id"]==model),None)
                if not r or r.get("mean") is None:continue
                s=_semantic(p,model);ax.scatter([mi],[r["mean"]],marker=s["marker"],c=s["color"],label=MODEL_LABELS[model]);vals=[x["rho"] for x in r["seed_values"] if x["rho"] is not None];ax.scatter([mi]*len(vals),vals,marker=".",c="#aaaaaa",s=10)
            ax.axhline(0,color="#777777",lw=p.style["line"]["reference_pt"]);ax.set(title=mode,xlim=(-.5,4.5),xticks=range(5),xticklabels=[MODEL_LABELS[m] for m in models],ylabel="Spearman ρ");ax.tick_params(axis="x",labelrotation=25)
        shared_legend(fig,axs,p,rows=2);fig._phonetics_resolution={"kind":"dimsim_correlation","five_seed_points":True}
        return figs
    except Exception:
        for fig in figs: plt.close(fig)
        raise

def render_evaluation(anchor_output, dimsim_output, output_dir, source_digest, *, protocol=None):
    """Render five PNG/EPS pairs after validating both complete output manifests."""
    from phonetics.evaluation.evaluation_outputs import load_evaluation_outputs
    aq=load_evaluation_outputs(anchor_output);ds=load_evaluation_outputs(dimsim_output);_check_paired_outputs(aq,ds)
    if not isinstance(source_digest,str) or not re.fullmatch(r"[0-9a-f]{64}",source_digest): raise ValueError("source_digest must be an explicit lowercase SHA-256 digest")
    p=protocol or load_protocol();apply_style(p);out=Path(output_dir);_guard_not_in_immutable_run(out)
    stems=FIGURE_STEMS
    targets=[out/f"{stem}.{ext}" for stem in stems for ext in ("png","eps","metadata.json")]
    if any(x.exists() for x in targets): raise FileExistsError("Refusing to overwrite evaluation figure exports")
    dplot=dict(ds["plot_data"]);dplot["rho"]=ds["tables"].get("model_rho",[])
    figs=build_figures(aq["plot_data"],dplot,p);saved=[]
    try:
        for fig,stem,fid in zip(figs,stems,("anchor_q_transitions","transition_shares","component_shares","dimsim_geometry","dimsim_correlation")):
            saved.extend(export_figure(fig,out,stem,fid,p,scientific_provenance={"source_digest":source_digest,"anchor_q_output":str(aq["root"]),"dimsim_output":str(ds["root"]),"anchor_q_manifest_sha256":_digest(aq["root"]/"manifest.json"),"dimsim_manifest_sha256":_digest(ds["root"]/"manifest.json")}))
            saved.append(out / f"{stem}.metadata.json")
        return saved
    finally:
        for fig in figs: plt.close(fig)
