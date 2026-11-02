"""Human-readable views of persisted evaluation tables; values remain unrounded in JSON."""
from __future__ import annotations
from pathlib import Path
import json

def _guard_not_in_immutable_run(target):
    path=Path(target).resolve()
    for ancestor in (path,*path.parents):
        manifest=ancestor/"manifest.json"
        if not manifest.is_file(): continue
        try: data=json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError,ValueError): continue
        if (data.get("scope")=="numerical_outputs" and data.get("analysis") in {"contextual-anchor-q","contextual-dimsim"}
                and data.get("status") in {"complete","failed"}):
            raise ValueError(f"Refusing to write inside immutable evaluation run: {ancestor}")

def _fmt(v): return "\u2014" if v is None else (str(v) if isinstance(v,int) and not isinstance(v,bool) else f"{v:.3f}")
def _md(title, headers, rows):
    out=[f"# {title}","","| "+" | ".join(headers)+" |","|"+"|".join("---" for _ in headers)+"|"]
    out += ["| "+" | ".join(str(x) for x in row)+" |" for row in rows]
    return "\n".join(out)+"\n"

def render_evaluation_tables(anchor_tables:dict,dimsim_tables:dict,output_dir):
    """Write Markdown tables from report tables without recalculating summaries."""
    out=Path(output_dir)
    _guard_not_in_immutable_run(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError("Table output directory must be empty")
    out.mkdir(parents=True,exist_ok=True)
    artifacts={}
    def write(name,body):
        if Path(name).name!=name or not name.endswith(".md"): raise ValueError("Unsafe table filename")
        (out/name).write_text(body,encoding="utf-8");artifacts[name]=body
    models=sorted({r["model_id"] for r in anchor_tables["model_seed_edges"]})
    # Keep the primary 12-edge table small and self-contained for each run.
    for model in models:
        for seed in range(5):
            rows=[r for r in anchor_tables["model_seed_edges"] if r["model_id"]==model and r["seed"]==seed]
            fields=("edge_id","q","mc_se","draw_count","unique_triplets","unique_anchor_occurrences","unique_anchor_syllables","unique_occurrences")
            write(f"anchor_q_edges_{model}_seed_{seed}.md",_md(f"Anchor-Q edges: {model}, seed {seed}",list(fields),[[r["edge_id"],_fmt(r["q"]),_fmt(r["mc_se"]),r["draw_count"],*[(r.get("coverage") or {}).get(k,"—") for k in fields[4:]]] for r in rows]))
        metric_rows=[r for r in anchor_tables["model_seed_metrics"] if r["model_id"]==model]
        summary=anchor_tables.get("model_metric_seed_summary",[])
        fields=("metric","seed","q","mc_se")
        write(f"anchor_q_metrics_{model}.md",_md(f"Anchor-Q metrics: {model}",list(fields),[[r["metric"],r["seed"],_fmt(r["q"]),_fmt(r["mc_se"])] for r in metric_rows]))
        sm=[r for r in summary if r["model_id"]==model]
        write(f"anchor_q_metric_summary_{model}.md",_md(f"Anchor-Q mean and sample SD: {model}",["Metric","Mean","Sample SD","MC SE"],[[r["metric"],_fmt(r["mean"]),_fmt(r["sample_sd"]),_fmt(r["mc_se"])] for r in sm]))
        erows=[r for r in anchor_tables.get("model_edge_seed_summary",[]) if r["model_id"]==model]
        write(f"anchor_q_edge_summary_{model}.md",_md(f"Anchor-Q edge summary: {model}",["Edge","Mean","Sample SD","MC SE"],[[r["edge_id"],_fmt(r["mean"]),_fmt(r["sample_sd"]),_fmt(r["mc_se"])] for r in erows]))
        metric_names=("initial","final","tone","overall")
        by_seed={seed:{r["metric"]:_fmt(r["q"]) for r in anchor_tables["model_seed_metrics"] if r["model_id"]==model and r["seed"]==seed} for seed in range(5)}
        smap={r["metric"]:r for r in anchor_tables.get("model_metric_seed_summary",[]) if r["model_id"]==model}
        rows=[[f"Seed {seed}"]+[by_seed[seed].get(metric,"—") for metric in metric_names] for seed in range(5)]
        rows.append(["Mean ± sample SD"]+[f"{_fmt(smap[m]['mean'])} ± {_fmt(smap[m]['sample_sd'])}" for m in metric_names if m in smap])
        write(f"anchor_q_metrics_wide_{model}.md",_md(f"Anchor-Q metrics by seed: {model}",["Seed",*metric_names],rows))
    for key,title,fields in (("model_seed_edges","Anchor-Q edge values",("model_id","seed","edge_id","q","mc_se","draw_count")),("model_seed_metrics","Anchor-Q metric values",("model_id","seed","metric","q","mc_se")),("global_metric_summary","Anchor-Q global summaries",("model_id","metric","mean","sample_sd","mc_se")),("paired_deltas","Paired Anchor-Q differences",("model_id","reference_model","metric","mean_delta_q","sample_sd_seed_delta","mc_se")),("transition_shares","Anchor-Q transition shares",("model_id","edge_id","share","status")),("component_shares","Anchor-Q component shares",("model_id","component","share","status"))):
        rows=[[r.get(f) if isinstance(r.get(f),str) else _fmt(r.get(f)) for f in fields] for r in anchor_tables[key]]
        write(f"{key}.md",_md(title,list(fields),rows))
    paired=[]
    for r in anchor_tables["paired_deltas"]:
        paired.append([r["model_id"],r["metric"],f"{_fmt(r['mean_delta_q'])} ({_fmt(r['sample_sd_seed_delta'])}; {_fmt(r['mc_se'])})"])
    write("anchor_q_paired_delta_cells.md",_md("Paired Anchor-Q differences",["Model","Metric","Mean ΔQ (sample SD; MC SE)"],paired))
    metric_names=("initial","final","tone","overall")
    paired_map={(r["model_id"],r["metric"]):r for r in anchor_tables["paired_deltas"]}
    paired_models=sorted({r["model_id"] for r in anchor_tables["paired_deltas"]})
    paired_wide=[[model]+[f"{_fmt(paired_map[(model,m)]['mean_delta_q'])} ({_fmt(paired_map[(model,m)]['sample_sd_seed_delta'])}; {_fmt(paired_map[(model,m)]['mc_se'])})" for m in metric_names] for model in paired_models]
    write("anchor_q_paired_deltas_wide.md",_md("Paired Anchor-Q differences",["Model",*metric_names],paired_wide))
    global_map={(r["model_id"],r["metric"]):r for r in anchor_tables["global_metric_summary"]}
    global_models=sorted({r["model_id"] for r in anchor_tables["global_metric_summary"]})
    global_wide=[[model]+[f"{_fmt(global_map[(model,m)]['mean'])} ± {_fmt(global_map[(model,m)]['sample_sd'])}" for m in metric_names]+[_fmt(global_map[(model,m)]["mc_se"]) for m in metric_names] for model in global_models]
    write("anchor_q_global_summary_wide.md",_md("Anchor-Q global summary",["Model",*[f"{m} mean ± SD" for m in metric_names],*[f"{m} MC SE" for m in metric_names]],global_wide))
    for key,title,fields in (("seed_rho","DIMSIM full-range rho by seed",("model_id","seed","mode","rho","valid_draw_count","unique_pair_count","status")),("model_rho","DIMSIM full-range rho summaries",("model_id","mode","mean","sample_sd","status")),("paired_rho_deltas","Paired DIMSIM rho differences",("model_id","reference_model","mode","mean","sample_sd","status")),("seed_boundary","Exploratory boundary thresholds",("model_id","seed","mode","threshold","status")),("bins","DIMSIM curve bins and support",("model_id","mode","bin","mean_cosine","supported"))):
        rows=[]
        for r in dimsim_tables[key]:
            row=[]
            for f in fields:
                v=r.get(f)
                if f=="bin" and isinstance(r.get("seed_points"),list):
                    pts=r["seed_points"];v=f"{v}; draws={pts[0].get('draw_count')}; unique={pts[0].get('unique_pair_count')}" if pts else v
                row.append(v if isinstance(v,(str,bool)) else _fmt(v))
            rows.append(row)
        write(f"{key}.md",_md(title,list(fields),rows))
    if "model_boundary" in dimsim_tables:
        rows=[[r.get("model_id"),r.get("mode"),_fmt(r.get("mean")),_fmt(r.get("sample_sd")),_fmt(r.get("median")),r.get("status")] for r in dimsim_tables["model_boundary"]]
        write("dimsim_boundary_summary.md",_md("Exploratory DIMSIM boundary summary",["Model","Mode","Mean","Sample SD","Median","Status"],rows))
    if "model_rho" in dimsim_tables:
        modes=sorted({r["mode"] for r in dimsim_tables["model_rho"]});rho_map={(r["model_id"],r["mode"]):r for r in dimsim_tables["model_rho"]};dmodels=sorted({r["model_id"] for r in dimsim_tables["model_rho"]})
        wide=[[model]+[f"{_fmt(rho_map[(model,mode)]['mean'])} ± {_fmt(rho_map[(model,mode)]['sample_sd'])}" for mode in modes] for model in dmodels]
        write("dimsim_model_rho_wide.md",_md("DIMSIM full-range rho by model and mode",["Model",*modes],wide))
    return sorted(out.glob("*.md"))
