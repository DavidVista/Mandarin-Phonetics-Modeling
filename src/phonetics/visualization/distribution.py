"""Render the three approved DIMSIM distribution figure families."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from . import apply_style, export_figure, layout, load_protocol, plot_ecdf, shared_legend

MASK_ORDER = tuple(f"{i:03b}" for i in range(8))

def render_distribution(data_dir, figures_dir, source_digest):
    data_dir, figures_dir = Path(data_dir), Path(figures_dir)
    saved = np.load(data_dir / "distribution_plot_data.npz", allow_pickle=False)
    rows = json.loads((data_dir / "component_shares.json").read_text(encoding="utf-8"))
    protocol = load_protocol(); apply_style(protocol); results=[]; created=[]
    try:
        spec=protocol.figure("dimsim_distributions"); fig,axes=layout(3,protocol); created.append(fig); d=saved["distance"]
        axes[0].hist(d,bins=saved["full_histogram_edges"],color=protocol.style["color"]["foreground"]); axes[0].set(title="Full range",xlabel=spec["xlabel"],ylabel="Pair count")
        x=np.sort(d,kind="mergesort"); axes[1].step(x,np.arange(1,len(x)+1)/len(x),where="post",color=protocol.style["color"]["foreground"]); axes[1].set(title="ECDF",xlabel=spec["xlabel"],ylabel="ECDF",ylim=(0,1))
        axes[2].hist(saved["detail_values"],bins=50,color=protocol.style["color"]["foreground"]); axes[2].set(title="Detail: D < 10",xlabel=spec["xlabel"],ylabel="Pair count",xlim=(0,10))
        fig._phonetics_resolution={"panels":["full_histogram","ecdf","detail_lt_10"],"counts":[len(d),len(d),len(saved["detail_values"])],"full_edges":saved["full_histogram_edges"].tolist(),"detail_bins":50}
        results.extend(export_figure(fig,figures_dir,"dimsim-distributions","dimsim_distributions",protocol,scientific_provenance={"source_digest":source_digest})); plt.close(fig)
        spec=protocol.figure("dimsim_overlap_distributions"); fig,axes=layout(1,protocol,legend_rows=2); created.append(fig); series={m:d[saved["overlap_mask"]==m] for m in MASK_ORDER}
        for mask,values in series.items():
            if not len(values): continue
            x=np.sort(values,kind="mergesort"); sem=protocol.registry["semantics"][mask]; axes[0].step(x,np.arange(1,len(x)+1)/len(x),where="post",label=sem["label"],color=sem["color"],linestyle=sem["linestyle"],marker=sem["marker"],markevery=max(1,len(x)//20),markersize=protocol.style["marker"]["size_pt"])
        axes[0].set(xlabel=spec["xlabel"],ylabel=spec["ylabel"],ylim=(0,1)); shared_legend(fig,axes,protocol,rows=2); fig._phonetics_resolution={"mask_order":list(MASK_ORDER),"counts":{m:len(v) for m,v in series.items()}}
        results.extend(export_figure(fig,figures_dir,"component-overlap-distributions","dimsim_overlap_distributions",protocol,scientific_provenance={"source_digest":source_digest})); plt.close(fig)
        spec=protocol.figure("dimsim_component_contributions"); fig,axes=layout(1,protocol,legend_rows=1,width_mm=protocol.style["layout"]["multi_width_mm"]); created.append(fig); ax=axes[0]; labels=[f"{10*i}\u2013{10*(i+1)}" for i in range(12)]+["\u2265120"]; bottom=np.zeros(len(rows))
        for (key,sid),hatch in zip((("initial_share","initial"),("final_share","final"),("tone_share","tone")),("","///","xxx")):
            values=np.asarray([np.nan if r[key] is None else r[key] for r in rows],float); sem=protocol.registry["semantics"][sid]; ax.bar(np.arange(len(rows)),values,bottom=bottom,label=sem["label"],color=sem["color"],edgecolor=protocol.style["color"]["foreground"],linewidth=protocol.style["line"]["axes_pt"],hatch=hatch); bottom=np.where(np.isfinite(values),bottom+values,bottom)
        ax.set(xticks=np.arange(len(rows)),xticklabels=labels,xlabel=spec["xlabel"],ylabel=spec["ylabel"],ylim=(0,1)); shared_legend(fig,axes,protocol); fig._phonetics_resolution={"bins":[r["bin"] for r in rows],"display_labels":labels,"counts":[r["count"] for r in rows],"undefined_bins":[r["bin"] for r in rows if r["initial_share"] is None]}
        results.extend(export_figure(fig,figures_dir,"dimsim-component-contributions","dimsim_component_contributions",protocol,scientific_provenance={"source_digest":source_digest})); plt.close(fig); return results
    finally:
        for fig in created: plt.close(fig)
