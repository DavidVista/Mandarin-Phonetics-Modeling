"""Render the three approved graduality figure families from saved plot data."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import apply_style,export_figure,layout,load_protocol,plot_ecdf,plot_histogram,shared_legend
from ..evaluation.graduality import MASK_ORDER

def render_graduality(data_dir,figures_dir,source_digest):
    """Render all figure families; caller supplies complete plot-ready arrays."""
    data_dir=Path(data_dir);saved=np.load(data_dir/"pairs.npz",allow_pickle=False);boot=np.load(data_dir/"bootstrap_plot.npz",allow_pickle=False);spec_data=__import__("json").loads((data_dir/"plot_spec.json").read_text(encoding="utf-8"));out=Path(figures_dir);p=load_protocol();apply_style(p);results=[];masks=saved["masks"];cos={"base":saved["base"],"zh-cn":saved["zh_cn"]};created=[]
    try:
     for figure_id,stem,plotter in (("pilot_ecdf","cosine-distributions",plot_ecdf),("pilot_histogram","cosine-density",plot_histogram)):
        fig,axes=layout(7,p,legend_rows=1);created.append(fig);panels=[]
        for ax,mask in zip(axes,MASK_ORDER):
            series={"base":np.asarray(cos["base"])[masks==mask],"zh_cn":np.asarray(cos["zh-cn"])[masks==mask]};spec=p.figure(figure_id)
            if plotter is plot_histogram:edges=plotter(ax,series,spec,p,edges=np.linspace(-1,1,61))
            else:plotter(ax,series,spec,p);edges=None
            ax.set_title(mask);panels.append({"mask":mask,"counts":{k:len(v) for k,v in series.items()},"edges":edges.tolist() if edges is not None else None,"xlim":list(ax.get_xlim()),"ylim":list(ax.get_ylim())})
        if figure_id=="pilot_histogram":
            ymax=max(ax.get_ylim()[1] for ax in axes)
            for ax in axes:ax.set_ylim(0,ymax)
            for row in panels:row["ylim"]=[0,ymax]
        shared_legend(fig,axes,p);fig._phonetics_resolution={"kind":spec["kind"],"finite_policy":"raise","panels":panels,"shared_x":True,"shared_histogram_y":figure_id=="pilot_histogram","plot_spec":spec_data};results.extend(export_figure(fig,out,stem,figure_id,p,scientific_provenance={"source_digest":source_digest}));plt.close(fig)
     metrics=[("Q_overall9:all","ΔQ","q_values","q_edges"),("D","ΔD","d_values","d_edges")];fig,axes=layout(2,p,legend_rows=0);created.append(fig);panels=[];spec=p.figure("pilot_bootstrap")
     for ax,(metric,title,value_key,edge_key) in zip(axes,metrics):
        raw=boot[value_key];values=raw[np.isfinite(raw)];edges=plot_histogram(ax,{"delta":values},spec,p,edges=boot[edge_key]);ax.axvline(0,color=p.style["color"]["quaternary"],linestyle="--",linewidth=p.style["line"]["reference_pt"]);ax.set_title(title);panels.append({"metric":metric,"count":len(values),"edges":edges.tolist(),"histogram_count":int(np.histogram(values,bins=edges)[0].sum()),"xlim":list(ax.get_xlim()),"ylim":list(ax.get_ylim()),"zero_line":True})
     fig._phonetics_resolution={"kind":"bootstrap_histogram","finite_policy":"drop-recorded","panels":panels,"plot_spec":spec_data};results.extend(export_figure(fig,out,"bootstrap-differences","pilot_bootstrap",p,scientific_provenance={"source_digest":source_digest}));plt.close(fig);return results
    finally:
     for fig in created:plt.close(fig)

def close_all():plt.close("all")
