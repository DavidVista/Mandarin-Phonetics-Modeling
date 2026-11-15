"""Render the three approved DIMSIM distribution figure families."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
import numpy as np
from . import apply_style, export_figure, layout, load_protocol, shared_legend

from .core import COMPONENT_FILLS, DISTANCE_COLORS

MASK_ORDER = tuple(f"{i:03b}" for i in range(8))

def render_distribution(data_dir, figures_dir, source_digest):
    data_dir, figures_dir = Path(data_dir), Path(figures_dir)
    saved = np.load(data_dir / "distribution_plot_data.npz", allow_pickle=False)
    rows = json.loads((data_dir / "component_shares.json").read_text(encoding="utf-8"))
    protocol = load_protocol(); apply_style(protocol); results=[]; created=[]
    try:
        spec=protocol.figure("dimsim_distributions"); fig,axes=layout(3,protocol); created.append(fig); d=saved["distance"]
        axes[0].hist(d,bins=saved["full_histogram_edges"],color=DISTANCE_COLORS[1],edgecolor="white",linewidth=.4); axes[0].set(title="Full range",xlabel=spec["xlabel"],ylabel="Pair count")
        x=np.sort(d,kind="mergesort"); axes[1].step(x,np.arange(1,len(x)+1)/len(x),where="post",color=protocol.registry["semantics"]["population"]["color"]); axes[1].set(title="ECDF",xlabel=spec["xlabel"],ylabel="ECDF",ylim=(0,1))
        right = saved["full_histogram_edges"][-1]
        for ax in axes[:2]:
            ax.set_xlim(0, right)
        axes[0].yaxis.set_major_formatter(EngFormatter(sep=""))
        axes[2].hist(saved["detail_values"],bins=50,color=DISTANCE_COLORS[1],edgecolor="white",linewidth=.4); axes[2].set(title="Detail: D < 10",xlabel=spec["xlabel"],ylabel="Pair count",xlim=(0,10))
        fig._phonetics_resolution={"panels":["full_histogram","ecdf","detail_lt_10"],"counts":[len(d),len(d),len(saved["detail_values"])],"full_edges":saved["full_histogram_edges"].tolist(),"detail_bins":50}
        results.extend(export_figure(fig,figures_dir,"dimsim-distributions","dimsim_distributions",protocol,scientific_provenance={"source_digest":source_digest})); plt.close(fig)
        spec = protocol.figure("dimsim_overlap_distributions")
        series = {mask: d[saved["overlap_mask"] == mask] for mask in MASK_ORDER}
        groups = {
            count: [mask for mask, values in series.items() if mask.count("1") == count and len(values)]
            for count in range(4)
        }
        groups = {count: masks for count, masks in groups.items() if masks}
        fig, axes = layout(len(groups), protocol, width_mm=protocol.style["layout"]["multi_width_mm"])
        created.append(fig)
        for ax, (count, masks) in zip(axes, groups.items()):
            for mask in masks:
                x, frequencies = np.unique(series[mask], return_counts=True)
                y = frequencies.cumsum() / len(series[mask])
                sem = protocol.registry["semantics"][mask]
                ax.step(np.r_[0, x, right], np.r_[0, y, 1], where="post",
                        color=sem["color"], linestyle=sem["linestyle"], label=sem["label"],
                        linewidth=protocol.style["line"]["data_pt"])
            ax.set(title="1 match" if count == 1 else f"{count} matches",
                   xlabel=spec["xlabel"], ylabel=spec["ylabel"], xlim=(0, right), ylim=spec["ylim"])
            ax.legend(loc="lower right", framealpha=1)
        fig._phonetics_resolution = {
            "mask_order": list(MASK_ORDER),
            "counts": {mask: len(values) for mask, values in series.items()},
            "panels": [{"matches": count, "masks": masks} for count, masks in groups.items()],
        }
        results.extend(export_figure(fig,figures_dir,"component-overlap-distributions","dimsim_overlap_distributions",protocol,scientific_provenance={"source_digest":source_digest})); plt.close(fig)
        spec=protocol.figure("dimsim_component_contributions"); fig,axes=layout(1,protocol,legend_rows=1,width_mm=protocol.style["layout"]["multi_width_mm"]); created.append(fig); ax=axes[0]; labels=[f"{10*i}\u2013{10*(i+1)}" for i in range(12)]+["\u2265120"]; bottom=np.zeros(len(rows))
        for sid, color in COMPONENT_FILLS.items():
            values = np.asarray([np.nan if row[f"{sid}_share"] is None else row[f"{sid}_share"]
                                 for row in rows], float)
            ax.bar(np.arange(len(rows)), values, bottom=bottom, width=.82,
                   label=protocol.registry["semantics"][sid]["label"], color=color,
                   edgecolor="white", linewidth=.45)
            bottom = np.where(np.isfinite(values), bottom + values, bottom)
        ax.set(xticks=np.arange(len(rows)), xlabel=spec["xlabel"], ylabel=spec["ylabel"], ylim=(0,1))
        ax.set_xticklabels(labels, fontsize=protocol.style["font"]["tick_pt"] - 1)
        shared_legend(fig, axes, protocol)
        fig._phonetics_resolution = {
            "bins": [r["bin"] for r in rows], "display_labels": labels, "counts": [r["count"] for r in rows],
            "undefined_bins": [r["bin"] for r in rows if r["initial_share"] is None],
        }
        results.extend(export_figure(fig,figures_dir,"dimsim-component-contributions","dimsim_component_contributions",protocol,scientific_provenance={"source_digest":source_digest})); plt.close(fig); return results
    finally:
        for fig in created: plt.close(fig)
