"""Static coefficient profiles and row-wise DIMSIM neighbourhoods."""
import numpy as np

from .core import apply_style, export_figure, load_protocol
from .encoder_geometry import _system_name
from .reference_geometry_template import DISTANCE_COLORS


def render_geometry_figures(output, labels, distances, maps, summary, *, systems, provenance, architecture):
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    protocol = load_protocol()
    profiles = {system: sorted((r for r in summary["mask_profiles"] if r["system"] == system),
                               key=lambda r: r["mask"]) for system in systems}
    order = sorted(range(8), key=lambda i: (profiles["DIMSIM"][i]["mean_similarity_mean"]
                    if profiles["DIMSIM"][i]["mean_similarity_mean"] is not None else float("inf"), i))
    with plt.rc_context():
        apply_style(protocol)
        fig, left = plt.subplots(figsize=(10, 5.5))
        right = left.twinx()
        try:
            left.axvspan(order.index(7)-.2, order.index(7)+.2, color="#f0edf5", zorder=0)
            for i, system in enumerate(systems):
                rows = [profiles[system][j] for j in order]
                axis = left if system == "DIMSIM" else right
                mean = [r["mean_similarity_mean"] if r["mean_similarity_mean"] is not None else np.nan for r in rows]
                sd = [r["mean_similarity_sd"] or 0 for r in rows]
                axis.errorbar(range(8), mean, yerr=sd, marker="o", capsize=3,
                              color="black" if i == 0 else plt.get_cmap("tab10")((i-1) % 10),
                              linestyle="--" if i == 0 else "-", label=_system_name(system))
            left.set(xticks=range(8), xticklabels=[f"{i:03b}" for i in order],
                     xlabel="Overlap mask", ylabel="Negative DIMSIM distance")
            right.set_ylabel("Mean cosine similarity")
            handles, names = left.get_legend_handles_labels()
            h, n = right.get_legend_handles_labels()
            fig.suptitle(f"{architecture} · Mask coefficients", y=.99)
            fig.legend(handles+h, names+n, loc="upper center", bbox_to_anchor=(.5,.94), ncol=3, frameon=False)
            fig.subplots_adjust(top=.78, bottom=.15, right=.9)
            fig._phonetics_resolution = {"mask_order": [f"{i:03b}" for i in order],
                                        "interval": "sample SD across training seeds",
                                        "deterministic_sources_without_sd": [s for s in systems if s == "pretrained-bert"]}
            export_figure(fig, output, "mask-coefficients", "encoder_mask_coefficients", protocol,
                          scientific_provenance=provenance)
        finally:
            plt.close(fig)
        # Each anchor has its own candidate ordering. These are not symmetric heatmaps.
        anchors = maps["component_order"]
        candidates = maps["neighborhood"][anchors]
        candidates = np.asarray([row[row != anchor] for anchor, row in zip(anchors, candidates)])
        shared_max = float(np.nanmax(distances[1:])) or 1
        for i, system in enumerate(systems):
            values = distances[i][anchors[:, None], candidates]
            fig, axis = plt.subplots(figsize=(7, 6))
            try:
                limit = (float(np.nanmax(distances[0])) or 1) if i == 0 else shared_max
                cmap = LinearSegmentedColormap.from_list("distance", DISTANCE_COLORS)
                view = axis.imshow(values, aspect="auto", interpolation="nearest", vmin=0, vmax=limit, cmap=cmap)
                axis.set(xlabel="Candidate rank by DIMSIM", ylabel="Anchor", xticks=[], yticks=[],
                         title=f"{architecture} · {_system_name(system)}")
                fig.colorbar(view, ax=axis, label="DIMSIM distance" if i == 0 else "Cosine distance")
                fig.tight_layout()
                fig._phonetics_resolution = {"anchor_labels": [labels[j] for j in anchors],
                    "candidate_indices_by_anchor": candidates.tolist(), "color_limits": [0, limit],
                    "self_excluded": True, "candidate_order": "ascending DIMSIM distance; stable label ties"}
                export_figure(fig, output, f"{system.lower()}-dimsim-neighborhood", "encoder_dimsim_neighborhood",
                              protocol, scientific_provenance=provenance)
            finally:
                plt.close(fig)
