"""Render reference geometry artifacts from the numerical report."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import re

import numpy as np

from ..evaluation.reference_geometry import SYSTEMS, TRANSITIONS
from .core import DISTANCE_COLORS, apply_style, export_figure, layout, load_protocol
from .reference_geometry_template import PAGE

_PANEL_MODELS = (0, 2, 1)
_REFERENCE_STYLE = """
body{margin:0 auto;max-width:1560px;padding:0 clamp(12px,3vw,40px) 28px;background:#fbfafc;color:#292330;font:15px/1.5 system-ui,sans-serif}
.report-head{padding:24px 0 16px;border:0;border-bottom:1px solid #ded7e6;border-radius:0;background:transparent}
.report-context,.report-count{margin:0;color:#5e5268}
.report-context{font-weight:600}
.report-head h1{margin:2px 0 0;font-size:clamp(27px,3.4vw,39px);line-height:1.12;letter-spacing:-.025em}
.map-title{margin:16px 0 3px;font-size:22px}
.toolbar{padding:6px 0;border:0;border-radius:0;background:transparent}
.panels>section,#map-results>section,#profile-results,details{border-radius:6px}
details{background:#fff;border-color:#ded7e6}
table{font-variant-numeric:tabular-nums}thead th{background:#f4f0f7}
a{color:#563671;text-underline-offset:3px}
:focus-visible{outline:2px solid #79549a;outline-offset:2px}
"""
_REFERENCE_HEADER = """<header class="report-head"><p class="report-context">DIMSIM and wav2vec2 reference report</p>
<h1>Reference Geometry</h1></header>
<h2 class="map-title">Mandarin syllable distance maps</h2>
<p class="report-count">@@syllables@@ syllables · @@pairs@@ unique pairs</p>"""
_REFERENCE_PAGE = re.sub(r'<header class="report-head">.*?</header>', _REFERENCE_HEADER, PAGE, count=1, flags=re.S)
_REFERENCE_PAGE = (_REFERENCE_PAGE
                   .replace("<title>Mandarin syllable distance maps</title>",
                            "<title>DIMSIM and wav2vec2 reference report</title>", 1)
                   .replace("</style>", _REFERENCE_STYLE + "</style>", 1)
                   .replace("<details><summary>Coverage and report files</summary>",
                            "@@same_mask_summary@@<details><summary>Coverage and report files</summary>", 1))


def _same_mask_summary(rows):
    rendered = []
    for row in rows:
        scope = "No shared components (000)" if row["scope"] == "Same mask 000" else row["scope"]
        score = f"{row['agreement']:.2%}" if row["agreement"] is not None else "—"
        rendered.append(
            f"<tr><th scope='row'>{html.escape(row['system'])}</th><td>{html.escape(scope)}</td>"
            f"<td>{row['triples']:,}</td><td>{row['comparisons']:,}</td><td>{score}</td></tr>"
        )
    return (
        '<details id="same-mask-summary" open><summary>Same mask</summary>'
        '<div class="scroll"><table><thead><tr><th scope="col">System</th><th scope="col">Scope</th>'
        '<th scope="col">Triples</th><th scope="col">Comparisons</th><th scope="col">DIMSIM agreement</th>'
        '</tr></thead><tbody>' + "".join(rendered) + '</tbody></table></div>'
        '<p><a href="tables/same_mask_breakdown.csv">Breakdown CSV</a> · '
        '<a href="tables/same_mask_agreement.csv">Per-mask CSV</a></p></details>'
    )


def _encode(values, dtype):
    return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")


def _rank_statistics(distances, neighborhood, tables, tolerance):
    n = distances.shape[1]
    tail, top_k = (n - 1) // 10, min(10, n - 1)
    correlations = {row["system"]: row["spearman"] for row in tables["pairwise_spearman"]}
    overlap_rates, near_far_rates = [], []
    for model in _PANEL_MODELS:
        rankings = np.argsort(distances[model], axis=1, kind="stable")
        overlap = wins = 0
        for anchor, ranking in enumerate(rankings):
            reference = neighborhood[anchor][neighborhood[anchor] != anchor]
            candidates = ranking[ranking != anchor]
            overlap += len(set(reference[:top_k]) & set(candidates[:top_k]))
            if tail:
                near = distances[model, anchor, reference[:tail]]
                far = np.sort(distances[model, anchor, reference[-tail:]])
                lower = np.searchsorted(far, near - tolerance, side="left")
                upper = np.searchsorted(far, near + tolerance, side="right")
                wins += np.sum(tail - upper + .5 * (upper - lower))
        overlap_rates.append(overlap / (n * top_k))
        near_far_rates.append(float(wins / (n * tail * tail)) if tail else None)
    return [overlap_rates, [correlations[SYSTEMS[i]] for i in _PANEL_MODELS], near_far_rates], tail


def _profile_payload(tables):
    profiles, residuals = [], []
    for system in (SYSTEMS[i] for i in _PANEL_MODELS):
        means = [row["mean_similarity"] for row in tables["mask_profiles"] if row["system"] == system]
        r2 = [row["r_squared"] for row in tables["mask_summary"] if row["system"] == system]
        profiles.append({"name": system, "beta": means, "r2": r2})
        residuals.append([{"a": r["left"], "b": r["right"], "mask": r["mask"], "residual": r["residual"]}
                          for r in tables.get("residuals_top50", ()) if r["system"] == system])
    return profiles, residuals


def _map_payload(maps, order, tables):
    inverse = np.argsort(order)
    cells = maps["controlled_cells"].copy()
    values = maps["controlled_values"][:, _PANEL_MODELS]
    cells[:, 1:3] = inverse[cells[:, 1:3]]
    changes = np.bitwise_xor.reduce(TRANSITIONS, axis=1)[cells[:, 0]]
    # The page assigns bits 1/2/4 to I/F/T. The numerical plan uses 4/2/1.
    cells[:, 0] = (changes & 2) | ((changes & 1) << 2) | ((changes & 4) >> 2)
    wins = np.rint(values[:, :, 1] * cells[:, 3, None] * 2)
    agreement = np.rint(np.nan_to_num(values[:, 1:, 2]) * cells[:, 4, None] * 2)
    support = maps["agreement_counts"][:, order][:, :, order]
    projections = maps["agreement"][:, [2, 1]][:, :, order][:, :, :, order]
    points = np.rint(np.nan_to_num(projections) * support[:, None] * 2)
    ties = {row["slice"]: row["dimsim_ties"] for row in tables["dimsim_agreement"] if row["system"] == SYSTEMS[0]}
    return {
        "cells": _encode(cells, "<u2"), "effects": _encode(values[:, :, 0], "<f8"),
        "wins": _encode(wins, "<u2"), "agree": _encode(agreement, "<u2"),
        "support": [_encode(matrix, "<u2") for matrix in support],
        "points": [[_encode(matrix, "<u2") for matrix in group] for group in points],
        "ties": [ties["Trade-off"], ties["Same mask"]],
    }


def render_page(path, labels, components, distances, maps, tables, *, evaluation_id, excluded, tolerance):
    if len(labels) > 32_768:
        raise ValueError("the interactive report exceeds its 16-bit map encoding")
    order = maps["component_order"]
    ordered = distances[np.asarray(_PANEL_MODELS)][:, order][:, :, order]
    data = {
        "labels": np.asarray(labels)[order].tolist(),
        "rank": _encode(np.argsort(order)[maps["neighborhood"][order]], "<u2"),
        "maps": [_encode(matrix, "<f4") for matrix in ordered],
        "maxima": [float(ordered[0].max()), float(ordered[1:].max())],
    }
    profiles, residuals = _profile_payload(tables)
    rank_stats, rank_group_size = _rank_statistics(distances, maps["neighborhood"], tables, tolerance)
    values = {
        "data": data, "components": np.asarray(components)[order].T.tolist(),
        "prepared": _map_payload(maps, order, tables), "profileData": profiles,
        "residualTopPairs": residuals, "rankStats": rank_stats, "rankGroupSize": rank_group_size,
    }
    replacements = {key: json.dumps(value, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
                    for key, value in values.items()}
    replacements.update(
        syllables=f"{len(labels):,}", pairs=f"{len(labels)*(len(labels)-1)//2:,}",
        evaluation_id=html.escape(evaluation_id), excluded=str(len(excluded)), tolerance=f"{tolerance:g}",
        same_mask_summary=_same_mask_summary(tables["same_mask_breakdown"]),
        coverage="".join(f"<tr><th scope='row'>{html.escape(row['population'])}</th><td>{row['count']:,}</td></tr>"
                         for row in tables["coverage"]),
    )
    Path(path).write_text(re.sub(r"@@(\w+)@@", lambda match: replacements[match[1]], _REFERENCE_PAGE), encoding="utf-8")


REFERENCE_MAPS = (
    ("reference_dimsim_distances", "dimsim-distances"),
    ("reference_base_cosine_distances", "base-cosine-distances"),
    ("reference_zh_cn_cosine_distances", "zh-cn-cosine-distances"),
)


def render_distance_maps(root, labels, distances, order, *, provenance, figures=REFERENCE_MAPS, encoder_models=None):
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    protocol = load_protocol()
    apply_style(protocol)
    if encoder_models is not None:
        figures = []
        for model in encoder_models:
            figure_id = f"encoder_{model.replace('-', '_')}_cosine_distances"
            if figure_id not in protocol.registry["figures"]:
                figure_id = "encoder_cosine_distances"
            figures.append((figure_id, f"{model}-cosine-distances"))
    specs = [protocol.figure(figure_id) for figure_id, _ in figures]
    maxima = {}
    for spec, matrix in zip(specs, distances):
        group = spec["color_scale_group"]
        maxima[group] = max(maxima.get(group, 0), float(np.nanmax(matrix)))
    for index, ((figure_id, stem), spec) in enumerate(zip(figures, specs)):
        figure, (axis,) = layout(1, protocol)
        try:
            limit = maxima[spec["color_scale_group"]] or 1.0
            cmap = (LinearSegmentedColormap.from_list("lilac", DISTANCE_COLORS)
                    if spec["colormap"] == "lilac" else plt.get_cmap(spec["colormap"]))
            cmap = cmap.with_extremes(bad="#bfbfbf")
            image = axis.imshow(distances[index][np.ix_(order, order)], origin="upper",
                                interpolation="nearest", cmap=cmap, vmin=0, vmax=limit)
            axis.set(xlabel=spec["xlabel"], ylabel=spec["ylabel"], xticks=[], yticks=[])
            for label in axis.texts:
                label.set_visible(False)
            colorbar = figure.colorbar(image, ax=axis, pad=.04, fraction=.20,
                                       anchor=(0, .5), ticks=MaxNLocator(4))
            colorbar.set_label(spec["colorbar_label"])
            figure._phonetics_resolution = {
                "ordered_labels": np.asarray(labels)[order].tolist(),
                "color_limits": [0.0, limit], "axis_tick_labels": False,
            }
            export_figure(figure, root, stem, figure_id, protocol, scientific_provenance=provenance)
        finally:
            plt.close(figure)
