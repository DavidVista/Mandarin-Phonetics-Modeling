"""Render reference geometry artifacts from the numerical report."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path
import re

import numpy as np

from ..evaluation.reference_geometry import SYSTEMS, TRANSITIONS
from .core import apply_style, export_figure, layout, load_protocol
from .reference_geometry_template import PAGE

_PANEL_MODELS = (0, 2, 1)


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
        steps = []
        gains = (row for row in tables["full_match_gains"] if row["system"] == system)
        for row, final in zip(gains, ("011 → 111", "101 → 111", "110 → 111")):
            steps.append({
                "labels": [row[f"step_{i}"].replace(" to ", " → ") for i in range(1, 4)] + [final],
                "values": [row[f"gain_{i}"] for i in range(1, 4)] + [row["full_match_gain"]],
                "baseline": row["ordinary_mean"], "absolute": row["j"],
            })
        profiles.append({"name": system, "beta": means, "r2": r2, "steps": steps})
        residuals.append([{"a": row["left"], "b": row["right"], "mask": row["mask"], "residual": row["residual"]}
                          for row in tables["residuals_top50"] if row["system"] == system])
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
        coverage="".join(f"<tr><th scope='row'>{html.escape(row['population'])}</th><td>{row['count']:,}</td></tr>"
                         for row in tables["coverage"]),
    )
    Path(path).write_text(re.sub(r"@@(\w+)@@", lambda match: replacements[match[1]], PAGE), encoding="utf-8")


def render_distance_maps(root, labels, distances, order, *, provenance):
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    protocol = load_protocol()
    apply_style(protocol)
    figures = (
        ("reference_dimsim_distances", "dimsim-distances"),
        ("reference_base_cosine_distances", "base-cosine-distances"),
        ("reference_zh_cn_cosine_distances", "zh-cn-cosine-distances"),
    )
    specs = [protocol.figure(figure_id) for figure_id, _ in figures]
    maxima = {}
    for spec, matrix in zip(specs, distances):
        group = spec["color_scale_group"]
        maxima[group] = max(maxima.get(group, 0), float(np.max(matrix)))
    for index, ((figure_id, stem), spec) in enumerate(zip(figures, specs)):
        figure, (axis,) = layout(1, protocol)
        try:
            limit = maxima[spec["color_scale_group"]] or 1.0
            image = axis.imshow(distances[index][np.ix_(order, order)], origin="upper",
                                interpolation="nearest", cmap=spec["colormap"], vmin=0, vmax=limit)
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
