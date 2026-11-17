"""Residual distributions and the character composition of their tails."""

from collections import Counter, defaultdict

import numpy as np

from ..objectives import BASE_MODEL_IDS, parse_model_id
from .core import apply_style, export_figure, load_protocol
from .reference_geometry_template import SIGNED_COLORS

FIGURE_ID = "encoder_residual_histograms"
STEM = "residual-histograms-by-mask"


def residual_histogram_seed(distances, masks, labels, occurrences, edges):
    """Summarize a seed without retaining its distance matrices or hidden states."""
    label_index = {label: i for i, label in enumerate(labels)}
    characters = [Counter() for _ in labels]
    for row in occurrences:
        characters[label_index[row["pinyin"]]][row["character"]] += 1
    shared = np.zeros(masks.shape, dtype=bool)
    readings = defaultdict(list)
    for i, counts in enumerate(characters):
        for char in counts:
            readings[char].append(i)
    for indices in readings.values():
        shared[np.ix_(indices, indices)] = True
    dominant = np.array([2 * max(counts.values(), default=0) > counts.total() for counts in characters])
    single = np.array([len(counts) == 1 for counts in characters])
    left, right = np.triu_indices(len(labels), 1)
    pair_masks = masks[left, right]
    bins = len(edges) - 1
    histograms = np.zeros((len(distances), 8, bins))
    tails = np.zeros((len(distances), 7, 2, 2))
    modes = np.zeros((len(distances), 2, bins))
    for mask in range(8):
        if mask < 7:
            selected = pair_masks == mask
            a, b = left[selected], right[selected]
            properties = np.stack((shared[a, b], dominant[a] & dominant[b]))
        for model, matrix in enumerate(distances):
            if mask == 7:
                raw = matrix.diagonal()
                valid = np.isfinite(raw)
                values = raw[valid]
            else:
                values = matrix[a, b]
            if not len(values):
                continue
            residuals = values.mean() - values
            counts = np.histogram(residuals, edges)[0]
            if counts.sum() != len(values):
                raise ValueError("residual histogram edges exclude observations")
            histograms[model, mask] = counts / len(values)
            if mask == 7:
                one = single[valid]
                for group, selection in enumerate((~one, one)):
                    modes[model, group] = np.histogram(residuals[selection], edges)[0] / len(values)
                continue
            size = (len(values) + 19) // 20
            low, high = np.partition(residuals, (size - 1, len(values) - size))[[size - 1, len(values) - size]]
            for side, selection in enumerate((residuals <= low, residuals >= high)):
                tails[model, mask, side] = properties[:, selection].mean(axis=1)
    return histograms, tails, modes


def _model_order(model):
    family, weight = parse_model_id(model)
    return (BASE_MODEL_IDS.index(family), 0) if weight is None else (len(BASE_MODEL_IDS), weight)


def render_residual_histograms(output_dir, models, summaries, *, protocol, provenance):
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    histograms, tails, modes = (np.mean(part, axis=0) for part in zip(*summaries))
    order = sorted(range(len(models)), key=lambda index: _model_order(models[index]))
    spec = protocol.figure(FIGURE_ID)
    edges = np.linspace(spec["edges_start"], spec["edges_stop"], spec["edges_count"])
    centers = (edges[:-1] + edges[1:]) / 2
    occupied = np.flatnonzero(histograms.any(axis=(0, 1)))
    xlim = min(-.65, edges[occupied[0]]), max(1.05, edges[occupied[-1] + 1])
    ylim = 0, max(.29, float(histograms.max()) + .025)
    style = protocol.style
    ink, muted = (style["color"][role] for role in ("foreground", "secondary"))
    red, _, purple = SIGNED_COLORS
    rows = len(order)
    with plt.rc_context():
        apply_style(protocol)
        fig, axes = plt.subplots(rows, 8, figsize=(16.5, max(3.6, 1.55 * rows + 1.5)),
                                 sharex=True, sharey=True, squeeze=False,
                                 gridspec_kw={"width_ratios": [1] * 7 + [1.33]})
        try:
            top = .86 if rows == 1 else .89
            gap = .94
            # Reserve room below the last row for the tick labels, tail shares and legend.
            tail_offset = .57 / (rows + (rows - 1) * gap)
            bottom = (.165 + tail_offset * top) / (1 + tail_offset)
            fig.subplots_adjust(left=.112, right=.984, top=top, bottom=bottom, wspace=.26, hspace=gap)
            for row, model in enumerate(order):
                family, weight = parse_model_id(models[model])
                label = family if weight is None else f"λ = {weight:g}"
                for mask, ax in enumerate(axes[row]):
                    h = histograms[model, mask]
                    ax.stairs(h, edges, color=ink)
                    ax.axvline(0, color=muted, lw=style["line"]["reference_pt"], ls="--", zorder=0)
                    ax.set(xlim=xlim, ylim=ylim)
                    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
                    ax.locator_params(axis="y", nbins=3)
                    ax.spines[["top", "right"]].set_visible(False)
                    if mask:
                        ax.tick_params(axis="y", left=False)
                    if row == 0:
                        ax.set_title(f"{mask:03b}", fontweight="bold")
                    if mask == 0:
                        ax.text(-.47, .5, label, transform=ax.transAxes, ha="right", va="center", fontweight="bold")
                    if mask == 7 and row == 0:
                        for group, text, x in ((0, "Multiple\ncharacters", .2), (1, "One\ncharacter", .8)):
                            peak = modes[model, group]
                            if peak.any():
                                i = int(np.argmax(peak))
                                ax.annotate(text, (centers[i], h[i]), xytext=(x, 1.03),
                                            textcoords="axes fraction", ha="center", va="bottom",
                                            fontsize=7, arrowprops={"arrowstyle": "-", "color": muted, "lw": .65})
                    if mask == 7 or not h.any():
                        continue
                    for side, x in enumerate((.24, .76)):
                        for group, color in enumerate((purple, red)):
                            value = 100 * tails[model, mask, side, group]
                            text = f"{value:.2f}" if 0 < value < 1 else f"{value:.1f}".removesuffix(".0")
                            ax.text(x, -(.38 if row == rows - 1 else .23) - .19 * group, text + "%",
                                    transform=ax.transAxes, ha="center", va="center", color=color, fontweight="bold")
            fig.suptitle(spec["name"], y=.98, fontsize=15, fontweight="bold")
            subtitle = f"Mean across {len(summaries)} seeds" if len(summaries) > 1 else "One seed"
            fig.text(.5, .944, subtitle, ha="center", color=muted)
            fig.text(.016, .53, spec["ylabel"], ha="center", va="center", rotation=90)
            fig.text(.55, .126, spec["xlabel"], ha="center")
            fig.text(.55, .10, "Left = lower 5%   ·   Right = upper 5%   ·   Cutoff ties included",
                     ha="center", color=muted)
            fig.text(.55, .068, "Share of pairs with a common character", ha="center", color=purple)
            fig.text(.55, .036, "Share with a dominant character in both syllables (>50% each)", ha="center", color=red)
            fig._phonetics_resolution = {"models": [models[i] for i in order], "mask_order": [f"{i:03b}" for i in range(8)]}
            return export_figure(fig, output_dir, STEM, FIGURE_ID, protocol, scientific_provenance=provenance)
        finally:
            plt.close(fig)
