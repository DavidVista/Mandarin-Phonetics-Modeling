"""Shared geometry map layout and stable system colours."""
from ..objectives import parse_model_id


def system_color(system):
    fixed = {"DIMSIM": "#292330", "pretrained-bert": "#7f7f7f",
             "syllable": "#1f77b4", "components": "#ff7f0e", "acoustic": "#2ca02c",
             "components-acoustic-lambda-1": "#d62728",
             "components-acoustic-lambda-3": "#9467bd",
             "components-acoustic-lambda-10": "#8c564b",
             "syllable-acoustic-lambda-1": "#e377c2"}
    if system in fixed:
        return fixed[system]
    family, _ = parse_model_id(system)
    return "#9467bd" if family == "components-acoustic" else "#e377c2"


def geometry_map(values, *, limit, cmap, xlabel, colorbar_label, title):
    """Use identical physical map/colorbar bounds for both matrix orderings."""
    import numpy as np
    from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_axes([.12, .14, .68, .68 * 7 / 6])
    color_axis = figure.add_axes([.83, .14, .035, .68 * 7 / 6])
    view = axis.imshow(values, origin="upper", aspect="auto", interpolation="nearest",
                       cmap=cmap, vmin=0, vmax=limit)
    axis.set(xlabel=xlabel, ylabel="Anchor", xticks=[], yticks=[])
    axis.set_title(title, fontsize=10, pad=10)
    bar = figure.colorbar(view, cax=color_axis, ticks=np.linspace(0, limit, 5))
    bar.set_label(colorbar_label)
    return figure, axis


def map_layout(figure):
    return {"figure_inches": figure.get_size_inches().tolist(),
            "map_bounds": list(figure.axes[0].get_position().bounds),
            "colorbar_bounds": list(figure.axes[1].get_position().bounds),
            "colorbar_ticks": figure.axes[1].get_yticks().tolist()}
