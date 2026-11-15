"""Plot objective histories for the models and seeds in a training report."""

VALIDATION_FIGURES = {
    "validation-accuracy": ("Accuracy", {"syllable_accuracy": "Syllable", "joint_ift_accuracy": "Joint components"}),
    "validation-loss-components": ("Loss", {"component_ce": "Component CE", "cosine": "Cosine distance"}),
    "component-accuracy": ("Accuracy", {"initials_accuracy": "Initial", "finals_accuracy": "Final", "tones_accuracy": "Tone"}),
    "invalid-initial-final-combinations": ("Share", {"invalid_if_rate": "Invalid combinations"}),
}


def render_training_curves(histories, destination, selected_epochs):
    from matplotlib import rc_context

    from .core import apply_style, load_protocol

    protocol = load_protocol()
    with rc_context():
        apply_style(protocol)
        return _render_training_curves(histories, destination, selected_epochs, protocol)


def _render_training_curves(histories, destination, selected_epochs, protocol):
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    semantics = protocol.registry["semantics"]
    colors = [semantics[key]["color"] for key in ("syllable", "components", "acoustic")]

    def panel_figure(count, columns):
        rows = (count + columns - 1) // columns
        figure = Figure(figsize=(4 * columns, 2.5 * rows), layout="constrained")
        FigureCanvasAgg(figure)
        panels = figure.subplots(rows, columns, squeeze=False).ravel()
        for axes in panels[count:]:
            axes.remove()
        for axes in panels[:count]:
            axes.set_prop_cycle(color=colors)
        return figure, panels[:count]

    def draw(axes, model, split, metric, label=None):
        seeds = histories[model]
        epochs = grids[model]
        values = np.array([[epoch[split][metric] for epoch in history] for history in seeds.values()], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite {metric} history for {model}")
        mean = values.mean(axis=0)
        line, = axes.plot(epochs, mean, label=label, marker="." if len(epochs) == 1 else None,
                          linestyle="--" if split == "train" else "-")
        if len(seeds) > 1:
            sd = values.std(axis=0, ddof=1)
            axes.fill_between(epochs, mean - sd, mean + sd, color=line.get_color(), alpha=0.2)
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        return line

    grids = {}
    for model, seeds in histories.items():
        epochs = [epoch["epoch"] for epoch in next(iter(seeds.values()))]
        if not epochs or epochs != list(range(1, len(epochs) + 1)):
            raise ValueError(f"invalid epoch grid for {model}")
        if any([epoch["epoch"] for epoch in history] != epochs for history in seeds.values()):
            raise ValueError(f"seed histories have different epoch grids for {model}")
        grids[model] = epochs

    figure, panels = panel_figure(2 * len(histories), 2)
    for index, (model, seeds) in enumerate(histories.items()):
        for axes, split in zip(panels[2 * index:2 * index + 2], ("train", "validation")):
            line = draw(axes, model, split, "objective")
            if split == "validation" and selected_epochs[model]:
                selected = selected_epochs[model]
                if any(epoch not in grids[model] for epoch in selected.values()):
                    raise ValueError(f"selected epoch falls outside the history for {model}")
                axes.scatter(list(selected.values()),
                             [seeds[seed][epoch - 1][split]["objective"] for seed, epoch in selected.items()],
                             color=line.get_color(), facecolors="none", label="Selected epoch")
                axes.legend()
            axes.set(title=f"{model} · {split}", xlabel="Epoch", ylabel="Objective")
    figure.savefig(destination, dpi=150)
    outputs = [destination]

    for name, (ylabel, labels) in VALIDATION_FIGURES.items():
        available = {model: {key: label for key, label in labels.items()
                             if all(key in epoch["validation"] for history in seeds.values() for epoch in history)}
                     for model, seeds in histories.items()}
        applicable = [model for model, metrics in available.items() if metrics]
        if not applicable:
            continue
        figure, panels = panel_figure(len(applicable), min(2, len(applicable)))
        for axes, model in zip(panels, applicable):
            for metric, label in available[model].items():
                draw(axes, model, "validation", metric, label)
            axes.set(title=model, xlabel="Epoch", ylabel=ylabel)
            axes.legend()
        output = destination.with_name(name + ".png")
        figure.savefig(output, dpi=150)
        outputs.append(output)
    return outputs
