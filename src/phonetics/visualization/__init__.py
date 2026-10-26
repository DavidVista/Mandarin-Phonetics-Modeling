"""Project-wide scientific visualization protocol."""
from .core import (Protocol, VisualizationError, apply_style, create_snapshot, ecdf, export_figure, layout, load_protocol, plot_ecdf, plot_histogram, pooled_edges, require_arial, shared_legend, verify_snapshot)

__all__ = ["Protocol", "VisualizationError", "apply_style", "create_snapshot", "ecdf", "export_figure", "layout", "load_protocol", "plot_ecdf", "plot_histogram", "pooled_edges", "require_arial", "shared_legend", "verify_snapshot"]
