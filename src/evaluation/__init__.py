"""Evaluation module for time series generative models.

Exports evaluation metrics and visualization utilities.
"""

from .metrics import (
    acf_difference,
    compute_fid,
    compute_mmd,
    discriminative_score,
    evaluate_all,
    predictive_score,
)
from .visualization import (
    plot_acf_comparison,
    plot_time_series_comparison,
    plot_training_curves,
    plot_tsne,
)

__all__ = [
    # Metrics
    "discriminative_score",
    "predictive_score",
    "compute_mmd",
    "compute_fid",
    "acf_difference",
    "evaluate_all",
    # Visualization
    "plot_tsne",
    "plot_time_series_comparison",
    "plot_training_curves",
    "plot_acf_comparison",
]
