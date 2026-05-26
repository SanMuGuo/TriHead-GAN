"""Visualization utilities for time series generative model evaluation.

Provides t-SNE plots, time series comparisons, training curves,
and ACF comparison charts.
"""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

# Use a clean style similar to seaborn-whitegrid
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        logger.debug("Seaborn style not found, using default.")


def _to_numpy(data: np.ndarray) -> np.ndarray:
    """Ensure data is a numpy array."""
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.asarray(data, dtype=np.float32)


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to path or display interactively."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", save_path)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def plot_tsne(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """t-SNE visualization of real vs fake time series.

    Flatten each window to a vector, reduce to 2-D with t-SNE,
    then create a scatter plot (blue = real, red = fake).

    Args:
        real_data: Array of shape (n, seq_len, n_features).
        fake_data: Array of shape (n, seq_len, n_features).
        save_path: If provided, save figure to this path.
    """
    real_data = _to_numpy(real_data)
    fake_data = _to_numpy(fake_data)

    n_real = real_data.shape[0]
    n_fake = fake_data.shape[0]

    X_real = real_data.reshape(n_real, -1)
    X_fake = fake_data.reshape(n_fake, -1)
    X_all = np.concatenate([X_real, X_fake], axis=0)

    # Subsample for speed if dataset is large
    max_samples = 2000
    if X_all.shape[0] > max_samples:
        idx = np.random.choice(X_all.shape[0], max_samples, replace=False)
        X_all = X_all[idx]
        labels = np.concatenate([
            np.ones(n_real), np.zeros(n_fake)
        ])[idx]
    else:
        labels = np.concatenate([np.ones(n_real), np.zeros(n_fake)])

    logger.info("Running t-SNE on %d samples...", X_all.shape[0])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings = tsne.fit_transform(X_all)

    real_mask = labels == 1
    fake_mask = labels == 0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        embeddings[real_mask, 0], embeddings[real_mask, 1],
        c="royalblue", alpha=0.5, s=15, label="Real",
    )
    ax.scatter(
        embeddings[fake_mask, 0], embeddings[fake_mask, 1],
        c="crimson", alpha=0.5, s=15, label="Fake",
    )
    ax.set_title("t-SNE: Real vs Fake")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend()

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Time series comparison
# ---------------------------------------------------------------------------

def plot_time_series_comparison(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    n_samples: int = 5,
    feature_idx: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side comparison of real and fake time series.

    Plot n random real samples and n random fake samples for a
    given feature index.

    Args:
        real_data: Array of shape (n, seq_len, n_features).
        fake_data: Array of shape (n, seq_len, n_features).
        n_samples: Number of samples to plot per group.
        feature_idx: Which feature dimension to visualize.
        save_path: If provided, save figure to this path.
    """
    real_data = _to_numpy(real_data)
    fake_data = _to_numpy(fake_data)

    real_idx = np.random.choice(real_data.shape[0], n_samples, replace=False)
    fake_idx = np.random.choice(fake_data.shape[0], n_samples, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for i, idx in enumerate(real_idx):
        axes[0].plot(real_data[idx, :, feature_idx], alpha=0.7,
                     label=f"Sample {i}" if i < 3 else None)
    axes[0].set_title("Real Data")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel(f"Feature {feature_idx}")
    axes[0].legend(loc="upper right")

    for i, idx in enumerate(fake_idx):
        axes[1].plot(fake_data[idx, :, feature_idx], alpha=0.7,
                     label=f"Sample {i}" if i < 3 else None)
    axes[1].set_title("Fake Data")
    axes[1].set_xlabel("Time Step")
    axes[1].legend(loc="upper right")

    fig.suptitle(f"Time Series Comparison (Feature {feature_idx})")
    fig.tight_layout()

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training loss curves from history dict.

    Expected keys: 'd_loss', 'g_loss', 'w_distance'.
    Missing keys are silently skipped.

    Args:
        history: Dict mapping metric names to lists of values.
        save_path: If provided, save figure to this path.
    """
    plot_keys = [k for k in ["d_loss", "g_loss", "w_distance"] if k in history]
    if not plot_keys:
        logger.warning("No recognized keys in history dict. Nothing to plot.")
        return

    n_plots = len(plot_keys)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    titles = {
        "d_loss": "Discriminator Loss",
        "g_loss": "Generator Loss",
        "w_distance": "Wasserstein Distance",
    }
    colors = {
        "d_loss": "steelblue",
        "g_loss": "coral",
        "w_distance": "seagreen",
    }

    for ax, key in zip(axes, plot_keys):
        values = history[key]
        ax.plot(values, color=colors.get(key, "gray"), linewidth=1.2)
        ax.set_title(titles.get(key, key))
        ax.set_xlabel("Iteration")
        ax.set_ylabel(key)

    fig.suptitle("Training Curves")
    fig.tight_layout()

    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# ACF comparison
# ---------------------------------------------------------------------------

def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation for a 1-D series using numpy."""
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var < 1e-10:
        return np.zeros(max_lag + 1)
    acf_vals = np.correlate(x_centered, x_centered, mode="full")
    acf_vals = acf_vals[n - 1:]
    acf_vals = acf_vals[: max_lag + 1] / (var * n)
    return acf_vals


def plot_acf_comparison(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    feature_idx: int = 0,
    max_lag: int = 24,
    save_path: Optional[str] = None,
) -> None:
    """Compare autocorrelation functions of real vs fake data.

    Compute mean ACF across samples for a specific feature,
    then overlay real and fake ACF curves.

    Args:
        real_data: Array of shape (n, seq_len, n_features).
        fake_data: Array of shape (n, seq_len, n_features).
        feature_idx: Which feature to compute ACF for.
        max_lag: Maximum lag for ACF.
        save_path: If provided, save figure to this path.
    """
    real_data = _to_numpy(real_data)
    fake_data = _to_numpy(fake_data)
    max_lag = min(max_lag, real_data.shape[1] - 1)

    acf_real = np.mean(
        [_compute_acf(real_data[i, :, feature_idx], max_lag)
         for i in range(real_data.shape[0])],
        axis=0,
    )
    acf_fake = np.mean(
        [_compute_acf(fake_data[i, :, feature_idx], max_lag)
         for i in range(fake_data.shape[0])],
        axis=0,
    )

    lags = np.arange(max_lag + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lags, acf_real, "o-", color="royalblue", markersize=4,
            label="Real", linewidth=1.5)
    ax.plot(lags, acf_fake, "s--", color="crimson", markersize=4,
            label="Fake", linewidth=1.5)
    ax.set_title(f"ACF Comparison (Feature {feature_idx})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    fig.tight_layout()
    _save_or_show(fig, save_path)
