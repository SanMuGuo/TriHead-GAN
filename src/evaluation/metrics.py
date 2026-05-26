"""Evaluation metrics for time series generative models.

Implements discriminative score, predictive score, MMD, FID,
and ACF difference to assess quality of generated time series.
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------

class _LSTMClassifier(nn.Module):
    """Simple 2-layer LSTM for discriminative score."""

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        logits = self.fc(out[:, -1, :])
        return logits.squeeze(-1)


class _GRUPredictor(nn.Module):
    """Simple 2-layer GRU for predictive score."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        pred = self.fc(out[:, -1, :])
        return pred


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_numpy(data: np.ndarray) -> np.ndarray:
    """Ensure data is a numpy array of float32."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return np.asarray(data, dtype=np.float32)


def _get_device() -> torch.device:
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Discriminative score
# ---------------------------------------------------------------------------

def discriminative_score(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    n_splits: int = 5,
) -> float:
    """Post-hoc LSTM classifier to distinguish real vs fake.

    Train a 2-layer LSTM on mixed real+fake data with k-fold CV.

    Args:
        real_data: Array of shape (n_real, seq_len, n_features).
        fake_data: Array of shape (n_fake, seq_len, n_features).
        n_splits: Number of cross-validation folds.

    Returns:
        Discriminative score = |mean_accuracy - 0.5|. Lower is better.
    """
    real_data = _to_numpy(real_data)
    fake_data = _to_numpy(fake_data)

    n_real = real_data.shape[0]
    n_fake = fake_data.shape[0]
    n_features = real_data.shape[2]

    device = _get_device()
    # IMPORTANT: real windows are produced by a sliding window with stride
    # < window_size, so adjacent windows overlap heavily.  A plain
    # ``KFold(shuffle=True)`` would scatter overlapping windows across the
    # train and validation folds and leak information, inflating
    # classifier accuracy.  Instead, split each class chronologically
    # (shuffle=False) and concatenate. At most one pair of boundary
    # windows can overlap per fold, which is negligible.
    real_kf = KFold(n_splits=n_splits, shuffle=False)
    fake_kf = KFold(n_splits=n_splits, shuffle=False)
    accuracies = []

    real_splits = list(real_kf.split(np.arange(n_real)))
    fake_splits = list(fake_kf.split(np.arange(n_fake)))

    for fold, ((r_train, r_val), (f_train, f_val)) in enumerate(
        zip(real_splits, fake_splits)
    ):
        X_train_np = np.concatenate(
            [real_data[r_train], fake_data[f_train]], axis=0
        )
        y_train_np = np.concatenate(
            [np.ones(len(r_train)), np.zeros(len(f_train))], axis=0
        )
        X_val_np = np.concatenate(
            [real_data[r_val], fake_data[f_val]], axis=0
        )
        y_val_np = np.concatenate(
            [np.ones(len(r_val)), np.zeros(len(f_val))], axis=0
        )
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)
        X_val = torch.tensor(X_val_np, dtype=torch.float32)
        y_val = torch.tensor(y_val_np, dtype=torch.float32)

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        model = _LSTMClassifier(n_features).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(20):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val.to(device))
            preds = (torch.sigmoid(logits) > 0.5).float().cpu()
            acc = (preds == y_val).float().mean().item()
        accuracies.append(acc)
        logger.debug("Fold %d/%d accuracy: %.4f", fold + 1, n_splits, acc)

    mean_acc = float(np.mean(accuracies))
    score = abs(mean_acc - 0.5)
    logger.info("Discriminative score: %.4f (mean acc=%.4f)", score, mean_acc)
    return score


# ---------------------------------------------------------------------------
# Predictive score
# ---------------------------------------------------------------------------

def predictive_score(
    real_data: np.ndarray,
    fake_data: np.ndarray,
) -> float:
    """Train GRU predictor on fake data, evaluate MAE on real data.

    Args:
        real_data: Array of shape (n, seq_len, n_features).
        fake_data: Array of shape (n, seq_len, n_features).

    Returns:
        MAE on real test data. Lower is better.
    """
    real_data = _to_numpy(real_data)
    fake_data = _to_numpy(fake_data)
    n_features = real_data.shape[2]
    device = _get_device()

    # Prepare fake: input = t[:-1], target = t[-1]
    X_fake = torch.tensor(fake_data[:, :-1, :], dtype=torch.float32)
    y_fake = torch.tensor(fake_data[:, -1, :], dtype=torch.float32)
    train_ds = TensorDataset(X_fake, y_fake)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = _GRUPredictor(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    model.train()
    for epoch in range(20):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Evaluate on real data
    X_real = torch.tensor(real_data[:, :-1, :], dtype=torch.float32)
    y_real = torch.tensor(real_data[:, -1, :], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        preds = model(X_real.to(device)).cpu()
        mae = criterion(preds, y_real).item()

    logger.info("Predictive score (MAE): %.4f", mae)
    return float(mae)


# ---------------------------------------------------------------------------
# MMD
# ---------------------------------------------------------------------------

def _median_heuristic(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute median heuristic bandwidth for RBF kernel."""
    combined = np.concatenate([X, Y], axis=0)
    n = min(combined.shape[0], 1000)
    subset = combined[np.random.choice(combined.shape[0], n, replace=False)]
    dists = np.sum((subset[:, None, :] - subset[None, :, :]) ** 2, axis=-1)
    median_dist = float(np.median(dists[np.triu_indices(n, k=1)]))
    return max(median_dist, 1e-8)


def compute_mmd(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    kernel: str = "rbf",
) -> float:
    """Maximum Mean Discrepancy with RBF kernel.

    Flatten time series to vectors, compute MMD with median heuristic.

    Args:
        real_data: Array of shape (n, seq_len, n_features).
        fake_data: Array of shape (n, seq_len, n_features).
        kernel: Kernel type. Only 'rbf' is supported.

    Returns:
        MMD value. Lower is better.
    """
    if kernel != "rbf":
        raise ValueError(f"Unsupported kernel: {kernel}. Use 'rbf'.")

    real_data = _to_numpy(real_data)
    fake_data = _to_numpy(fake_data)

    X = real_data.reshape(real_data.shape[0], -1)
    Y = fake_data.reshape(fake_data.shape[0], -1)

    sigma_sq = _median_heuristic(X, Y)

    def rbf(A: np.ndarray, B: np.ndarray) -> float:
        dist = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return float(np.mean(np.exp(-dist / (2.0 * sigma_sq))))

    mmd = rbf(X, X) - 2.0 * rbf(X, Y) + rbf(Y, Y)
    mmd = max(mmd, 0.0)
    logger.info("MMD: %.6f", mmd)
    return float(mmd)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(
    real_data: np.ndarray,
    fake_data: np.ndarray,
) -> float:
    """Frechet Inception Distance adapted for time series.

    Uses mean and covariance of flattened feature vectors.

    Args:
        real_data: Array of shape (n, seq_len, n_features).
        fake_data: Array of shape (n, seq_len, n_features).

    Returns:
        FID value. Lower is better.
    """
    real_data = _to_numpy(real_data)
    fake_data = _to_numpy(fake_data)

    X = real_data.reshape(real_data.shape[0], -1)
    Y = fake_data.reshape(fake_data.shape[0], -1)

    mu_r, mu_f = np.mean(X, axis=0), np.mean(Y, axis=0)
    cov_r = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-6
    cov_f = np.cov(Y, rowvar=False) + np.eye(Y.shape[1]) * 1e-6

    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(cov_r @ cov_f, disp=False)

    # Handle numerical issues: discard imaginary part
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(
        diff @ diff + np.trace(cov_r + cov_f - 2.0 * covmean)
    )
    fid = max(fid, 0.0)
    logger.info("FID: %.4f", fid)
    return fid


# ---------------------------------------------------------------------------
# ACF difference
# ---------------------------------------------------------------------------

def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation for a 1-D series using numpy.

    Args:
        x: 1-D array.
        max_lag: Maximum lag.

    Returns:
        ACF values of shape (max_lag + 1,).
    """
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var < 1e-10:
        return np.zeros(max_lag + 1)
    acf_vals = np.correlate(x_centered, x_centered, mode="full")
    acf_vals = acf_vals[n - 1:]  # keep non-negative lags
    acf_vals = acf_vals[: max_lag + 1] / (var * n)
    return acf_vals


def acf_difference(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    max_lag: int = 24,
) -> float:
    """Autocorrelation function difference between real and fake.

    Compute ACF per feature averaged over samples, compare.

    Args:
        real_data: Array of shape (n, seq_len, n_features).
        fake_data: Array of shape (n, seq_len, n_features).
        max_lag: Maximum lag for ACF.

    Returns:
        Mean absolute ACF difference across features and lags.
    """
    real_data = _to_numpy(real_data)
    fake_data = _to_numpy(fake_data)
    n_features = real_data.shape[2]
    max_lag = min(max_lag, real_data.shape[1] - 1)

    diffs = []
    for f in range(n_features):
        acf_real = np.mean(
            [_compute_acf(real_data[i, :, f], max_lag)
             for i in range(real_data.shape[0])],
            axis=0,
        )
        acf_fake = np.mean(
            [_compute_acf(fake_data[i, :, f], max_lag)
             for i in range(fake_data.shape[0])],
            axis=0,
        )
        diffs.append(np.mean(np.abs(acf_real - acf_fake)))

    score = float(np.mean(diffs))
    logger.info("ACF difference: %.4f", score)
    return score


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    real_data: np.ndarray,
    fake_data: np.ndarray,
    max_lag: int = 24,
) -> Dict[str, float]:
    """Run all evaluation metrics.

    Args:
        real_data: Array of shape (n, seq_len, n_features).
        fake_data: Array of shape (n, seq_len, n_features).
        max_lag: Maximum lag for ACF.

    Returns:
        Dict with keys: discriminative_score, predictive_score,
        mmd, fid, acf_diff.
    """
    logger.info("Running full evaluation suite...")
    results: Dict[str, float] = {
        "discriminative_score": discriminative_score(real_data, fake_data),
        "predictive_score": predictive_score(real_data, fake_data),
        "mmd": compute_mmd(real_data, fake_data),
        "fid": compute_fid(real_data, fake_data),
        "acf_diff": acf_difference(real_data, fake_data, max_lag=max_lag),
    }
    logger.info("Evaluation results: %s", results)
    return results
