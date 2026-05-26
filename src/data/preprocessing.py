"""Preprocessing pipelines for public time series datasets."""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def _interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values via linear interpolation + forward/backward fill.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with no NaN values.
    """
    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        logger.info("Interpolating %d missing values.", n_missing)
        df = df.interpolate(method="linear")
        df = df.ffill().bfill()
    return df


def _normalize(data: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Min-max normalize to [0, 1].

    Args:
        data: 2-D array of shape (n_samples, n_features).

    Returns:
        Tuple of (normalized array, fitted MinMaxScaler).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized = scaler.fit_transform(data)
    return normalized, scaler


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def create_sliding_windows(data: np.ndarray, window_size: int = 24,
                           stride: int = 12) -> np.ndarray:
    """Segment a 2-D array into overlapping sliding windows.

    Args:
        data: Array of shape (n_samples, n_features).
        window_size: Length of each window.
        stride: Step size between consecutive windows.

    Returns:
        Array of shape (n_windows, window_size, n_features).

    Raises:
        ValueError: If data has fewer samples than window_size.
    """
    n_samples, n_features = data.shape
    if n_samples < window_size:
        raise ValueError(
            f"Data length ({n_samples}) is shorter than window_size "
            f"({window_size})."
        )

    starts = range(0, n_samples - window_size + 1, stride)
    windows = np.stack([data[i:i + window_size] for i in starts])
    logger.info(
        "Created %d windows (window_size=%d, stride=%d, features=%d).",
        len(windows), window_size, stride, n_features,
    )
    return windows


# ---------------------------------------------------------------------------
# Generic preprocessing
# ---------------------------------------------------------------------------


def preprocess_generic(
    csv_path: str,
    target_col: str = "OT",
    window_size: int = 24,
    stride: int = 12,
) -> Tuple[np.ndarray, MinMaxScaler, List[str]]:
    """Preprocess a generic CSV dataset (ETTh1, chinaCarbon, usCarbon).

    Pipeline: drop date column -> missing value interpolation -> MinMax
    normalization.

    Args:
        csv_path: Path to the CSV file.
        target_col: Name of the target column (used for logging only; all
            numeric columns are kept).
        window_size: Sliding window length (unused here, kept for API
            consistency).
        stride: Sliding window step (unused here, kept for API consistency).

    Returns:
        Tuple of (normalized_data as 2-D ndarray, fitted MinMaxScaler,
        list of feature names).
    """
    logger.info("Preprocessing generic dataset from '%s'.", csv_path)
    df = pd.read_csv(csv_path)

    # Drop date column if present
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    # Enforce target_col as the *last* column so that downstream code
    # which assumes data[..., -1] is the regression target is correct
    # by construction, not by luck.
    if target_col in df.columns:
        other_cols = [c for c in df.columns if c != target_col]
        df = df[other_cols + [target_col]]
        logger.info(
            "Reordered columns: target '%s' placed at the last position.",
            target_col,
        )
    else:
        logger.warning(
            "target_col '%s' not found in %s; using natural column order "
            "(the last column will still be treated as the target).",
            target_col, list(df.columns),
        )

    feature_names = df.columns.tolist()
    logger.info("Features (%d): %s", len(feature_names), feature_names)

    # --- Missing value interpolation ---
    df = _interpolate_missing(df)

    # --- Normalization ---
    normalized, scaler = _normalize(df.values)
    logger.info("Generic preprocessing complete. Shape: %s", normalized.shape)

    return normalized, scaler, feature_names
