"""PyTorch Dataset and DataLoader factory for time series windows."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from .preprocessing import (
    create_sliding_windows,
    preprocess_generic,
)

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Sliding-window time series dataset.

    Attributes:
        data: Tensor of shape (n_windows, window_size, n_features).
    """

    def __init__(self, windows: np.ndarray) -> None:
        """Initialize with pre-computed sliding windows.

        Args:
            windows: Array of shape (n_windows, window_size, n_features).
        """
        self.data = torch.FloatTensor(windows)
        logger.info(
            "TimeSeriesDataset created: %d windows, shape %s.",
            len(self.data), tuple(self.data.shape),
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


# ---------------------------------------------------------------------------
# Routing table: dataset name -> preprocessing function
# ---------------------------------------------------------------------------

# Generic datasets mapped to their CSV filenames
_GENERIC_CSV_MAP: Dict[str, str] = {
    "chinaCarbon": "chinaCarbon.csv",
    "chinacarbon": "chinaCarbon.csv",
    "usCarbon": "usCarbon.csv",
    "uscarbon": "usCarbon.csv",
    "ETTh1": "ETTh1.csv",
    "etth1": "ETTh1.csv",
}


def create_dataloader(
    dataset_name: str,
    data_dir: str,
    window_size: int = 24,
    stride: int = 12,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, MinMaxScaler, List[str], int]:
    """Build a DataLoader by dataset name with automatic preprocessing routing.

    Args:
        dataset_name: Name of the dataset (e.g. "ETTh1", "chinaCarbon").
        data_dir: Directory containing the CSV files.
        window_size: Sliding window length.
        stride: Sliding window step.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the DataLoader.
        num_workers: Number of DataLoader workers.

    Returns:
        Tuple of (DataLoader, fitted MinMaxScaler, feature name list,
        number of features).

    Raises:
        ValueError: If dataset_name is not recognized.
        FileNotFoundError: If the CSV file does not exist.
    """
    data_path = Path(data_dir)

    # --- Route to correct preprocessing function ---
    if dataset_name in _GENERIC_CSV_MAP:
        csv_filename = _GENERIC_CSV_MAP[dataset_name]
        csv_path = data_path / csv_filename
        _assert_file_exists(csv_path, dataset_name)
        # Target column (preprocessing moves it to the last position).
        target_col = "OT"
        normalized, scaler, features = preprocess_generic(
            str(csv_path), target_col, window_size, stride,
        )

    else:
        # Fallback: try to find a CSV with the given name
        csv_path = data_path / f"{dataset_name}.csv"
        if csv_path.exists():
            logger.warning(
                "Dataset '%s' not in routing table; using generic pipeline.",
                dataset_name,
            )
            normalized, scaler, features = preprocess_generic(
                str(csv_path), "OT", window_size, stride,
            )
        else:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: "
                f"{sorted(set(_GENERIC_CSV_MAP.keys()))}"
            )

    # --- Create sliding windows and DataLoader ---
    windows = create_sliding_windows(normalized, window_size, stride)
    n_features = windows.shape[2]

    dataset = TimeSeriesDataset(windows)
    use_cuda = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    logger.info(
        "DataLoader ready: dataset=%s, batches=%d, batch_size=%d, "
        "features=%d.",
        dataset_name, len(dataloader), batch_size, n_features,
    )

    return dataloader, scaler, features, n_features


def _assert_file_exists(path: Path, dataset_name: str) -> None:
    """Raise FileNotFoundError if the given path does not exist.

    Args:
        path: Expected CSV file path.
        dataset_name: Dataset name for the error message.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file for dataset '{dataset_name}' not found at {path}."
        )
