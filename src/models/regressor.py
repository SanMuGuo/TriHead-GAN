"""Pre-training DNN regressor for Carbon-TGAN.

The DNN regressor learns to predict the target column (e.g. CO2) from
the remaining input features. Its trained weights are then transferred
to the discriminator's R-Head CNN branch and R head to provide a warm
start for adversarial training.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DNN(nn.Module):
    """Pre-training DNN regressor.

    Maps input features (excluding the target column) to the target value
    using a 1D-CNN architecture that mirrors the discriminator's R-Head
    CNN branch and R head.

    Args:
        n_features: Total number of features (including target).
            The input to this model has n_features - 1 columns.
        seq_len: Length of the time series.
        hidden_dim: Number of filters in the CNN layers.
    """

    def __init__(self, n_features: int, seq_len: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        input_channels = n_features - 1

        # 3-layer 1D-CNN (mirrors discriminator's R-Head CNN branch)
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([hidden_dim, seq_len]),
            # Layer 2
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([hidden_dim, seq_len]),
            # Layer 3
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([hidden_dim, seq_len]),
        )

        # Regression head: predict target per time step
        self.head = nn.Linear(hidden_dim, 1)

        logger.info(
            "DNN regressor initialized: n_features=%d (input=%d), "
            "seq_len=%d, hidden_dim=%d",
            n_features, input_channels, seq_len, hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict target values from input features.

        Args:
            x: Input tensor of shape (batch, seq_len, n_features-1),
               containing all features except the target column.

        Returns:
            Predicted target values of shape (batch, seq_len, 1).
        """
        # Permute for Conv1d: (batch, n_features-1, seq_len)
        h = x.permute(0, 2, 1)

        # CNN feature extraction
        h = self.cnn(h)  # (batch, hidden_dim, seq_len)

        # Permute back: (batch, seq_len, hidden_dim)
        h = h.permute(0, 2, 1)

        # Regression prediction
        y_pred = self.head(h)  # (batch, seq_len, 1)

        return y_pred
