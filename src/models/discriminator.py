"""Triple-head temporal discriminator for Carbon-TGAN.

Supports optional Spectral Normalization on Conv1d/Linear layers for stable
WGAN training and reduced Wasserstein Distance oscillation.
"""

import logging
from functools import partial

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

logger = logging.getLogger(__name__)


def _maybe_sn(layer: nn.Module, use_sn: bool) -> nn.Module:
    """Apply spectral normalization to a layer if enabled."""
    if use_sn and isinstance(layer, (nn.Conv1d, nn.Linear)):
        return spectral_norm(layer)
    return layer


class _CausalConv1d(nn.Module):
    """Conv1d with causal (left-only) padding to prevent future leakage."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class Discriminator(nn.Module):
    """Triple-head temporal discriminator.

    Routes the input through three independent task-specific 1D-CNN
    branches, each feeding its own head for Wasserstein critic scoring,
    cross-variable regression, and temporal difference prediction.

    Architecture:
        time series x (batch, seq_len, n_features)
          |-> D branch (3-layer 1D-CNN) -> D Head: Wasserstein critic score (scalar)
          |-> R branch (3-layer 1D-CNN) -> R Head: target regression prediction (batch, seq_len, 1)
          |-> T branch (2-layer causal 1D-CNN) -> T Head: temporal difference prediction (batch, seq_len-1, n_features)

    The three branches are fully independent and do not share weights.
    Leakage-free design: the R branch receives only the non-target
    features, so target information cannot leak into the regression
    representation; the T branch uses causal padding so future
    information cannot leak into the temporal-difference prediction.

    R Head performs cross-variable regression: its branch receives the
    non-target features (``n_features - 1`` channels); the target column
    is never fed into the representation used to predict it.

    Args:
        n_features: Number of input features per time step (>= 2).
        seq_len: Length of the input time series.
        hidden_dim: Number of filters in the CNN layers.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_dim: int = 64,
        causal: bool = False,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.causal = causal

        sn = partial(_maybe_sn, use_sn=use_spectral_norm)

        # D-Head 1D-CNN branch (independent; R Head and T Head each
        # own a separate branch).
        self.d_cnn = nn.Sequential(
            sn(nn.Conv1d(n_features, hidden_dim, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([hidden_dim, seq_len]),
            sn(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([hidden_dim, seq_len]),
            sn(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([hidden_dim, seq_len]),
        )

        # Dedicated R-Head CNN branch, leakage-free.
        # in_channels = n_features - 1 (non-target columns only).
        r_in_channels = max(n_features - 1, 1)
        self._r_in_channels = r_in_channels

        # Vanilla convolutions: cross-variable regression relies on
        # pointwise correlations, not long context.
        r_channels = [r_in_channels, hidden_dim, hidden_dim, hidden_dim]
        r_layers: list[nn.Module] = []
        for i in range(3):
            r_layers += [
                nn.Conv1d(
                    r_channels[i], r_channels[i + 1],
                    kernel_size=3, padding=1,
                ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.LayerNorm([hidden_dim, seq_len]),
            ]
        self.r_cnn = nn.Sequential(*r_layers)

        # Separate causal CNN for T Head (prevents future information leakage).
        self.temporal_cnn = nn.Sequential(
            _CausalConv1d(n_features, hidden_dim, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([hidden_dim, seq_len]),
            _CausalConv1d(hidden_dim, hidden_dim, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([hidden_dim, seq_len]),
        )

        # D Head: Wasserstein critic score (no sigmoid for WGAN)
        self.d_head = nn.Sequential(
            nn.Flatten(),
            sn(nn.Linear(hidden_dim * seq_len, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            sn(nn.Linear(256, 1)),
        )

        # R Head: target regression (predict last column).
        self.r_head = nn.Linear(hidden_dim, 1)

        # T Head: temporal difference prediction.
        self.t_head = nn.Linear(hidden_dim, n_features)

        logger.info(
            "Discriminator initialized: n_features=%d, seq_len=%d, "
            "hidden_dim=%d, causal=%s, spectral_norm=%s, r_in_channels=%d",
            n_features, seq_len, hidden_dim, causal, use_spectral_norm,
            r_in_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_r: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all three heads.

        Args:
            x: Input time series of shape (batch, seq_len, n_features).
                Used by D Head and T Head (full view).
            x_r: R-Head branch input of shape
                (batch, seq_len, n_features-1) = non-target columns.
                When ``None``, it is constructed as ``x[..., :-1]``.

        Returns:
            Tuple of:
                d_score: Wasserstein critic score, shape (batch, 1).
                r_pred: Target regression prediction, shape (batch, seq_len, 1).
                t_pred: Temporal difference prediction, shape (batch, seq_len-1, n_features).
        """
        x_perm = x.permute(0, 2, 1)

        h = self.d_cnn(x_perm)
        h = h.permute(0, 2, 1)

        d_score = self.d_head(h)

        if x_r is None:
            x_r = x[..., :-1] if self.n_features > 1 else x
        r_perm = x_r.permute(0, 2, 1)
        h_r = self.r_cnn(r_perm).permute(0, 2, 1)
        r_pred = self.r_head(h_r)

        h_temporal = self.temporal_cnn(x_perm).permute(0, 2, 1)
        t_pred = self.t_head(h_temporal[:, :-1, :])

        return d_score, r_pred, t_pred
