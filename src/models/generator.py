"""Transformer-based temporal generator for Carbon-TGAN.

Enhancements over a plain Transformer generator:
  - Local temporal convolution after Transformer to capture fine-grained dynamics.
  - Per-step noise injection to increase temporal variation and combat over-smoothing.
"""

import math
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences.

    Adds position-dependent signals so the Transformer can reason about
    the ordering of time steps.

    Args:
        d_model: Dimension of the model embeddings.
        max_len: Maximum sequence length supported.
        dropout: Dropout probability applied after adding the encoding.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encoding matrix: (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor with positional encoding added, same shape as input.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Generator(nn.Module):
    """Transformer-based temporal generator for Carbon-TGAN.

    Architecture:
        noise z (batch, seq_len, z_dim)
          -> Linear Projection -> (batch, seq_len, d_model)
          -> + Positional Encoding (sinusoidal)
          -> TransformerEncoder (N layers, H heads)
          -> Local Temporal Conv (captures fine-grained dynamics)
          -> + Per-step Noise Injection (combats over-smoothing)
          -> Linear Projection -> (batch, seq_len, n_features)
          -> Tanh -> [-1, 1]

    The caller is responsible for rescaling to the target range.

    Args:
        z_dim: Dimension of the input noise vector per time step.
        n_features: Number of output features per time step.
        seq_len: Length of the generated time series.
        d_model: Internal dimension of the Transformer.
        nhead: Number of attention heads.
        num_layers: Number of TransformerEncoder layers.
        dim_feedforward: Hidden dimension of the feed-forward network.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        z_dim: int,
        n_features: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection: z_dim -> d_model
        self.input_proj = nn.Linear(z_dim, d_model)

        # Positional encoding
        self.pos_encoder = TemporalPositionalEncoding(d_model, max_len=seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Local temporal convolution to capture fine-grained dynamics.
        # Transformer attention tends to produce over-smoothed outputs;
        # local conv adds high-frequency temporal detail.
        self.local_conv = nn.Sequential(
            # (batch, d_model, seq_len)
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=1),
        )

        # Learnable noise injection scale (per-feature).
        # Initialized small so it doesn't disrupt early training.
        self.noise_scale = nn.Parameter(torch.full((1, 1, d_model), 0.05))

        # Output projection: d_model -> n_features
        self.output_proj = nn.Linear(d_model, n_features)

        # Tanh activation to bound output to [-1, 1]
        self.tanh = nn.Tanh()

        logger.info(
            "Generator initialized: z_dim=%d, n_features=%d, seq_len=%d, "
            "d_model=%d, nhead=%d, num_layers=%d",
            z_dim, n_features, seq_len, d_model, nhead, num_layers,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate fake time series from noise.

        Args:
            z: Noise tensor of shape (batch, seq_len, z_dim).

        Returns:
            Generated time series of shape (batch, seq_len, n_features),
            with values in [-1, 1].
        """
        # Project noise to d_model dimension
        h = self.input_proj(z)  # (batch, seq_len, d_model)

        # Add positional encoding
        h = self.pos_encoder(h)  # (batch, seq_len, d_model)

        # Transformer encoding
        h = self.transformer_encoder(h)  # (batch, seq_len, d_model)

        # Local temporal conv (residual connection)
        h_perm = h.permute(0, 2, 1)  # (batch, d_model, seq_len)
        h = h + self.local_conv(h_perm).permute(0, 2, 1)  # residual add

        # Per-step noise injection to combat over-smoothing
        if self.training:
            noise = torch.randn_like(h) * self.noise_scale
            h = h + noise

        # Project to output features
        out = self.output_proj(h)  # (batch, seq_len, n_features)

        # Bound output to [-1, 1]
        fake_ts = self.tanh(out)

        return fake_ts
