"""Carbon-TGAN training and generation manager.

Orchestrates pre-training, adversarial training with WGAN-GP, and
sample generation for the Carbon-TGAN model.

Key optimizations:
  - Generator outputs [-1, 1] (Tanh); data rescaled to [-1, 1] internally.
  - Exponential Moving Average (EMA) of Generator weights for generation.
  - Adaptive auxiliary-loss warmup to avoid interfering with early WGAN training.
  - Cosine-annealing learning-rate schedule.
  - Gradient clipping for both G and D.
  - Anti-smoothing loss to match temporal variation of real data.
  - Spectral Normalization on discriminator for stable WGAN training.
  - Local temporal conv + noise injection in generator.

R Head performs cross-variable regression (multivariate only): predict
the target column from non-target features via a dedicated CNN branch
whose input excludes the target, so the target cannot leak into the
representation used to predict it.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .generator import Generator
from .discriminator import Discriminator
from .regressor import DNN
from ..utils.batch_cache import ShuffledBatchCache

logger = logging.getLogger(__name__)

# Minimum number of warmup epochs before saving best model
_BEST_MODEL_WARMUP = 100


class _DictConfig:
    """Wrapper to access dict keys as attributes."""

    def __init__(self, d: dict) -> None:
        self._d = d

    def __getattr__(self, key: str) -> Any:
        try:
            return self._d[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")


class MLPGenerator(nn.Module):
    """Optional simple MLP generator variant.

    Outputs values in [-1, 1] via Tanh.
    """

    def __init__(self, z_dim: int, n_features: int, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        flat_dim = seq_len * z_dim
        flat_out = seq_len * n_features
        self.net = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, flat_out),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.size(0)
        out = self.net(z.view(batch, -1))
        return out.view(batch, self.seq_len, self.n_features)


# ------------------------------------------------------------------
# EMA helper
# ------------------------------------------------------------------

class _EMA:
    """Exponential Moving Average with timm-style decay warmup.

    The shadow copy starts from the model's (random) init; at that point
    the "EMA" is a pure average of random and noisy early-training
    parameters, which degrades generation quality until enough steps
    have accumulated.  We therefore ramp the effective decay from 0 to
    ``decay`` via ``d_eff(step) = min(decay, (1 + step) / (10 + step))``
    so the shadow tracks the live model closely at the start and only
    smooths progressively as training matures.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._step = 0

    def _effective_decay(self) -> float:
        warmup = (1.0 + self._step) / (10.0 + self._step)
        return min(self.decay, warmup)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self._step += 1
        d = self._effective_decay()
        for s_param, m_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            s_param.data.mul_(d).add_(m_param.data, alpha=1.0 - d)
        # Also sync non-parameter state (e.g. BatchNorm running_mean /
        # running_var, num_batches_tracked).  Without this the shadow
        # generator keeps its init-time running stats forever, which can make
        # BN-containing generator variants produce poor samples at eval time.
        # We copy rather than EMA-smooth because
        # BN buffers already carry an internal momentum average.
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            if s_buf.dtype.is_floating_point:
                s_buf.data.copy_(m_buf.data)
            else:
                # Integer buffers (e.g. num_batches_tracked) can't EMA;
                # mirror them exactly.
                s_buf.data.copy_(m_buf.data)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.shadow(*args, **kwargs)


class CarbonTGAN:
    """Carbon-TGAN training and generation manager.

    Args:
        config: Dict or attribute-accessible config with keys:
            z_dim, n_features, seq_len, d_model, nhead, num_layers,
            dim_feedforward, dropout, hidden_dim, lr, n_critic,
            lambda_gp, alpha, beta, gamma, delta, epochs,
            pretrain_epochs, pretrain_lr, device.
            Optional: use_mlp_generator, warmup_epochs, grad_clip,
            ema_decay.
    """

    def __init__(self, config: Any) -> None:
        if isinstance(config, dict):
            config = _DictConfig(config)
        self.cfg = config
        self.device = torch.device(getattr(config, "device", "cuda"))

        n_features = config.n_features
        if n_features < 2:
            raise ValueError(
                f"CarbonTGAN is multivariate-only; got n_features={n_features}."
            )
        self._half = config.seq_len // 2

        # Build generator.
        use_mlp = getattr(config, "use_mlp_generator", False)
        if use_mlp:
            self.G = MLPGenerator(
                z_dim=config.z_dim,
                n_features=n_features,
                seq_len=config.seq_len,
            ).to(self.device)
            logger.info("Using MLP generator variant.")
        else:
            self.G = Generator(
                z_dim=config.z_dim,
                n_features=n_features,
                seq_len=config.seq_len,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
            ).to(self.device)

        use_sn = getattr(config, "spectral_norm", False)
        self.D = Discriminator(
            n_features=n_features,
            seq_len=config.seq_len,
            hidden_dim=config.hidden_dim,
            causal=False,
            use_spectral_norm=use_sn,
        ).to(self.device)

        self.dnn = DNN(
            n_features=n_features,
            seq_len=config.seq_len,
            hidden_dim=config.hidden_dim,
        ).to(self.device)

        lr = getattr(config, "lr", 1e-4)
        self.opt_G = torch.optim.Adam(
            self.G.parameters(), lr=lr, betas=(0.5, 0.9),
        )
        self.opt_D = torch.optim.Adam(
            self.D.parameters(), lr=lr, betas=(0.5, 0.9),
        )

        self.mse = nn.MSELoss()
        self.best_wd: float = float("inf")

        # EMA
        ema_decay = getattr(config, "ema_decay", 0.999)
        self.ema = _EMA(self.G, decay=ema_decay)

        # Training hyper-params with defaults for backward compatibility
        self._grad_clip = getattr(config, "grad_clip", 1.0)
        self._warmup_epochs = getattr(config, "warmup_epochs", 1000)
        self._eta = getattr(config, "eta", 0.0)  # anti-smoothing weight

        # AMP mixed precision (FP16 on CUDA for ~20-40% speedup)
        self._use_amp = self.device.type == "cuda"
        self.scaler_G = GradScaler("cuda", enabled=self._use_amp)
        self.scaler_D = GradScaler("cuda", enabled=self._use_amp)

        logger.info(
            "CarbonTGAN initialized on device=%s, "
            "ema_decay=%.4f, warmup=%d, grad_clip=%.1f, amp=%s",
            self.device, ema_decay,
            self._warmup_epochs, self._grad_clip, self._use_amp,
        )

    # ------------------------------------------------------------------
    # Data transform helpers: [0,1] <-> [-1,1]
    # ------------------------------------------------------------------

    @staticmethod
    def _to_model_space(x: torch.Tensor) -> torch.Tensor:
        """Convert data from [0,1] to [-1,1]."""
        return x * 2.0 - 1.0

    @staticmethod
    def _to_data_space(x: torch.Tensor) -> torch.Tensor:
        """Convert data from [-1,1] to [0,1]."""
        return (x + 1.0) / 2.0

    # ------------------------------------------------------------------
    # Auxiliary loss warmup
    # ------------------------------------------------------------------

    def _aux_weight(self, epoch: int, target_weight: float) -> float:
        """Linearly ramp auxiliary loss weight over warmup period."""
        if self._warmup_epochs <= 0:
            return target_weight
        progress = min(1.0, epoch / self._warmup_epochs)
        return target_weight * progress

    # ------------------------------------------------------------------
    # R Head loss helpers
    # ------------------------------------------------------------------

    def _r_head_loss(
        self, r_pred: torch.Tensor, data: torch.Tensor
    ) -> torch.Tensor:
        """R-Head loss: predict target (last) column from non-target features."""
        target = data[:, :, -1:]
        return self.mse(r_pred, target)

    # ------------------------------------------------------------------
    # Pre-training
    # ------------------------------------------------------------------

    def pretrain_regressor(
        self, dataloader: DataLoader, epochs: int = 1000
    ) -> None:
        """Pre-train the DNN regressor, then transfer weights to D.

        Skipped automatically when epochs <= 0.

        Note: DNN pre-training operates in [0,1] data space (original
        MinMaxScaler range), while adversarial training uses [-1,1].
        The transferred weights may be sub-optimal.  Set
        ``pretrain.epochs: 0`` (default) to disable and rely on the
        auxiliary-loss warmup instead.
        """
        if epochs <= 0:
            logger.info("Pre-training disabled (epochs=%d).", epochs)
            return

        pretrain_lr = getattr(self.cfg, "pretrain_lr", 1e-4)
        opt = torch.optim.Adam(self.dnn.parameters(), lr=pretrain_lr)
        scaler = GradScaler("cuda", enabled=self._use_amp)
        self.dnn.train()

        # Pre-cache batches on GPU
        gpu_batches = [batch.to(self.device) for batch in dataloader]

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for batch in gpu_batches:
                # DNN operates in [0,1] space (original data range)
                x_input = batch[:, :, :-1]
                y_target = batch[:, :, -1:]

                with autocast("cuda", enabled=self._use_amp):
                    y_pred = self.dnn(x_input)
                    loss = self.mse(y_pred, y_target)

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1

            if epoch % 100 == 0 or epoch == 1:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(
                    "DNN pre-train epoch %d/%d, loss=%.6f",
                    epoch, epochs, avg_loss,
                )

        self._transfer_weights()
        logger.info("Weight transfer from DNN to Discriminator completed.")

    def _transfer_weights(self) -> None:
        """Transfer pretrained DNN weights to the R-Head CNN branch and head.

        The DNN has input channels = n_features - 1, matching the
        multivariate R-Head branch ``D.r_cnn``. Transfer all shape-
        compatible parameters in order.
        """
        dnn_state = self.dnn.cnn.state_dict()
        d_state = self.D.r_cnn.state_dict()
        n_transferred = 0
        for key in dnn_state:
            if key in d_state and dnn_state[key].shape == d_state[key].shape:
                d_state[key].copy_(dnn_state[key])
                n_transferred += 1

        self.D.r_cnn.load_state_dict(d_state)
        self.D.r_head.load_state_dict(self.dnn.head.state_dict())
        logger.info(
            "Transferred %d parameter tensors from DNN.cnn to D.r_cnn "
            "+ R head.",
            n_transferred,
        )

    # ------------------------------------------------------------------
    # Gradient penalty
    # ------------------------------------------------------------------

    def compute_gradient_penalty(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute WGAN-GP gradient penalty.

        Always computed in FP32 for numerical stability, even when AMP
        is enabled.
        """
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        alpha = alpha.expand_as(real)

        interpolated = (
            alpha * real.float() + (1 - alpha) * fake.float()
        ).requires_grad_(True)

        # Disable AMP for GP to ensure FP32 gradients
        with autocast("cuda", enabled=False):
            d_score, _, _ = self.D(interpolated)

        gradients = torch.autograd.grad(
            outputs=d_score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_score),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.reshape(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        """Sample noise from standard normal distribution."""
        return torch.randn(
            batch_size, self.cfg.seq_len, self.cfg.z_dim, device=self.device
        )

    def train_step_discriminator(
        self, real_data: torch.Tensor, epoch: int
    ) -> Dict[str, float]:
        """One discriminator training step."""
        batch_size = real_data.size(0)
        cfg = self.cfg

        z = self._sample_noise(batch_size)
        with torch.no_grad():
            with autocast("cuda", enabled=self._use_amp):
                fake_data = self.G(z)

        with autocast("cuda", enabled=self._use_amp):
            # R-Head input drops the target column to prevent the target
            # from leaking into its own prediction.
            real_r_in = real_data[..., :-1]
            fake_r_in = fake_data[..., :-1]
            d_real, r_real, t_real = self.D(real_data, x_r=real_r_in)
            d_fake, r_fake, t_fake = self.D(fake_data, x_r=fake_r_in)

            l_wgan = d_fake.mean() - d_real.mean()
            l_reg_real = self._r_head_loss(r_real, real_data)
            l_reg_fake = self._r_head_loss(r_fake, fake_data)

            real_diff = real_data[:, 1:, :] - real_data[:, :-1, :]
            fake_diff = fake_data[:, 1:, :] - fake_data[:, :-1, :]
            l_temp_real = self.mse(t_real, real_diff)
            l_temp_fake = self.mse(t_fake, fake_diff)

        # GP computed in FP32 (handled inside compute_gradient_penalty)
        gp = self.compute_gradient_penalty(real_data, fake_data)

        # Adaptive auxiliary weights
        alpha_w = self._aux_weight(epoch, cfg.alpha)
        beta_w = self._aux_weight(epoch, cfg.beta)
        delta_w = self._aux_weight(epoch, cfg.delta)

        loss_d = (
            l_wgan
            + cfg.lambda_gp * gp
            + alpha_w * l_reg_real
            + beta_w * l_reg_fake
            + delta_w * (l_temp_real + l_temp_fake)
        )

        self.opt_D.zero_grad(set_to_none=True)
        self.scaler_D.scale(loss_d).backward()
        if self._grad_clip > 0:
            self.scaler_D.unscale_(self.opt_D)
            nn.utils.clip_grad_norm_(self.D.parameters(), self._grad_clip)
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()

        return {
            "loss_d": loss_d.item(),
            "l_wgan": l_wgan.item(),
            "gp": gp.item(),
            "l_reg_real": l_reg_real.item(),
            "l_reg_fake": l_reg_fake.item(),
            "l_temp_real": l_temp_real.item(),
            "l_temp_fake": l_temp_fake.item(),
            "wd": -l_wgan.item(),
        }

    def train_step_generator(
        self, real_data: torch.Tensor, batch_size: int, epoch: int
    ) -> Dict[str, float]:
        """One generator training step.

        Args:
            real_data: Real batch for computing anti-smoothing loss target.
            batch_size: Number of samples in the batch.
            epoch: Current epoch number.
        """
        cfg = self.cfg

        z = self._sample_noise(batch_size)
        with autocast("cuda", enabled=self._use_amp):
            fake_data = self.G(z)
            fake_r_in = fake_data[..., :-1]
            d_fake, r_fake, t_fake = self.D(fake_data, x_r=fake_r_in)
            l_reg_fake = self._r_head_loss(r_fake, fake_data)

            l_adv = -d_fake.mean()

            fake_diff = fake_data[:, 1:, :] - fake_data[:, :-1, :]
            l_temp_fake = self.mse(t_fake, fake_diff)

            # Anti-smoothing loss: match BOTH the mean and std of the
            # per-feature absolute first-difference distribution.
            # Matching only the mean pins the average step magnitude but
            # leaves the shape (in particular the tail of the variation
            # distribution) unconstrained.  The std term penalises
            # generators that match average variation but collapse the
            # distribution to a tight band, which can cause residual
            # over-smoothing on high-volatility carbon series.
            real_abs_diff = torch.abs(real_data[:, 1:, :] - real_data[:, :-1, :])
            fake_abs_diff = torch.abs(fake_diff)
            real_mean = real_abs_diff.mean(dim=(0, 1)).detach()
            fake_mean = fake_abs_diff.mean(dim=(0, 1))
            real_std = real_abs_diff.std(dim=(0, 1), unbiased=False).detach()
            fake_std = fake_abs_diff.std(dim=(0, 1), unbiased=False)
            l_smooth = self.mse(fake_mean, real_mean) + self.mse(fake_std, real_std)

            # Adaptive auxiliary weights
            gamma_w = self._aux_weight(epoch, cfg.gamma)
            delta_w = self._aux_weight(epoch, cfg.delta)
            eta_w = self._aux_weight(epoch, self._eta)

            loss_g = (
                l_adv
                + gamma_w * l_reg_fake
                + delta_w * l_temp_fake
                + eta_w * l_smooth
            )

        self.opt_G.zero_grad(set_to_none=True)
        self.scaler_G.scale(loss_g).backward()
        if self._grad_clip > 0:
            self.scaler_G.unscale_(self.opt_G)
            nn.utils.clip_grad_norm_(self.G.parameters(), self._grad_clip)
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()

        # Update EMA
        self.ema.update(self.G)

        return {
            "loss_g": loss_g.item(),
            "l_adv": l_adv.item(),
            "l_reg_fake": l_reg_fake.item(),
            "l_temp_fake": l_temp_fake.item(),
            "l_smooth": l_smooth.item(),
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader: DataLoader,
        epochs: Optional[int] = None,
        log_every: int = 100,
        save_every: int = 1000,
        save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Full adversarial training loop.

        Data from the dataloader is assumed to be in [0, 1].  It is
        rescaled to [-1, 1] internally to match the Tanh generator.

        Args:
            dataloader: Yields tensors of shape (batch, seq_len, n_features).
            epochs: Override config epochs if provided.
            log_every: Log metrics every N epochs.
            save_every: Save checkpoint every N epochs.
            save_dir: Directory to save periodic checkpoints.

        Returns:
            Training history dict with keys: d_loss, g_loss, w_distance.
        """
        epochs = epochs or self.cfg.epochs
        self.G.train()
        self.D.train()

        # LR schedulers (cosine annealing)
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_G, T_max=epochs, eta_min=1e-6,
        )
        scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_D, T_max=epochs, eta_min=1e-6,
        )

        history: Dict[str, List[float]] = {
            "d_loss": [], "g_loss": [], "w_distance": [],
        }

        # Pre-cache the full dataset on GPU once.  Each epoch re-samples
        # mini-batches via ``torch.randperm`` so both sample-level and
        # batch-level shuffling are preserved (a plain list-of-batches
        # cache would freeze mini-batch composition for the whole run).
        gpu_batches = ShuffledBatchCache(
            dataloader,
            device=self.device,
            transform=self._to_model_space,
        )

        logger.info("Starting adversarial training for %d epochs.", epochs)

        for epoch in range(1, epochs + 1):
            epoch_d_metrics: Dict[str, float] = {}
            epoch_g_metrics: Dict[str, float] = {}
            n_d_steps = 0
            n_g_steps = 0

            for real_data in gpu_batches:
                batch_size = real_data.size(0)

                # WGAN-GP best practice: resample real data on every critic
                # step so the discriminator sees the full training pool and
                # the Wasserstein-distance estimate stays well-calibrated.
                for _ in range(self.cfg.n_critic):
                    d_real = gpu_batches.sample_batch(batch_size)
                    d_metrics = self.train_step_discriminator(
                        d_real, epoch,
                    )
                    for k, v in d_metrics.items():
                        epoch_d_metrics[k] = epoch_d_metrics.get(k, 0.0) + v
                    n_d_steps += 1

                g_metrics = self.train_step_generator(real_data, batch_size, epoch)
                for k, v in g_metrics.items():
                    epoch_g_metrics[k] = epoch_g_metrics.get(k, 0.0) + v
                n_g_steps += 1

            if n_d_steps > 0:
                for k in epoch_d_metrics:
                    epoch_d_metrics[k] /= n_d_steps
            if n_g_steps > 0:
                for k in epoch_g_metrics:
                    epoch_g_metrics[k] /= n_g_steps

            # Step LR schedulers
            scheduler_G.step()
            scheduler_D.step()

            # Record history
            history["d_loss"].append(epoch_d_metrics.get("loss_d", 0.0))
            history["g_loss"].append(epoch_g_metrics.get("loss_g", 0.0))
            history["w_distance"].append(epoch_d_metrics.get("wd", 0.0))

            if epoch % log_every == 0 or epoch == 1:
                wd = epoch_d_metrics.get("wd", 0.0)
                lr_g = scheduler_G.get_last_lr()[0]
                logger.info(
                    "Epoch %d/%d | D_loss=%.4f WD=%.4f | G_loss=%.4f | lr=%.2e",
                    epoch, epochs,
                    epoch_d_metrics.get("loss_d", 0.0),
                    wd,
                    epoch_g_metrics.get("loss_g", 0.0),
                    lr_g,
                )

            # Save "best" model (only after warmup).
            # Note: selection criterion is min positive Wasserstein distance,
            # a discriminator-state metric, not a generation-quality proxy.
            # Downstream analysis / evaluation should prefer ``model_final.pt``.
            wd = epoch_d_metrics.get("wd", 0.0)
            warmup = min(_BEST_MODEL_WARMUP, epochs // 5)
            if epoch > warmup and wd > 0 and wd < self.best_wd:
                self.best_wd = wd
                if save_dir:
                    self.save(str(Path(save_dir) / "model_best.pt"))

            # Periodic checkpoint
            if save_dir and save_every > 0 and epoch % save_every == 0:
                self.save(str(Path(save_dir) / f"model_epoch_{epoch}.pt"))

        logger.info("Training completed. Best WD=%.6f", self.best_wd)
        return history

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, n_samples: int) -> np.ndarray:
        """Generate fake time series samples using EMA weights.

        Output is rescaled from [-1,1] back to [0,1] for evaluation.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Generated time series as numpy array, shape
            (n_samples, seq_len, n_features), values in [0, 1].
        """
        self.ema.shadow.eval()
        with torch.no_grad():
            z = self._sample_noise(n_samples)
            fake = self.ema.forward(z)
            # Rescale from [-1,1] to [0,1]
            fake = self._to_data_space(fake)
            # Clamp to ensure strict [0,1] range
            fake = fake.clamp(0.0, 1.0)
        return fake.cpu().numpy()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model checkpoints."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "generator": self.G.state_dict(),
            "discriminator": self.D.state_dict(),
            "opt_g": self.opt_G.state_dict(),
            "opt_d": self.opt_D.state_dict(),
            "best_wd": self.best_wd,
            "ema_generator": self.ema.shadow.state_dict(),
        }
        if self.dnn is not None:
            state["dnn"] = self.dnn.state_dict()
        torch.save(state, path)
        logger.info("Checkpoint saved to %s", path)

    def load(self, path: str) -> None:
        """Load model checkpoints."""
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False,
        )
        self.G.load_state_dict(checkpoint["generator"])
        self.D.load_state_dict(checkpoint["discriminator"])
        if self.dnn is not None and "dnn" in checkpoint:
            self.dnn.load_state_dict(checkpoint["dnn"])
        self.opt_G.load_state_dict(checkpoint["opt_g"])
        self.opt_D.load_state_dict(checkpoint["opt_d"])
        self.best_wd = checkpoint.get("best_wd", float("inf"))
        if "ema_generator" in checkpoint:
            self.ema.shadow.load_state_dict(checkpoint["ema_generator"])
        logger.info("Checkpoint loaded from %s", path)
