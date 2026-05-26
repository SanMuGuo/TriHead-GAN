"""GPU-resident batch cache with per-epoch sample-level shuffling.

Rationale: pre-caching an entire DataLoader on GPU eliminates CPU->GPU
transfer overhead, but freezes "which samples live in which mini-batch"
for the whole run.  For highly-correlated sliding-window time series
this weakens SGD noise and makes comparisons between models unfair.

``ShuffledBatchCache`` stores the concatenated dataset once on GPU and
yields freshly-composed mini-batches at every ``__iter__`` call using
``torch.randperm``.  The iteration contract mirrors an iterable of
tensors (``for batch in cache: ...``) so call sites can stay untouched.
"""

import logging
from typing import Callable, Iterable, Iterator, List, Optional

import torch

logger = logging.getLogger(__name__)


class ShuffledBatchCache:
    """All data stays resident on GPU; batches are re-sampled per epoch.

    Args:
        dataloader: Source DataLoader yielding tensors of shape
            ``(batch, seq_len, n_features)``.
        device: Target device (usually ``cuda``).
        transform: Optional callable applied once at cache time (e.g.
            ``x -> 2*x - 1``).  Not applied per iteration.
        drop_last: If True, drop the trailing partial batch.  Default
            False (matches DataLoader's ``drop_last=False``).
    """

    def __init__(
        self,
        dataloader: Iterable[torch.Tensor],
        device: torch.device,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        drop_last: bool = False,
    ) -> None:
        cached: List[torch.Tensor] = []
        self._batch_size = 0
        for batch in dataloader:
            batch = batch.to(device)
            if transform is not None:
                batch = transform(batch)
            self._batch_size = max(self._batch_size, batch.size(0))
            cached.append(batch)

        if not cached:
            raise ValueError("ShuffledBatchCache: empty dataloader.")

        # Infer the "intended" batch size from the DataLoader when possible
        # (the last partial batch may be smaller than the rest).
        bs = getattr(dataloader, "batch_size", None)
        if bs is not None and bs > 0:
            self._batch_size = bs

        self.all_data = torch.cat(cached, dim=0)  # (N, T, F)
        self.device = device
        self.drop_last = drop_last

        n_total = self.all_data.size(0)
        n_batches = n_total // self._batch_size
        if not drop_last and n_total % self._batch_size != 0:
            n_batches += 1
        self._n_batches = max(n_batches, 1)

        total_mb = (
            self.all_data.nelement() * self.all_data.element_size() / 1e6
        )
        logger.info(
            "ShuffledBatchCache ready: n_samples=%d, batch_size=%d, "
            "n_batches=%d (%.1f MB on %s).",
            n_total, self._batch_size, self._n_batches, total_mb, device,
        )

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        n = self.all_data.size(0)
        perm = torch.randperm(n, device=self.device)
        bs = self._batch_size
        last = n if not self.drop_last else (n // bs) * bs
        for start in range(0, last, bs):
            idx = perm[start : start + bs]
            yield self.all_data[idx]

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def n_samples(self) -> int:
        return self.all_data.size(0)

    def sample_batch(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Uniformly resample a fresh mini-batch from the cached pool.

        Intended for WGAN-GP critic inner loops where best practice is to
        draw new real samples on every discriminator step rather than reuse
        the outer-loop batch (which shrinks effective real-sample diversity
        by a factor of n_critic and risks critic overfitting).
        """
        bs = batch_size if batch_size is not None else self._batch_size
        n = self.all_data.size(0)
        idx = torch.randint(0, n, (bs,), device=self.device)
        return self.all_data[idx]
