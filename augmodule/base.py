from __future__ import annotations
import torch
import torch.nn as nn

class AugBase(nn.Module):
    """Common helpers for random, per-sample, probability-gated augmentations."""
    def __init__(self, p: float = 1.0):
        super().__init__()
        if not (0.0 <= float(p) <= 1.0):
            raise ValueError(f"p must be in [0,1], got {p}")
        self.p = float(p)

    @staticmethod
    def _check_input(x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Input must be floating dtype (e.g., float32).")

    @staticmethod
    def _rand(shape, device, generator=None):
        return torch.rand(shape, device=device, generator=generator)

    def _apply_prob(self, b: int, device, generator=None):
        """Return boolean mask of shape [B] selecting samples to augment."""
        if self.p >= 1.0:
            return torch.ones(b, dtype=torch.bool, device=device)
        if self.p <= 0.0:
            return torch.zeros(b, dtype=torch.bool, device=device)
        return self._rand((b,), device, generator) < self.p