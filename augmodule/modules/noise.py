from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from ..base import AugBase

class RandomGaussianNoise(AugBase):
    """
    Additive Gaussian noise with per-sample std drawn uniformly from [std_min, std_max].
    x' = clip(x + N(0, σ_i^2)), σ_i ~ U[std_min, std_max]

    Args:
        std_range:   (std_min, std_max), both >= 0
        p:           probability to apply noise per sample
        per_channel: if True, sample an independent σ per channel
        clip, clip_min, clip_max: clamp control
    """
    def __init__(
        self,
        std_range: Tuple[float, float],
        *,
        p: float = 0.5,
        per_channel: bool = False,
        clip: bool = True,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ):
        super().__init__(p=float(p))
        if not (isinstance(std_range, (tuple, list)) and len(std_range) == 2):
            raise ValueError("std_range must be a (std_min, std_max) tuple.")
        s0, s1 = float(std_range[0]), float(std_range[1])
        if s0 < 0 or s1 < 0 or s1 < s0:
            raise ValueError("std_range must satisfy 0 <= std_min <= std_max.")
        self.std_min = s0
        self.std_max = s1

        self.per_channel = bool(per_channel)
        self.clip = bool(clip)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        self._check_input(x)
        B, C, H, W = x.shape
        device = x.device

        mask = self._apply_prob(B, device, generator)
        if not mask.any():
            return x

        out = x.clone()

        # Sample std per sample (and per-channel if requested)
        if self.per_channel:
            u = torch.rand(B, C, device=device, generator=generator)
            std_vals = self.std_min + (self.std_max - self.std_min) * u  # [B,C]
        else:
            u = torch.rand(B, device=device, generator=generator)
            std_vals = self.std_min + (self.std_max - self.std_min) * u  # [B]

        std_b = _broadcast_std(std_vals.to(torch.float32), x.shape, self.per_channel)
        std_b_sel = std_b[mask]

        work_dtype = torch.float32
        noise = torch.empty_like(out[mask], dtype=work_dtype).normal_(mean=0.0, std=1.0, generator=generator)
        noise = noise * std_b_sel.to(work_dtype)

        y = out[mask].to(work_dtype) + noise
        y = y.to(out.dtype)

        if self.clip:
            y = torch.clamp(y, self.clip_min, self.clip_max)

        out[mask] = y
        return out