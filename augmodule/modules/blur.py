from __future__ import annotations
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import AugBase

def _pad_mode(padding_mode: str) -> str:
    if padding_mode == "zeros":
        return "constant"
    if padding_mode in ("reflect", "replicate"):
        return padding_mode
    raise ValueError("padding_mode must be 'zeros', 'reflect', or 'replicate'")

def _gaussian_kernel1d(sigma: float, *, dtype: torch.dtype, device) -> torch.Tensor:
    """Return normalized 1D Gaussian kernel of length 2*radius+1, radius≈3σ (min radius=1)."""
    if sigma <= 0:
        # Identity kernel of size 1
        k = torch.ones(1, dtype=dtype, device=device)
        return k
    radius = max(1, int(round(3.0 * float(sigma))))
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    return k

def _apply_separable_gauss(x: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor, *, padding_mode: str) -> torch.Tensor:
    """
    x: [B,C,H,W], kx: [Kx], ky: [Ky]
    padding_mode: 'zeros'|'reflect'|'replicate'
    """
    B, C, H, W = x.shape
    pm = _pad_mode(padding_mode)

    # Horizontal
    pad_x = kx.numel() // 2
    if pad_x > 0:
        x = F.pad(x, (pad_x, pad_x, 0, 0), mode=pm)
    wx = kx.view(1, 1, 1, -1).to(x.dtype)
    wx = wx.expand(C, 1, 1, -1)                 # [C,1,1,Kx]
    x = F.conv2d(x, wx, groups=C)

    # Vertical
    pad_y = ky.numel() // 2
    if pad_y > 0:
        x = F.pad(x, (0, 0, pad_y, pad_y), mode=pm)
    wy = ky.view(1, 1, -1, 1).to(x.dtype)
    wy = wy.expand(C, 1, -1, 1)                 # [C,1,Ky,1]
    x = F.conv2d(x, wy, groups=C)

    return x

class GaussianBlur(AugBase):
    """
    Deterministic Gaussian blur via separable convolution.

    Args:
        sigma: float (isotropic) or (sigma_y, sigma_x) for anisotropic.
        p: apply probability per-sample (default 1.0).
        padding_mode: 'reflect' (default) | 'replicate' | 'zeros'
    Notes:
        - Works on any device (cpu/cuda/mps), any float dtype.
        - For very small σ, radius becomes 1 → minimal blur. σ<=0 yields identity.
    """
    def __init__(
        self,
        sigma: Union[float, Tuple[float, float]],
        *,
        p: float = 1.0,
        padding_mode: str = "reflect",
    ):
        super().__init__(p=float(p))
        if isinstance(sigma, (tuple, list)):
            if len(sigma) != 2:
                raise ValueError("sigma tuple must be (sigma_y, sigma_x).")
            self.sigma_y = float(sigma[0])
            self.sigma_x = float(sigma[1])
        else:
            s = float(sigma)
            self.sigma_y = s
            self.sigma_x = s
        if padding_mode not in {"reflect", "replicate", "zeros"}:
            raise ValueError("padding_mode must be 'reflect', 'replicate', or 'zeros'.")
        self.padding_mode = padding_mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        self._check_input(x)
        B, C, H, W = x.shape
        mask = self._apply_prob(B, x.device, generator)
        if not mask.any():
            return x

        out = x.clone()
        # Precompute kernels once per forward for the (fixed) σ
        ky = _gaussian_kernel1d(self.sigma_y, dtype=out.dtype, device=out.device)
        kx = _gaussian_kernel1d(self.sigma_x, dtype=out.dtype, device=out.device)

        idx = mask.nonzero(as_tuple=False).flatten()
        y = _apply_separable_gauss(out[idx], kx, ky, padding_mode=self.padding_mode)
        out[idx] = y
        return out

class RandomGaussianBlur(AugBase):
    """
    Random Gaussian blur with per-sample σ.

    Args:
        sigma_range: (min_sigma, max_sigma) for isotropic blur
                     OR ((min_sy,max_sy), (min_sx,max_sx)) for anisotropic.
        p: apply probability per-sample (default 0.5).
        padding_mode: 'reflect' (default) | 'replicate' | 'zeros'
        attempts: ignored (kept for API symmetry if you add constraints later)
    Notes:
        - Different σ per sample → we loop over selected samples for correctness.
          (Still device-friendly; only a small Python loop over masked indices.)
    """
    def __init__(
        self,
        sigma_range: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]],
        *,
        p: float = 0.5,
        padding_mode: str = "reflect",
    ):
        super().__init__(p=float(p))
        self.anisotropic = isinstance(sigma_range[0], (tuple, list))  # type: ignore[index]
        if self.anisotropic:
            sy0, sy1 = float(sigma_range[0][0]), float(sigma_range[0][1])   # type: ignore[index]
            sx0, sx1 = float(sigma_range[1][0]), float(sigma_range[1][1])   # type: ignore[index]
            if not (sy0 >= 0 and sy1 >= sy0 and sx0 >= 0 and sx1 >= sx0):
                raise ValueError("Sigma ranges must satisfy 0 <= min <= max.")
            self.sy_range = (sy0, sy1)
            self.sx_range = (sx0, sx1)
        else:
            s0, s1 = float(sigma_range[0]), float(sigma_range[1])            # type: ignore[index]
            if not (s0 >= 0 and s1 >= s0):
                raise ValueError("Sigma range must satisfy 0 <= min <= max.")
            self.sy_range = (s0, s1)
            self.sx_range = (s0, s1)

        if padding_mode not in {"reflect", "replicate", "zeros"}:
            raise ValueError("padding_mode must be 'reflect', 'replicate', or 'zeros'.")
        self.padding_mode = padding_mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        self._check_input(x)
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        mask = self._apply_prob(B, device, generator)
        if not mask.any():
            return x

        out = x.clone()
        idx = mask.nonzero(as_tuple=False).flatten().tolist()

        # Per-sample σ sampling and blur
        for i in idx:
            # sample sy, sx uniformly in their ranges
            u1 = torch.rand((), device=device, generator=generator).item()
            u2 = torch.rand((), device=device, generator=generator).item()
            sy = self.sy_range[0] + (self.sy_range[1] - self.sy_range[0]) * u1
            sx = self.sx_range[0] + (self.sx_range[1] - self.sx_range[0]) * u2

            if sy <= 0 and sx <= 0:
                # identity
                continue

            ky = _gaussian_kernel1d(sy, dtype=dtype, device=device)
            kx = _gaussian_kernel1d(sx, dtype=dtype, device=device)
            out[i:i+1] = _apply_separable_gauss(out[i:i+1], kx, ky, padding_mode=self.padding_mode)

        return out