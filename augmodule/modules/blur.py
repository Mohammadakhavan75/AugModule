from __future__ import annotations
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import AugBase
from ..functional import gaussian_kernel1d, pad_mode, apply_separable_gauss

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


class GlassBlur(AugBase):
    """
    Glass blur: small random pixel displacements followed by Gaussian blur.
    Works on any device; pure torch ops. Input: [B,C,H,W] float in [0,1] or [0,255].
    """
    def __init__(self, sigma: float = 0.7, max_delta: int = 1, iters: int = 1, p: float = 1.0):
        super().__init__(p=p)
        self.sigma = float(sigma)
        self.max_delta = int(max_delta)
        self.iters = int(iters)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        self._check_input(x)
        B, C, H, W = x.shape
        mask = self._apply_prob(B, x.device, generator)
        if not mask.any():
            return x
        y = x.clone()

        for _ in range(self.iters):
            dx = torch.randint(-self.max_delta, self.max_delta + 1, (B, 1, H, W),
                               device=x.device, generator=generator)
            dy = torch.randint(-self.max_delta, self.max_delta + 1, (B, 1, H, W),
                               device=x.device, generator=generator)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"
            )
            gx = (grid_x + dx.squeeze(1)).clamp(0, W - 1)
            gy = (grid_y + dy.squeeze(1)).clamp(0, H - 1)
            idx = gy * W + gx
            y = y.view(B, C, H * W).gather(2, idx.view(B, 1, H * W).expand(-1, C, -1)).view(B, C, H, W)

        k1d = gaussian_kernel1d(self.sigma, dtype=y.dtype, device=y.device)
        kx = k1d.view(1, 1, -1).unsqueeze(2).expand(C, 1, -1, 1)
        ky = k1d.view(1, 1, -1).unsqueeze(3).expand(C, 1, -1, 1)
        pad = (k1d.numel() // 2)

        y = F.conv2d(F.pad(y, (pad, pad, 0, 0), mode="reflect"), kx, groups=C)
        y = F.conv2d(F.pad(y, (0, 0, pad, pad), mode="reflect"), ky.transpose(2, 3), groups=C)

        out = x.clone()
        out[mask] = y[mask]
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
