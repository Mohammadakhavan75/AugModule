from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from ..base import AugBase
from ..functional import gaussian_kernel1d

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

        # per-sample displacements
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

        # separable Gaussian blur
        k1d = gaussian_kernel1d(self.sigma, dtype=y.dtype, device=y.device)
        kx = k1d.view(1, 1, -1).unsqueeze(2).expand(C, 1, -1, 1)
        ky = k1d.view(1, 1, -1).unsqueeze(3).expand(C, 1, -1, 1)  # reuse via transpose
        pad = (k1d.numel() // 2)

        y = F.conv2d(F.pad(y, (pad, pad, 0, 0), mode="reflect"), kx, groups=C)
        y = F.conv2d(F.pad(y, (0, 0, pad, pad), mode="reflect"), ky.transpose(2, 3), groups=C)

        # keep unaugmented samples
        out = x.clone()
        out[mask] = y[mask]
        return out