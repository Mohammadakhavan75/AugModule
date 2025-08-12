from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import AugBase

# ---------- helpers ----------

def _make_rotation_grid(
    H: int,
    W: int,
    angles_rad: torch.Tensor,  # [B]
    device,
    dtype,
) -> torch.Tensor:
    """
    Build a [B, H, W, 2] sampling grid (normalized coords in [-1,1]) that rotates
    around the image center by angles_rad (counter-clockwise).
    """
    B = angles_rad.shape[0]

    # Base grid in normalized coordinates centered at 0
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
    gx = gx.unsqueeze(0)  # [1,H,W]
    gy = gy.unsqueeze(0)  # [1,H,W]

    cos = torch.cos(angles_rad).view(B, 1, 1)
    sin = torch.sin(angles_rad).view(B, 1, 1)

    # Rotate (x, y) -> (x', y')
    x_prime = gx * cos - gy * sin
    y_prime = gx * sin + gy * cos

    grid = torch.stack((x_prime, y_prime), dim=-1)  # [B,H,W,2]
    return grid


def _maybe_rot90(x: torch.Tensor, angle_deg: float) -> torch.Tensor | None:
    """Fast path for exact multiples of 90° (tolerates tiny float noise)."""
    a = angle_deg % 360.0
    eps = 1e-6
    if abs(a - 0.0) < eps:
        return x
    if abs(a - 90.0) < eps:
        return torch.rot90(x, k=1, dims=(2, 3))  # 90° CCW
    if abs(a - 180.0) < eps:
        return torch.rot90(x, k=2, dims=(2, 3))
    if abs(a - 270.0) < eps:
        return torch.rot90(x, k=3, dims=(2, 3))
    return None


# ---------- deterministic modules ----------

class Rotate(AugBase):
    """
    Deterministic rotation by an arbitrary angle (in degrees, counter-clockwise).
    Uses grid_sample (bilinear or nearest). Keeps HxW; areas outside are padded.

    Args:
        degrees: float (degrees CCW). Negative -> clockwise.
        mode: 'bilinear' (default) or 'nearest'
        padding_mode: 'zeros' | 'border' | 'reflection'
    """
    def __init__(
        self,
        degrees: float,
        *,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ):
        super().__init__(p=1.0)
        self.degrees = float(degrees)  # CCW
        if mode not in {"bilinear", "nearest"}:
            raise ValueError("mode must be 'bilinear' or 'nearest'")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
        self.mode = mode
        self.padding_mode = padding_mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        self._check_input(x)
        B, C, H, W = x.shape

        # Fast exact path for multiples of 90°
        fast = _maybe_rot90(x, self.degrees % 360.0)
        if fast is not None:
            return fast

        angles = torch.full((B,), math.radians(self.degrees), device=x.device, dtype=x.dtype)
        grid = _make_rotation_grid(H, W, angles, device=x.device, dtype=x.dtype)
        return F.grid_sample(
            x, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=False
        )


# ---------- stochastic module ----------

class RandomRotate(AugBase):
    """
    Random rotation with per-sample angles drawn uniformly from [min_deg, max_deg] (CCW).
    Applied with probability `p` per sample.

    Args:
        degrees: (min_deg, max_deg) in degrees (CCW). E.g., (-30, 30)
        p: probability to rotate a given sample
        mode: 'bilinear' | 'nearest'
        padding_mode: 'zeros' | 'border' | 'reflection'
    """
    def __init__(
        self,
        degrees: Tuple[float, float],
        *,
        p: float = 0.5,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ):
        super().__init__(p=float(p))
        if not (isinstance(degrees, (tuple, list)) and len(degrees) == 2):
            raise ValueError("degrees must be a (min_deg, max_deg) tuple")
        d0, d1 = float(degrees[0]), float(degrees[1])
        if d1 < d0:
            raise ValueError("degrees: max must be >= min")
        self.min_deg, self.max_deg = d0, d1

        if mode not in {"bilinear", "nearest"}:
            raise ValueError("mode must be 'bilinear' or 'nearest'")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
        self.mode = mode
        self.padding_mode = padding_mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        self._check_input(x)
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        mask = self._apply_prob(B, device, generator)
        if not mask.any():
            return x

        # Sample degrees per sample (even for unmasked, we won't use them)
        u = torch.rand(B, device=device, generator=generator)
        deg = self.min_deg + (self.max_deg - self.min_deg) * u  # [B]
        # Optimize exact multiples of 90° where possible
        out = x.clone()
        # For masked subset: build grid and sample
        angles_rad = deg[mask].to(dtype).mul(math.pi / 180.0)  # CCW in radians
        grid = _make_rotation_grid(H, W, angles_rad, device=device, dtype=dtype)
        y = F.grid_sample(x[mask], grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=False)
        out[mask] = y

        return out