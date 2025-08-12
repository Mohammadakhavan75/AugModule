from __future__ import annotations
import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import AugBase

class RandomCropResize(AugBase):
    """
    Randomly crop a sub-rectangle and resize it to (out_h, out_w).

    Args:
        size: int or (out_h, out_w) – output spatial size after resize.
        scale: (min_area, max_area) – crop area as fraction of original image area.
               e.g., (0.08, 1.0) like torchvision.
        ratio: (min_ratio, max_ratio) – aspect ratio h/w; sampled log-uniformly.
               e.g., (3/4, 4/3).
        p: apply probability per-sample.
        attempts: number of tries to find a valid crop before center-crop fallback.
        mode: 'bilinear' or 'nearest' for resizing.
        padding_mode: 'zeros' | 'border' | 'reflection' – used only if sampling goes outside,
                      which shouldn't happen here but kept for safety.
    Notes:
        - Works on [B,C,H,W] tensors, any float dtype, any device (cpu/cuda/mps).
        - Deterministic via passing a torch.Generator to forward(...).
    """
    def __init__(
        self,
        size: int | Tuple[int, int],
        *,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0/4.0, 4.0/3.0),
        p: float = 1.0,
        attempts: int = 10,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
    ):
        super().__init__(p=float(p))
        if isinstance(size, int):
            self.out_h = self.out_w = int(size)
        else:
            if len(size) != 2:
                raise ValueError("size must be int or (H, W)")
            self.out_h, self.out_w = int(size[0]), int(size[1])

        s0, s1 = float(scale[0]), float(scale[1])
        r0, r1 = float(ratio[0]), float(ratio[1])
        if not (0.0 < s0 <= s1 <= 1.0):
            raise ValueError("scale must satisfy 0 < min <= max <= 1.")
        if not (r0 > 0 and r1 > 0 and r0 <= r1):
            raise ValueError("ratio must satisfy 0 < min <= max.")
        if mode not in {"bilinear", "nearest"}:
            raise ValueError("mode must be 'bilinear' or 'nearest'")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError("padding_mode must be 'zeros', 'border', or 'reflection'")
        if attempts < 1:
            raise ValueError("attempts must be >= 1")

        self.scale = (s0, s1)
        self.ratio = (r0, r1)
        self.attempts = int(attempts)
        self.mode = mode
        self.padding_mode = padding_mode

        # precompute logs for log-uniform sampling of aspect ratio
        self._log_r0 = math.log(r0)
        self._log_r1 = math.log(r1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        self._check_input(x)
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        mask = self._apply_prob(B, device, generator)
        if not mask.any():
            # still need to resize if output size differs? No: this op is defined as
            # crop+resize only when applied; when skipped, return identity.
            return x

        # Prepare output and process only selected samples
        out = x.clone()
        idx = mask.nonzero(as_tuple=False).flatten()

        # Build per-sample affine matrices for the masked subset
        thetas = []
        for i in idx.tolist():
            # try to sample a valid crop
            found = False
            area = H * W
            for _ in range(self.attempts):
                target_area = area * (self.scale[0] + (self.scale[1] - self.scale[0]) * torch.rand((), device=device, generator=generator).item())
                # log-uniform aspect ratio
                ar = math.exp(self._log_r0 + (self._log_r1 - self._log_r0) * torch.rand((), device=device, generator=generator).item())

                h = int(round(math.sqrt(target_area / ar)))
                w = int(round(math.sqrt(target_area * ar)))

                if 1 <= w <= W and 1 <= h <= H:
                    y0 = int(torch.randint(0, H - h + 1, (1,), device=device, generator=generator).item())
                    x0 = int(torch.randint(0, W - w + 1, (1,), device=device, generator=generator).item())
                    found = True
                    break

            if not found:
                # Fallback: center crop to the largest square that fits
                min_side = min(H, W)
                h = w = min_side
                y0 = (H - h) // 2
                x0 = (W - w) // 2

            # Compute normalized center and scale for affine_grid (align_corners=False)
            # Crop center in pixel coordinates (0..W-1 / 0..H-1)
            cx = x0 + (w - 1) * 0.5
            cy = y0 + (h - 1) * 0.5

            # Convert center to normalized coords in [-1,1] using pixel-center mapping
            tx = ((cx + 0.5) / W) * 2.0 - 1.0
            ty = ((cy + 0.5) / H) * 2.0 - 1.0

            # Scale factors map output [-1,1] to input crop span: s = crop_size / full_size
            sx = w / W
            sy = h / H

            theta = torch.tensor([[sx, 0.0, tx],
                                  [0.0, sy, ty]], device=device, dtype=dtype)
            thetas.append(theta)

        thetas = torch.stack(thetas, dim=0)  # [M, 2, 3], M = masked count

        # Generate sampling grid and resample masked subset
        grid = F.affine_grid(thetas, size=(thetas.shape[0], C, self.out_h, self.out_w),
                             align_corners=False)
        y = F.grid_sample(x[idx], grid, mode=self.mode, padding_mode=self.padding_mode,
                          align_corners=False)

        out[idx] = y
        return out