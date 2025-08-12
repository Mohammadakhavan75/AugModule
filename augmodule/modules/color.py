# augmodule/modules/color.py
from __future__ import annotations
from typing import Tuple, Optional, List
import math
import torch
import torch.nn as nn
from ..base import AugBase

# Luma weights (ITU-R BT.601) used by torchvision
_WEIGHTS = (0.2989, 0.5870, 0.1140)

def _to_float(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype in (torch.float32, torch.float64) else x.to(torch.float32)

def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: [B,3,H,W] float
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = r * _WEIGHTS[0] + g * _WEIGHTS[1] + b * _WEIGHTS[2]
    return y  # [B,1,H,W]

def _adjust_brightness(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    # factor: [M] or [M,1,1,1] for broadcasting
    return x * factor

def _adjust_contrast(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    # Contrast via grayscale target (per-pixel), matches torchvision
    gray = _rgb_to_gray(x)
    gray = gray.expand_as(x)
    return gray + factor * (x - gray)

def _adjust_saturation(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    gray = _rgb_to_gray(x)
    gray = gray.expand_as(x)
    return gray * (1.0 - factor) + x * factor

def _adjust_hue_yiq(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Hue shift via rotation in the I-Q chroma plane of YIQ.
    delta in [-0.5, 0.5] maps to [-180°, 180°], per torchvision semantics.
    """
    # Compute Y,I,Q from RGB
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    I = 0.596 * r - 0.274 * g - 0.322 * b
    Q = 0.211 * r - 0.523 * g + 0.312 * b

    theta = delta * (2.0 * math.pi)  # radians
    c = torch.cos(theta).view(-1, 1, 1, 1)
    s = torch.sin(theta).view(-1, 1, 1, 1)

    I2 = I * c - Q * s
    Q2 = I * s + Q * c

    # Back to RGB
    R = Y + 0.956 * I2 + 0.621 * Q2
    G = Y - 0.272 * I2 - 0.647 * Q2
    B = Y - 1.106 * I2 + 1.703 * Q2
    out = torch.cat([R, G, B], dim=1)
    return out

def _ensure_tuple(val, name: str, clip_zero: bool = False, hue: bool = False):
    """
    Normalize a jitter spec:
      - float f -> (1-f, 1+f) (for brightness/contrast/saturation),
      - float h -> (-h, +h) for hue (capped to [-0.5, 0.5]),
      - tuple stays as-is.
    """
    if val is None:
        return None
    if isinstance(val, (tuple, list)):
        if len(val) != 2:
            raise ValueError(f"{name} tuple must be (min, max)")
        lo, hi = float(val[0]), float(val[1])
    else:
        f = float(val)
        if hue:
            lo, hi = -f, +f
        else:
            lo, hi = (max(0.0, 1.0 - f), 1.0 + f)
    if hue:
        lo = max(-0.5, lo); hi = min(0.5, hi)
    if not hue and clip_zero:
        lo = max(0.0, lo)
    if hi < lo:
        raise ValueError(f"{name} range must satisfy min <= max, got ({lo}, {hi}).")
    return (lo, hi)

class ColorJitter(AugBase):
    """
    Randomly jitter brightness, contrast, saturation, and hue (like torchvision) as a layer.

    Args:
        brightness: float or (min, max). float f -> factor ∈ [1-f, 1+f].
        contrast:   float or (min, max). float f -> factor ∈ [1-f, 1+f].
        saturation: float or (min, max). float f -> factor ∈ [1-f, 1+f].
        hue:        float or (min, max) in [-0.5, 0.5]. float h -> delta ∈ [-h, +h].
        p:          probability to apply per sample.
        random_order: if True, shuffle operation order per sample; otherwise fixed
                      order: brightness → contrast → saturation → hue.
        clip:       clamp to [0,1] at the end (default True).
    Notes:
        - Works on [B, C, H, W]; hue/saturation only affect C==3; safely bypassed for C!=3.
        - Deterministic with a provided torch.Generator.
    """
    def __init__(
        self,
        brightness: Optional[float | Tuple[float, float]] = None,
        contrast:   Optional[float | Tuple[float, float]] = None,
        saturation: Optional[float | Tuple[float, float]] = None,
        hue:        Optional[float | Tuple[float, float]] = None,
        *,
        p: float = 0.8,
        random_order: bool = True,
        clip: bool = True,
    ):
        super().__init__(p=float(p))
        self.brightness = _ensure_tuple(brightness, "brightness", clip_zero=True)
        self.contrast   = _ensure_tuple(contrast,   "contrast",   clip_zero=True)
        self.saturation = _ensure_tuple(saturation, "saturation", clip_zero=True)
        self.hue        = _ensure_tuple(hue,        "hue",        hue=True)
        if all(v is None for v in (self.brightness, self.contrast, self.saturation, self.hue)):
            raise ValueError("At least one of brightness/contrast/saturation/hue must be set.")
        self.random_order = bool(random_order)
        self.clip = bool(clip)

        # fixed op order when random_order=False
        self._ops: List[str] = []
        if self.brightness is not None: self._ops.append("brightness")
        if self.contrast   is not None: self._ops.append("contrast")
        if self.saturation is not None: self._ops.append("saturation")
        if self.hue        is not None: self._ops.append("hue")

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

        # Pre-sample all factors for each selected sample
        def sample_range(rng: Tuple[float, float] | None, size=1):
            if rng is None:
                return None
            lo, hi = rng
            return lo + (hi - lo) * torch.rand(size, device=device, generator=generator)

        b_f = sample_range(self.brightness, len(idx))
        c_f = sample_range(self.contrast,   len(idx))
        s_f = sample_range(self.saturation, len(idx))
        h_d = sample_range(self.hue,        len(idx))

        # Loop per selected sample to allow random op order
        for j, i in enumerate(idx):
            y = _to_float(out[i:i+1])  # [1,C,H,W] compute in fp32

            # choose operation order
            if self.random_order and len(self._ops) > 1:
                # Fisher–Yates style: sample a random permutation via torch.randperm
                order_idx = torch.randperm(len(self._ops), device=device, generator=generator).tolist()
                ops = [self._ops[k] for k in order_idx]
            else:
                ops = self._ops

            for op in ops:
                if op == "brightness":
                    factor = b_f[j].view(1,1,1,1)  # scalar → broadcast
                    y = _adjust_brightness(y, factor)
                elif op == "contrast":
                    factor = c_f[j].view(1,1,1,1)
                    if C == 3:
                        y = _adjust_contrast(y, factor)
                    else:
                        # Fallback for grayscale: contrast around mean per image
                        mean = y.mean(dim=(2,3), keepdim=True)
                        y = mean + factor * (y - mean)
                elif op == "saturation":
                    if C == 3:
                        factor = s_f[j].view(1,1,1,1)
                        y = _adjust_saturation(y, factor)
                    # else: no-op for single-channel
                elif op == "hue":
                    if C == 3:
                        delta = h_d[j].view(1)  # hue delta per image
                        y = _adjust_hue_yiq(y, delta)
                    # else: no-op

            if self.clip:
                y = y.clamp(0.0, 1.0)

            out[i:i+1] = y.to(dtype)

        return out