# augmodule/modules/color.py
from __future__ import annotations
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
from ..functional import _to_float, _rgb_to_gray, _adjust_brightness, _adjust_contrast, _adjust_saturation, _adjust_hue_yiq, _ensure_tuple
from ..base import AugBase

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
