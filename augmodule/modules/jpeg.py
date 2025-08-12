from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn

from ..base import AugBase

# Try torchvision backend first (fast, no PIL). Fallback to PIL if missing.
_TV_AVAILABLE = False
try:
    from torchvision.io import encode_jpeg, decode_jpeg
    _TV_AVAILABLE = True
except Exception:
    _TV_AVAILABLE = False

# Optional Pillow fallback
_PIL_AVAILABLE = False
if not _TV_AVAILABLE:
    try:
        from PIL import Image
        import numpy as np
        from io import BytesIO
        _PIL_AVAILABLE = True
    except Exception:
        _PIL_AVAILABLE = False

# Severity → JPEG quality mapping (lower quality = more artifacts)
_SEVERITY_QUALITY = [80, 65, 58, 50, 40]


def _to_uint8_cpu(x: torch.Tensor) -> torch.Tensor:
    """
    Expect x in [0,1] float. Return uint8 [C,H,W] on CPU.
    Caller ensures shape is [C,H,W].
    """
    x_u8 = (x.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return x_u8.cpu()  # encode_jpeg expects CPU tensor


def _from_uint8_like(x_u8: torch.Tensor, device, dtype) -> torch.Tensor:
    """Convert uint8 [C,H,W] to requested device/dtype in [0,1] range."""
    x = x_u8.to(torch.float32) / 255.0
    return x.to(device=device, dtype=dtype)


def _jpeg_one_torchvision(x_u8_CHW: torch.Tensor, quality: int) -> torch.Tensor:
    """
    x_u8_CHW: uint8 [C,H,W] CPU
    returns:  uint8 [C,H,W] CPU
    """
    # encode -> bytes tensor -> decode. decode_jpeg returns CHW uint8.
    encoded = encode_jpeg(x_u8_CHW, quality=quality)
    decoded = decode_jpeg(encoded)  # CHW, uint8
    return decoded


def _jpeg_one_pillow(x_u8_CHW: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Fallback using Pillow. x_u8_CHW is uint8 [C,H,W] CPU.
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError("Neither torchvision.io nor Pillow are available for JPEG compression.")

    c, h, w = x_u8_CHW.shape
    if c not in (1, 3):
        raise ValueError(f"JPEG expects 1 or 3 channels, got {c}")

    # Build HWC numpy array
    arr = x_u8_CHW.permute(1, 2, 0).contiguous().numpy()
    if c == 1:
        pil_img = Image.fromarray(arr.squeeze(-1), mode="L")
    else:
        pil_img = Image.fromarray(arr, mode="RGB")

    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf).convert("L" if c == 1 else "RGB")  # ensure mode matches
    out_arr = np.asarray(out)
    if c == 1:
        out_arr = out_arr[..., None]
    out_CHW = torch.from_numpy(out_arr).permute(2, 0, 1).contiguous()  # uint8 CHW
    return out_CHW


def _jpeg_one(x_u8_CHW: torch.Tensor, quality: int) -> torch.Tensor:
    if _TV_AVAILABLE:
        return _jpeg_one_torchvision(x_u8_CHW, quality)
    return _jpeg_one_pillow(x_u8_CHW, quality)


class _JpegBase(AugBase):
    """Common checks & helpers for JPEG modules."""
    def __init__(self, p: float):
        super().__init__(p=float(p))

    @staticmethod
    def _check_range01(x: torch.Tensor):
        # JPEG round-trips assume [0,1] inputs; enforce for consistency
        if x.min() < 0.0 or x.max() > 1.0:
            lo = float(x.min())
            hi = float(x.max())
            raise ValueError(f"Input tensor values must be in [0,1], got [{lo:.3f}, {hi:.3f}].")

    @staticmethod
    def _check_channels(x: torch.Tensor):
        c = x.shape[1]
        if c not in (1, 3):
            raise ValueError(f"JPEG supports C=1 (grayscale) or C=3 (RGB). Got C={c}.")


class JpegCompression(_JpegBase):
    """
    Apply JPEG compression with per-sample quality sampled uniformly from [q_min, q_max] (integers).
    Args:
        quality_range: (q_min, q_max), 1..100
        p:             probability per sample
    """
    def __init__(self, quality_range: Tuple[int, int], *, p: float = 0.5):
        super().__init__(p=p)
        if not (isinstance(quality_range, (tuple, list)) and len(quality_range) == 2):
            raise ValueError("quality_range must be a (q_min, q_max) tuple.")
        q0, q1 = int(quality_range[0]), int(quality_range[1])
        if not (1 <= q0 <= 100 and 1 <= q1 <= 100 and q0 <= q1):
            raise ValueError("quality_range values must be in [1,100] and q_min <= q_max.")
        self.q_min, self.q_max = q0, q1

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        self._check_input(x)
        self._check_range01(x)
        self._check_channels(x)

        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        mask = self._apply_prob(B, device, generator)
        if not mask.any():
            return x

        # Sample integer qualities per sample (we'll only use masked ones)
        # Use torch.randint to honor the provided generator
        qualities = torch.randint(self.q_min, self.q_max + 1, (B,), device=device, generator=generator)

        out = x.clone()
        idx = mask.nonzero(as_tuple=False).flatten()
        for i in idx.tolist():
            q = int(qualities[i].item())
            xi = x[i]
            xi_u8 = _to_uint8_cpu(xi)
            yi_u8 = _jpeg_one(xi_u8, q)
            yi = _from_uint8_like(yi_u8, device=device, dtype=dtype)
            out[i] = yi

        return out