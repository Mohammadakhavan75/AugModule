import torch
import math
import torch.nn.functional as F

_WEIGHTS = (0.2989, 0.5870, 0.1140)

def gaussian_kernel1d(sigma: float, dtype, device):
    radius = max(1, int(3 * float(sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma)**2)
    return (k / k.sum())

def pad_mode(padding_mode: str) -> str:
    if padding_mode == "zeros":
        return "constant"
    if padding_mode in ("reflect", "replicate"):
        return padding_mode
    raise ValueError("padding_mode must be 'zeros', 'reflect', or 'replicate'")

def apply_separable_gauss(x: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor, *, padding_mode: str) -> torch.Tensor:
    B, C, H, W = x.shape
    pm = pad_mode(padding_mode)

    pad_x = kx.numel() // 2
    if pad_x > 0:
        x = F.pad(x, (pad_x, pad_x, 0, 0), mode=pm)
    wx = kx.view(1, 1, 1, -1).to(x.dtype).expand(C, 1, 1, -1)
    x = F.conv2d(x, wx, groups=C)

    pad_y = ky.numel() // 2
    if pad_y > 0:
        x = F.pad(x, (0, 0, pad_y, pad_y), mode=pm)
    wy = ky.view(1, 1, -1, 1).to(x.dtype).expand(C, 1, -1, 1)
    x = F.conv2d(x, wy, groups=C)

    return x

def _to_float(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype in (torch.float32, torch.float64) else x.to(torch.float32)

def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return r * _WEIGHTS[0] + g * _WEIGHTS[1] + b * _WEIGHTS[2]

def _adjust_brightness(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return x * factor

def _adjust_contrast(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    gray = _rgb_to_gray(x).expand_as(x)
    return gray + factor * (x - gray)

def _adjust_saturation(x: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    gray = _rgb_to_gray(x).expand_as(x)
    return gray * (1.0 - factor) + x * factor

def _adjust_hue_yiq(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    Y = 0.299 * r + 0.587 * g + 0.114 * b
    I = 0.596 * r - 0.274 * g - 0.322 * b
    Q = 0.211 * r - 0.523 * g + 0.312 * b

    theta = delta * (2.0 * math.pi)
    c = torch.cos(theta).view(-1, 1, 1, 1)
    s = torch.sin(theta).view(-1, 1, 1, 1)

    I2 = I * c - Q * s
    Q2 = I * s + Q * c

    R = Y + 0.956 * I2 + 0.621 * Q2
    G = Y - 0.272 * I2 - 0.647 * Q2
    B = Y - 1.106 * I2 + 1.703 * Q2
    return torch.cat([R, G, B], dim=1)

def _ensure_tuple(val, name: str, clip_zero: bool = False, hue: bool = False):
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
