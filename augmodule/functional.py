import torch

def gaussian_kernel1d(sigma: float, dtype, device):
    radius = max(1, int(3 * float(sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma)**2)
    return (k / k.sum())