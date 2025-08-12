import torch
import pytest
from augmodule import GaussianBlur, RandomGaussianBlur

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_gaussian_blur_identity(device):
    x = torch.rand(2, 3, 16, 16, device=device)
    aug = GaussianBlur(sigma=0.0, p=1.0).to(device)
    y = aug(x)
    assert torch.allclose(y, x)


@pytest.mark.parametrize("device", DEVICES)
def test_random_gaussian_blur_determinism(device):
    x = torch.rand(4, 3, 16, 16, device=device)
    g1 = torch.Generator(device=device).manual_seed(42)
    g2 = torch.Generator(device=device).manual_seed(42)
    aug = RandomGaussianBlur(sigma_range=(0.0, 2.0), p=1.0).to(device)
    y1 = aug(x, generator=g1)
    y2 = aug(x, generator=g2)
    assert torch.allclose(y1, y2)
