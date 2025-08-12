import torch
import pytest
from augmodule import RandomGaussianNoise

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

@pytest.mark.parametrize("device", DEVICES)
def test_random_gaussian_noise_range(device):
    x = torch.zeros(2, 3, 8, 8, device=device) + 0.5
    aug = RandomGaussianNoise(std_range=(0.0, 0.2), p=1.0, per_channel=True).to(device)
    y = aug(x)
    assert y.shape == x.shape
    # Values should be within clamp range [0,1]
    assert (y >= 0).all() and (y <= 1).all()