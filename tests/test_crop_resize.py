import torch
import pytest
from augmodule import RandomCropResize

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
PARAMS = dict(size=(24, 32), scale=(0.5, 1.0), ratio=(0.75, 1.3333333), p=1.0)

@pytest.mark.parametrize("device", DEVICES)
def test_shape_and_device(device):
    x = torch.rand(4, 3, 64, 64, device=device)
    aug = RandomCropResize(**PARAMS).to(device)
    y = aug(x)
    assert y.shape == (4, 3, 24, 32)
    assert y.device == x.device

@pytest.mark.parametrize("device", DEVICES)
def test_p_zero_identity(device):
    x = torch.rand(2, 3, 32, 48, device=device)
    aug = RandomCropResize(size=32, p=0.0).to(device)
    y = aug(x)
    assert torch.allclose(x, y)

def test_determinism_cpu():
    device = "cpu"
    x = torch.rand(3, 3, 40, 60)
    g1 = torch.Generator(device=device).manual_seed(123)
    g2 = torch.Generator(device=device).manual_seed(123)
    aug = RandomCropResize(size=(20, 20), scale=(0.6, 0.9), ratio=(0.7, 1.2), p=1.0)
    y1 = aug(x, generator=g1)
    y2 = aug(x, generator=g2)
    assert torch.allclose(y1, y2)