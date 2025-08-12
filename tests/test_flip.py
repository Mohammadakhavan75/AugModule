import torch
import pytest
from augmodule import Flip, RandomFlip

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

@pytest.mark.parametrize("device", DEVICES)
def test_flip_deterministic_h(device):
    x = torch.arange(2*1*2*4, dtype=torch.float32, device=device).view(2,1,2,4)
    aug = Flip(horizontal=True, vertical=False).to(device)
    y = aug(x)
    # flipping width twice returns original
    z = torch.flip(y, dims=(3,))
    assert torch.allclose(z, x)

@pytest.mark.parametrize("device", DEVICES)
def test_randomflip_p0_identity(device):
    x = torch.rand(4, 3, 8, 8, device=device)
    aug = RandomFlip(horizontal=True, p=0.0).to(device)
    y = aug(x)
    assert torch.allclose(x, y)

@pytest.mark.parametrize("device", DEVICES)
def test_randomflip_deterministic_with_generator(device):
    x = torch.rand(6, 3, 16, 16, device=device)
    g1 = torch.Generator(device=device).manual_seed(123)
    g2 = torch.Generator(device=device).manual_seed(123)
    aug = RandomFlip(horizontal=True, vertical=True, p=1.0, mode="one_of").to(device)
    y1 = aug(x, generator=g1)
    y2 = aug(x, generator=g2)
    assert torch.allclose(y1, y2)

@pytest.mark.parametrize("device", DEVICES)
def test_randomflip_changes_with_p1(device):
    x = torch.rand(4, 3, 16, 16, device=device)
    aug = RandomFlip(horizontal=True, p=1.0).to(device)
    y = aug(x)
    assert not torch.allclose(x, y)