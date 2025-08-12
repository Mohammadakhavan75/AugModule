import math
import torch
import pytest
from augmodule import Rotate, RandomRotate

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

@pytest.mark.parametrize("device", DEVICES)
def test_rotate_arbitrary_identity_multiples(device):
    x = torch.rand(2,3,16,16, device=device)
    for deg, k in [(0,0),(90,1),(180,2),(270,3)]:
        aug = Rotate(degrees=deg).to(device)
        y = aug(x)
        z = torch.rot90(x, k=k, dims=(2,3))
        assert torch.allclose(y, z, atol=0, rtol=0)

@pytest.mark.parametrize("device", DEVICES)
def test_random_rotate_determinism(device):
    x = torch.rand(4,3,16,16, device=device)
    g1 = torch.Generator(device=device).manual_seed(42)
    g2 = torch.Generator(device=device).manual_seed(42)
    aug = RandomRotate(degrees=(-15, 15), p=1.0).to(device)
    y1 = aug(x, generator=g1)
    y2 = aug(x, generator=g2)
    assert torch.allclose(y1, y2)