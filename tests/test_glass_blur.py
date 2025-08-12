import torch
import pytest
from augmodule import GlassBlur

@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_shape_identity(device):
    x = torch.rand(4, 3, 32, 32, device=device)
    aug = GlassBlur(p=0.0).to(device)
    y = aug(x)
    assert y.shape == x.shape
    assert torch.allclose(x, y)

@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_changes_when_p1(device):
    x = torch.rand(2, 3, 32, 32, device=device)
    aug = GlassBlur(p=1.0).to(device)
    y = aug(x)
    assert not torch.allclose(x, y)

def test_determinism_cpu():
    g = torch.Generator().manual_seed(123)
    x = torch.rand(2, 3, 32, 32)
    aug = GlassBlur(p=1.0)
    y1 = aug(x, generator=g)
    g = torch.Generator().manual_seed(123)
    y2 = aug(x, generator=g)
    assert torch.allclose(y1, y2)