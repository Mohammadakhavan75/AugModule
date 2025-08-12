import torch
import pytest
from augmodule import ColorJitter

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_color_jitter_brightness_only(device):
    x = torch.ones(2, 3, 8, 8, device=device) * 0.5
    aug = ColorJitter(brightness=0.5, contrast=None, saturation=None, hue=None, p=1.0, random_order=False).to(device)
    g = torch.Generator(device=device).manual_seed(0)
    y = aug(x, generator=g)
    assert y.shape == x.shape
    assert not torch.allclose(y, x)


@pytest.mark.parametrize("device", DEVICES)
def test_color_jitter_determinism(device):
    x = torch.rand(4, 3, 8, 8, device=device)
    g1 = torch.Generator(device=device).manual_seed(42)
    g2 = torch.Generator(device=device).manual_seed(42)
    aug = ColorJitter(brightness=0.2, contrast=0.3, saturation=0.4, hue=0.1, p=1.0, random_order=False).to(device)
    y1 = aug(x, generator=g1)
    y2 = aug(x, generator=g2)
    assert torch.allclose(y1, y2)
