import torch
import pytest

try:
    from augmodule import JpegCompression
    _HAVE_AUG = True
except Exception:
    _HAVE_AUG = False

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

@pytest.mark.skipif(not _HAVE_AUG, reason="AugModule import failed")
@pytest.mark.parametrize("device", DEVICES)
def test_jpeg_quality_changes(device):
    x = torch.rand(2, 3, 32, 32, device=device)
    aug = JpegCompression(quality=40, p=1.0).to(device)
    y = aug(x)
    assert y.shape == x.shape
    # With low quality we should see some difference
    assert not torch.allclose(x, y)

@pytest.mark.skipif(not _HAVE_AUG, reason="AugModule import failed")
def test_random_jpeg_determinism_cpu():
    device = "cpu"
    x = torch.rand(3, 3, 24, 24, device=device)
    g1 = torch.Generator(device=device).manual_seed(123)
    g2 = torch.Generator(device=device).manual_seed(123)
    aug = JpegCompression((35, 80), p=1.0)
    y1 = aug(x, generator=g1)
    y2 = aug(x, generator=g2)
    assert torch.allclose(y1, y2)