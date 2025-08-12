# AugModule

**nn.Module-based, device-portable image augmentations for PyTorch**  
Build augmentation pipelines like layers, move them to `cpu`/`cuda`/`mps` with `.to(device)`, and drop them into training loops without special casing.

> Complements `torchvision` by offering augmentation *modules* instead of transform *callables*—so you can compose them in `nn.Sequential`, register them in your model, and control them like any other layer.

---

## ✨ Features

- **Layer-style API:** Every augmentation is an `nn.Module` with `forward(x)` → `x_aug`.
- **Device portable:** `.to("cpu"|"cuda"|"mps")` works end-to-end; no host <-> device ping-pong.
- **Batch-first:** Operates on `B×C×H×W` tensors (float in `[0,1]` or `[0,255]`).
- **Deterministic when needed:** Optional `torch.Generator` seeding per call or per module.
- **JIT/AMP friendly:** Pure-PyTorch ops; plays well with mixed precision & acceleration.  
- **Composable:** Use in `nn.Sequential`, Lightning Modules, or your custom `forward`.

---

## Quickstart

### 1) Install

```bash
# Option A: from GitHub (recommended while under active dev)
pip install "git+https://github.com/<you-or-org>/AugModule.git"

# Option B: (planned) from PyPI once published
# pip install augmodule
```

### 2) Use it like layers

```python
import torch
import torch.nn as nn
from augmodule import RandomGaussianBlur, RandomSolarize, RandomFlip

device = "cuda" if torch.cuda.is_available() else "cpu"

augment = nn.Sequential(
    RandomFlip(p=0.5, horizontal=True, vertical=False),
    RandomGaussianBlur(sigma=(0.1, 2.0), p=0.3),
    RandomSolarize(threshold=0.5, p=0.2),
).to(device)

images = torch.rand(32, 3, 224, 224, device=device)  # [B,C,H,W] in [0,1]
images_aug = augment(images)
```

### 3) Drop into your model

```python
class ModelWithAug(nn.Module):
    def __init__(self, backbone, augment):
        super().__init__()
        self.augment = augment                # registered as submodule
        self.backbone = backbone

    def forward(self, x):
        if self.training:
            x = self.augment(x)
        return self.backbone(x)

model = ModelWithAug(backbone=..., augment=augment).to(device)
```

---

## Why AugModule?

- **One mental model:** Everything is a layer—move devices, save/load state, toggle train/eval.
- **Speed:** Avoids CPU-only transforms that force extra copies or host bottlenecks.
- **Control:** Augmentations live in your module tree (e.g., freeze, swap, inspect).
- **Reproducibility:** Module-level RNG control and optional parameter logging.

---

## Available modules (initial set)

- `RandomFlip` – horizontal/vertical flips
- `RandomGaussianBlur` – separable Gaussian blur with per-sample sigma
- `RandomSolarize` – threshold-based solarization
- `RandomBrightnessContrast` – affine intensity transform in logit / linear space
- `RandomPerspective` – projective warp
- `RandomCutout` – zero/mean patches (a.k.a. erasing)
- `GlassBlur` – “glass” displacement + blur (CPU & CUDA variants when supported)

> Tip: Each module accepts `p` (apply probability) and supports per-sample randomness in a batched call.

---

## API pattern

All modules follow this shape:

```python
class SomeAug(nn.Module):
    def __init__(self, ..., p: float = 1.0):
        super().__init__()
        self.p = p
        # register buffers for constants / kernels if needed
        # register parameters if augmentation is learnable

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """
        x: [B,C,H,W] float tensor on any device.
        generator: optional RNG for determinism.
        """
        # sample masks / params
        # apply out-of-place ops and return x_aug
```

**Conventions**

- `@torch.no_grad()` is used by default (augmentations are usually not learned).  
  Remove it if you want gradients through the op.
- Expect `float32` input; modules internally cast as needed and restore dtype.
- Randomness: per-sample when meaningful (e.g., each image gets its own sigma).

---

## Example: `GlassBlur` as a module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlassBlur(nn.Module):
    """
    Glass blur: small random pixel displacements followed by Gaussian blur.
    Works on any device; uses torch ops only.
    """
    def __init__(self, sigma: float = 0.7, max_delta: int = 1, iters: int = 1, p: float = 1.0):
        super().__init__()
        self.sigma = float(sigma)
        self.max_delta = int(max_delta)
        self.iters = int(iters)
        self.p = float(p)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.p < 1.0:
            mask = torch.rand(x.shape[0], device=x.device, generator=generator) < self.p
            if not mask.any():
                return x
            x = x.clone()
            x[~mask] = x[~mask]
        B, C, H, W = x.shape

        # Random small swaps (displacements)
        for _ in range(self.iters):
            dx = torch.randint(-self.max_delta, self.max_delta + 1, (B, 1, H, W),
                               device=x.device, generator=generator)
            dy = torch.randint(-self.max_delta, self.max_delta + 1, (B, 1, H, W),
                               device=x.device, generator=generator)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"
            )
            grid_x = (grid_x + dx.squeeze(1)).clamp(0, W - 1)
            grid_y = (grid_y + dy.squeeze(1)).clamp(0, H - 1)
            idx = grid_y * W + grid_x
            x = x.view(B, C, H * W).gather(2, idx.view(B, 1, H * W).expand(-1, C, -1)).view(B, C, H, W)

        # Gaussian blur via separable conv
        radius = max(1, int(3 * self.sigma))
        t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        kernel1d = torch.exp(-0.5 * (t / self.sigma) ** 2)
        kernel1d = (kernel1d / kernel1d.sum()).view(1, 1, -1)
        pad = (radius, radius)

        # horizontal
        x = F.conv2d(F.pad(x, (pad[0], pad[1], 0, 0), mode="reflect"),
                     kernel1d.expand(C, 1, -1).unsqueeze(2), groups=C)
        # vertical
        x = F.conv2d(F.pad(x, (0, 0, pad[0], pad[1]), mode="reflect"),
                     kernel1d.transpose(1, 2).expand(C, 1, -1).unsqueeze(3), groups=C)

        return x
```

---

## Determinism & seeding

```python
g = torch.Generator(device=device).manual_seed(123)
images_aug_1 = augment(images, generator=g)
images_aug_2 = augment(images, generator=g)  # different, because generator advanced
```

For **repeatable** batches, pass a freshly seeded `Generator` per batch or set module-local generators.

---

## Data types & ranges

- Input: `float32` `[0,1]` (preferred).  
  If you pass `[0,255]`, normalize inside your pipeline (e.g., `x/255.`) and re-scale later if needed.
- Color order: `C=3` expects RGB.
- Shapes: `B×C×H×W`; per-sample randomness is broadcast across `B`.

---

## Integration patterns

**With torchvision transforms**

```python
from torchvision.transforms.v2 import ToDtype, Normalize

pipeline = nn.Sequential(
    RandomFlip(p=0.5),
    RandomGaussianBlur(p=0.2),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
)
```

**With PyTorch Lightning**

Register the augment module in `LightningModule` and apply in `training_step` or inside `forward` under `self.training`.

---

## Repo layout (suggested)

```
AugModule/
  augmodule/
    __init__.py
    base.py            # common utilities: RNG helpers, param sampling
    functional.py      # low-level ops (pure functions)
    modules/
      flip.py
      blur.py
      solarize.py
      perspective.py
      glass_blur.py
  tests/
    test_flip.py
    test_blur.py
  examples/
    contrastive_learning.py
    classification_with_aug.py
  benchmarks/
    speed_cpu_cuda_mps.ipynb
  README.md
  pyproject.toml / setup.cfg
  LICENSE
```

---

## Contributing

1. **Open a discussion/issue** with a short spec (name, parameters, edge cases).
2. **Add a module** under `augmodule/modules/`, following the API pattern above.
3. **Write tests** (`pytest`) covering shapes, dtype, device, determinism, and probability `p`.
4. **Benchmarks** (optional) for CPU/CUDA/MPS.
5. **PR checklist**
   - [ ] Type hints & docstrings
   - [ ] Tests pass on CPU
   - [ ] GPU/MPS smoke test (if available)
   - [ ] README entry & example

Run tests:

```bash
pip install -e ".[dev]"
pytest -q
```

---

## Roadmap

- [ ] More geometric ops: rotate, affine, elastic deformation  
- [ ] Color ops: hue/sat, gamma, posterize, JPEG artifacts  
- [ ] Video augmentations with temporal coherence  
- [ ] Learnable augmentations (trainable params)  
- [ ] TorchScript / compile (`torch.compile`) test matrix  
- [ ] Mixed precision test coverage (AMP)

---

## FAQ

**Q: How is this different from `torchvision` / `kornia`?**  
A: We focus on *layer-style modules* that live inside your model and move with `.to(device)`. This complements callable-style transforms and aims for minimal friction in model code.

**Q: Gradients through augmentations?**  
A: Most modules are wrapped in `no_grad` since augmentation isn’t learned. If you need gradients (e.g., for adversarial training), drop the decorator or add a flag.

**Q: MPS support?**  
A: Modules use core tensor ops that generally work on MPS. Some kernels may hit missing ops; tests will mark those and default to CPU if necessary.

---

## License

MIT © You/Your Org

---

## Citation

If this helps your research, consider citing (template to update upon release):

```bibtex
@software{augmodule_2025,
  title = {AugModule: nn.Module-based, device-portable augmentations for PyTorch},
  author = {<Your Name>},
  year = {2025},
  url = {https://github.com/<you-or-org>/AugModule}
}
```
