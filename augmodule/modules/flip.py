from __future__ import annotations
import torch
import torch.nn as nn
from ..base import AugBase

class Flip(AugBase):
    """
    Deterministic flip. Always applies the specified directions.
    - horizontal=True flips along width (dim=3)
    - vertical=True flips along height (dim=2)

    Example:
        Flip(horizontal=True)      # always H-flip
        Flip(vertical=True)        # always V-flip
        Flip(horizontal=True, vertical=True)  # 180° flip (H+V)
    """
    def __init__(self, *, horizontal: bool = True, vertical: bool = False):
        # Deterministic layer: still subclass AugBase just to reuse checks
        super().__init__(p=1.0)
        if not (horizontal or vertical):
            raise ValueError("At least one of horizontal or vertical must be True.")
        self.horizontal = bool(horizontal)
        self.vertical = bool(vertical)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        self._check_input(x)
        out = x
        if self.horizontal:
            out = torch.flip(out, dims=(3,))  # width axis
        if self.vertical:
            out = torch.flip(out, dims=(2,))  # height axis
        return out


class RandomFlip(AugBase):
    """
    Random flip with overall probability `p`. If both directions are allowed,
    per-sample we pick exactly one of {H, V} uniformly (mode='one_of').

    Args:
        horizontal: allow horizontal flip
        vertical:   allow vertical flip
        p:          probability to apply a flip to a given sample
        mode:       'one_of' (default) = choose H or V exclusively when both True
                    'independent' = apply H and/or V independently (can result in both)

    Notes:
        - Works on [B, C, H, W] float tensors on any device.
        - Determinism via passing a torch.Generator.
    """
    def __init__(
        self,
        *,
        horizontal: bool = True,
        vertical: bool = False,
        p: float = 0.5,
        mode: str = "one_of",  # or "independent"
    ):
        super().__init__(p=float(p))
        if not (horizontal or vertical):
            raise ValueError("At least one of horizontal or vertical must be True.")
        if mode not in {"one_of", "independent"}:
            raise ValueError("mode must be 'one_of' or 'independent'.")
        self.horizontal = bool(horizontal)
        self.vertical = bool(vertical)
        self.mode = mode

    @torch.no_grad()
    def forward(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        self._check_input(x)
        B, C, H, W = x.shape
        device = x.device

        # Which samples get augmented at all
        mask = self._apply_prob(B, device, generator)
        if not mask.any():
            return x

        out = x.clone()

        if self.horizontal and self.vertical and self.mode == "one_of":
            # Choose exactly one direction per selected sample
            choice = torch.randint(0, 2, (B,), device=device, generator=generator).bool()
            h_mask = mask & choice           # True → horizontal
            v_mask = mask & (~choice)        # False → vertical
            if h_mask.any():
                out[h_mask] = torch.flip(out[h_mask], dims=(3,))
            if v_mask.any():
                out[v_mask] = torch.flip(out[v_mask], dims=(2,))
            return out

        # Otherwise, apply independently as requested
        if self.horizontal:
            h_take = mask if self.mode == "one_of" else (mask & (torch.randint(0, 2, (B,), device=device, generator=generator).bool()))
            if h_take.any():
                out[h_take] = torch.flip(out[h_take], dims=(3,))
        if self.vertical:
            v_take = mask if self.mode == "one_of" and not self.horizontal else (mask & (torch.randint(0, 2, (B,), device=device, generator=generator).bool()))
            if v_take.any():
                out[v_take] = torch.flip(out[v_take], dims=(2,))

        return out