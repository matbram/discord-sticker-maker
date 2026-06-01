"""JudgeNet — a tiny CPU CNN that predicts a perceptual distance scalar.

BandingNet-spirit, with the temporal dimension folded into channels (cheap on CPU,
and it directly carries the flicker signal). Input is the ``judge_features`` stack
``(B, T, C, H, W)`` (or pre-flattened ``(B, T*C, H, W)``); output is a single
non-negative distance (softplus). Pairwise training feeds two candidates through
the same weights (Siamese) and compares the scalars.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..judge_features import N_CHANNELS, T_SAMPLES

IN_CH = N_CHANNELS * T_SAMPLES  # temporal folded into channels


class JudgeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        def blk(i: int, o: int, s: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.body = nn.Sequential(
            blk(IN_CH, 32, 2),   # 96 -> 48
            blk(32, 48, 2),      # 48 -> 24
            blk(48, 64, 2),      # 24 -> 12
            blk(64, 64, 2),      # 12 -> 6
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:                 # (B, T, C, H, W) -> (B, T*C, H, W)
            x = x.flatten(1, 2)
        return self.act(self.head(self.body(x))).squeeze(-1)
