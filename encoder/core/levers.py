"""Encoding levers and their ordered ladders.

A ``LeverState`` is one point in the decision space. Each engine's *primary*
ladder is ordered so that a higher index means a bigger file — the monotone
assumption the size search relies on. Resolution (``scale``) is the last-resort
lever: only engaged when nothing else makes a clip fit (spec §7).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class LeverKind(str, Enum):
    COLORS = "colors"     # palette size
    DITHER = "dither"     # dithering mode
    LOSSY = "lossy"       # gifsicle lossy-LZW strength
    QUALITY = "quality"   # gifski quality
    SCALE = "scale"       # resolution multiplier (last resort)


@dataclass(frozen=True)
class LeverState:
    colors: int | None = None
    dither: str | None = None
    lossy: int | None = None
    quality: int | None = None
    scale: float = 1.0

    def with_(self, **kw) -> "LeverState":
        return replace(self, **kw)

    def as_dict(self) -> dict:
        """Compact, JSON-friendly view of the *set* levers (for the report)."""
        d: dict = {}
        if self.colors is not None:
            d["colors"] = self.colors
        if self.dither is not None:
            d["dither"] = self.dither
        if self.lossy is not None:
            d["lossy"] = self.lossy
        if self.quality is not None:
            d["quality"] = self.quality
        if self.scale != 1.0:
            d["scale"] = round(self.scale, 3)
        return d


# Primary ladders — index up => bigger file.
# A fairly fine COLORS ladder so the size search can land close to the byte target
# (using the budget for more colors = less banding) rather than stopping well under
# it because the next coarse step overshot.
FFMPEG_COLORS: tuple[int, ...] = (
    8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 224, 256
)
GIFSKI_QUALITY: tuple[int, ...] = (30, 40, 50, 60, 70, 80, 90, 100)
GIFSICLE_LOSSY: tuple[int, ...] = (200, 160, 120, 90, 60, 40, 20, 0)  # high lossy = small file

# ffmpeg dither modes explored in the quality phase (size-affecting, quality-affecting).
FFMPEG_DITHERS: tuple[str, ...] = ("sierra2_4a", "bayer", "floyd_steinberg", "none")

# Resolution descent (last resort): tried in order when even the lossiest fit overshoots.
SCALE_VALUES: tuple[float, ...] = (1.0, 0.9, 0.8, 0.66, 0.5)
