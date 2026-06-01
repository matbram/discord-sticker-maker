"""Parametric degradations with KNOWN severity orderings (no human labels needed).

Each family maps a clean RGBA frame list + a severity ``s in [0,1]`` to a degraded
frame list of the *same* count and size. Because we control ``s``, a monotone chain
``s0 < s1 < ...`` yields reliable pairwise labels ("lower s is closer to source").
``blur`` is a low-frequency negative control so the judge can't simply equate
"more high-frequency detail" with "better".
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _split(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return frame[..., :3].astype(np.float32), frame[..., 3]


def _join(rgb_f: np.ndarray, a: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)
    return np.dstack([rgb, a])


def banding(frames: list[np.ndarray], s: float) -> list[np.ndarray]:
    """Posterize to fewer levels -> false contours in smooth gradients (s=1 worst)."""
    levels = max(2, int(round(2 + (1.0 - s) * 30)))   # s=0 -> 32 levels, s=1 -> 2
    q = 255.0 / (levels - 1)
    out = []
    for f in frames:
        rgb, a = _split(f)
        out.append(_join(np.round(rgb / q) * q, a))
    return out


def flicker(frames: list[np.ndarray], s: float) -> list[np.ndarray]:
    """Per-frame luma jitter -> temporal shimmer in otherwise-static regions."""
    amp = s * 22.0
    rng = np.random.default_rng(12345)
    out = []
    for f in frames:
        rgb, a = _split(f)
        noise = rng.normal(0.0, amp, size=rgb.shape[:2]).astype(np.float32)
        out.append(_join(rgb + noise[..., None], a))
    return out


def choppiness(frames: list[np.ndarray], s: float) -> list[np.ndarray]:
    """Frame-decimate-and-hold -> choppy motion at the same frame count (s=1 worst)."""
    n = len(frames)
    stride = 1 + int(round(s * 4))                    # s=0 -> 1 (identity), s=1 -> 5
    if stride <= 1:
        return [f.copy() for f in frames]
    return [frames[(i // stride) * stride].copy() for i in range(n)]


def blur(frames: list[np.ndarray], s: float) -> list[np.ndarray]:
    """Gaussian blur -> low-frequency washout (negative control vs banding)."""
    radius = s * 2.5
    if radius < 1e-3:
        return [f.copy() for f in frames]
    return [np.asarray(Image.fromarray(f, "RGBA").filter(ImageFilter.GaussianBlur(radius)), np.uint8)
            for f in frames]


FAMILIES = {
    "banding": banding,
    "flicker": flicker,
    "choppiness": choppiness,
    "blur": blur,
}
