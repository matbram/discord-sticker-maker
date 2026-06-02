"""Banding-aware color distance — the term that lets the search be *trusted on color*.

The default judge scores luma MS-SSIM, which is structurally blind to color
banding: it actually *prefers* a banded result over a dithered one, because dither
adds the high-frequency noise MS-SSIM dislikes while the smooth false-contour of
banding barely moves the structural score (the blind spot documented in
``docs/m2-judge.md``). That is the bug that forced the bridge's color-floor bandaid.

The fix rests on one perceptual fact:

  * **Dithering preserves the local *mean* color** — the eye integrates the dither
    pattern back to the true value, so a low-pass of the dithered image matches a
    low-pass of the source.
  * **Banding shifts the local mean in steps** — a posterized ramp's local mean
    jumps at each false contour, so its low-pass deviates from the source.

So we low-pass both images in **OKLab** (a perceptually-uniform space) and measure
the mean ΔE. That term is large for banding and small for dithering — the opposite
of MS-SSIM's mistake — and it sees chroma, which luma MS-SSIM cannot.
"""
from __future__ import annotations

import numpy as np


def _stack_rgba(frames: list[np.ndarray]) -> np.ndarray:
    """List of HxWx{3,4} uint8 frames -> (N, H, W, 4) float, defaulting alpha=255."""
    out = np.empty((len(frames),) + frames[0].shape[:2] + (4,), dtype=np.float64)
    for i, fr in enumerate(frames):
        arr = np.asarray(fr, dtype=np.float64)
        if arr.shape[-1] == 4:
            out[i] = arr
        else:
            out[i, ..., :3] = arr
            out[i, ..., 3] = 255.0
    return out


def rgba_to_oklab(stack_rgba: np.ndarray) -> np.ndarray:
    """(N, H, W, 4) sRGB -> (N, H, W, 3) OKLab, alpha composited over black.

    Compositing over a fixed background makes transparent regions identical between
    reference and candidate, so 1-bit GIF transparency cannot pollute the score
    (mirrors ``composite_luma_stack``).
    """
    rgb = stack_rgba[..., :3] / 255.0
    alpha = stack_rgba[..., 3:4] / 255.0
    rgb = rgb * alpha  # over black

    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]

    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)

    big_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    big_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    big_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([big_l, big_a, big_b], axis=-1)


def box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur over the last two axes (edge-padded), O(N) via cumsum.

    Uses plain slicing (not fancy indexing) for the window difference, which is
    several times faster on big stacks.
    """
    if radius < 1:
        return a
    k = 2 * radius + 1
    for axis in (-2, -1):
        n = a.shape[axis]
        pad = [(0, 0)] * a.ndim
        pad[axis] = (radius, radius)
        ap = np.pad(a, pad, mode="edge")
        cs = np.cumsum(ap, axis=axis)
        zero_shape = list(cs.shape)
        zero_shape[axis] = 1
        cs = np.concatenate([np.zeros(zero_shape, cs.dtype), cs], axis=axis)
        sl_hi = [slice(None)] * cs.ndim
        sl_lo = [slice(None)] * cs.ndim
        sl_hi[axis] = slice(k, k + n)
        sl_lo[axis] = slice(0, n)
        a = (cs[tuple(sl_hi)] - cs[tuple(sl_lo)]) / k
    return a


def _resize_oklab(stack: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear-resize an (N, H, W, 3) OKLab stack to ``target_hw`` (the scale lever)."""
    from PIL import Image

    n, h, w = stack.shape[:3]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return stack
    out = np.empty((n, th, tw, 3), dtype=np.float64)
    for i in range(n):
        for c in range(3):
            im = Image.fromarray(stack[i, ..., c].astype(np.float32), mode="F")
            out[i, ..., c] = np.asarray(im.resize((tw, th), Image.BILINEAR), dtype=np.float64)
    return out


def banding_per_frame(
    reference: list[np.ndarray], candidate: list[np.ndarray], *, radius: int = 2, passes: int = 2
) -> np.ndarray:
    """Per-frame low-pass OKLab ΔE between reference and candidate (banding term).

    Two box-blur passes ≈ a Gaussian wide enough to integrate dither noise (a few
    px) yet narrow enough to keep banding's wide false-contour steps. Returns a
    length-N array of mean ΔE per frame (0 = identical local color).
    """
    ref = rgba_to_oklab(_stack_rgba(reference))
    cand = rgba_to_oklab(_stack_rgba(candidate))
    n = min(ref.shape[0], cand.shape[0])
    ref, cand = ref[:n], cand[:n]
    cand = _resize_oklab(cand, ref.shape[1:3])

    # Blur over spatial axes per (frame, channel): move channel to axis 1.
    rb = np.moveaxis(ref, -1, 1)
    cb = np.moveaxis(cand, -1, 1)
    for _ in range(max(1, passes)):
        rb = box_blur(rb, radius)
        cb = box_blur(cb, radius)
    de = np.sqrt(np.sum((rb - cb) ** 2, axis=1))  # (N, H, W) OKLab ΔE
    return de.reshape(n, -1).mean(axis=1)
