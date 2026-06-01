"""Pure-numpy Multi-Scale SSIM (Wang et al.) over luminance.

No scipy/skimage dependency: the Gaussian window is applied as a separable
1-D 'valid' convolution. Operates on a stack of luma frames ``(N, H, W)`` and
returns a per-frame MS-SSIM in [0, 1] (1 == identical).

For small frames (tiny stickers) the number of scales is reduced so the deepest
scale still fits the window; when a frame is smaller than the window we fall back
to a single global-statistics SSIM rather than crash.
"""
from __future__ import annotations

import numpy as np

# Standard 5-scale MS-SSIM weights (Wang, Simoncelli & Bovik 2003).
DEFAULT_WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)
_L = 255.0
_C1 = (0.01 * _L) ** 2
_C2 = (0.03 * _L) ** 2
_EPS = 1e-8


def _gauss_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _conv1d_valid(a: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Separable 'valid' 1-D convolution of ``a`` with kernel ``k`` along ``axis``."""
    a = np.moveaxis(a, axis, -1)
    out_len = a.shape[-1] - len(k) + 1
    acc = np.zeros(a.shape[:-1] + (out_len,), dtype=np.float64)
    for i, ki in enumerate(k):
        acc += ki * a[..., i:i + out_len]
    return np.moveaxis(acc, -1, axis)


def _blur(stack: np.ndarray, k: np.ndarray) -> np.ndarray:
    return _conv1d_valid(_conv1d_valid(stack, k, axis=-1), k, axis=-2)


def _ssim_components(x: np.ndarray, y: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-frame mean contrast-structure ``cs`` and luminance ``l`` maps."""
    mu_x = _blur(x, k)
    mu_y = _blur(y, k)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = _blur(x * x, k) - mu_x2
    sigma_y2 = _blur(y * y, k) - mu_y2
    sigma_xy = _blur(x * y, k) - mu_xy
    cs = (2.0 * sigma_xy + _C2) / (sigma_x2 + sigma_y2 + _C2)
    lum = (2.0 * mu_xy + _C1) / (mu_x2 + mu_y2 + _C1)
    return cs.mean(axis=(1, 2)), lum.mean(axis=(1, 2))


def _downsample(stack: np.ndarray) -> np.ndarray:
    """Non-overlapping 2x2 average pool (crops odd edge)."""
    n, h, w = stack.shape
    h2, w2 = (h // 2) * 2, (w // 2) * 2
    s = stack[:, :h2, :w2]
    return s.reshape(n, h2 // 2, 2, w2 // 2, 2).mean(axis=(2, 4))


def _global_ssim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Single-window SSIM over the whole frame — fallback for sub-window sizes."""
    mu_x = x.mean(axis=(1, 2))
    mu_y = y.mean(axis=(1, 2))
    var_x = x.var(axis=(1, 2))
    var_y = y.var(axis=(1, 2))
    cov = ((x - mu_x[:, None, None]) * (y - mu_y[:, None, None])).mean(axis=(1, 2))
    lum = (2 * mu_x * mu_y + _C1) / (mu_x ** 2 + mu_y ** 2 + _C1)
    cs = (2 * cov + _C2) / (var_x + var_y + _C2)
    return np.clip(lum * cs, 0.0, 1.0)


def _max_scales(h: int, w: int, win: int, max_scales: int = 5) -> int:
    scales = 1
    side = min(h, w)
    while scales < max_scales and (side // 2) >= win:
        side //= 2
        scales += 1
    return scales


def msssim_per_frame(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Per-frame MS-SSIM for two luma stacks ``(N, H, W)`` -> ``(N,)`` in [0, 1]."""
    ref = ref.astype(np.float64, copy=False)
    cand = cand.astype(np.float64, copy=False)
    _, h, w = ref.shape

    win = min(11, h, w)
    if win % 2 == 0:
        win -= 1
    if win < 3 or min(h, w) < 3:
        return _global_ssim(ref, cand)

    k = _gauss_kernel(win, sigma=1.5)
    scales = _max_scales(h, w, win)
    weights = DEFAULT_WEIGHTS[:scales]
    weights = weights / weights.sum()

    x, y = ref, cand
    cs_per_scale: list[np.ndarray] = []
    lum_last: np.ndarray | None = None
    for s in range(scales):
        cs, lum = _ssim_components(x, y, k)
        cs_per_scale.append(np.clip(cs, _EPS, None))
        lum_last = np.clip(lum, _EPS, None)
        if s < scales - 1:
            x, y = _downsample(x), _downsample(y)

    out = np.ones(ref.shape[0], dtype=np.float64)
    for s in range(scales):
        if s < scales - 1:
            out *= cs_per_scale[s] ** weights[s]
        else:
            out *= (lum_last * cs_per_scale[s]) ** weights[s]
    return np.clip(out, 0.0, 1.0)
