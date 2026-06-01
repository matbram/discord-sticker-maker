"""Shared input-feature builder for the learned judge (M2).

Both the training data pipeline and the runtime ``LearnedMetric`` turn an aligned
``(reference, candidate)`` frame pair into the *same* model input tensor here, so
there is no train/inference skew. Pure numpy + Pillow — importable without torch,
so the runtime (onnxruntime) needs no training stack.

Design: a small set of perceptually-motivated channels at a downscaled proxy,
sampled at ``T`` evenly-spaced frames. The temporal-delta channels carry the
flicker/choppiness signal (the difference of candidate vs reference deltas is the
same quantity ``metrics/temporal.py`` is built around); the chroma-error channels
let the judge see color washout that luma-only MS-SSIM is blind to.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

PROXY = 96           # in-loop proxy resolution (CPU-cheap, full-res only for reports)
T_SAMPLES = 5        # evenly-spaced frames sampled per clip
N_CHANNELS = 7       # ref-Y, cand-Y, |Δ|Y, cand-ΔtY, ref-ΔtY, ΔCb, ΔCr

# BT.601 luma/chroma from linear-ish sRGB in [0,1]; Cb/Cr centered at 0.
_RGB2Y = np.array([0.299, 0.587, 0.114], dtype=np.float32)
_RGB2CB = np.array([-0.168736, -0.331264, 0.5], dtype=np.float32)
_RGB2CR = np.array([0.5, -0.418688, -0.081312], dtype=np.float32)


def sample_indices(n: int, t: int = T_SAMPLES) -> list[int]:
    """``t`` evenly-spaced frame indices (clamped/repeated for very short clips)."""
    if n <= 0:
        return [0] * t
    if n <= t:
        return (list(range(n)) + [n - 1] * t)[:t]
    return [int(round(i * (n - 1) / (t - 1))) for i in range(t)]


def to_proxy_stack(frames: list[np.ndarray], proxy: int = PROXY, t: int = T_SAMPLES) -> np.ndarray:
    """Sample ``t`` frames and resize to ``proxy`` -> ``(t, proxy, proxy, 4)`` uint8 RGBA."""
    idx = sample_indices(len(frames), t)
    out = np.empty((t, proxy, proxy, 4), dtype=np.uint8)
    for j, i in enumerate(idx):
        fr = frames[i]
        if fr.shape[-1] == 3:  # add opaque alpha if missing
            fr = np.dstack([fr, np.full(fr.shape[:2], 255, np.uint8)])
        im = Image.fromarray(fr, "RGBA").resize((proxy, proxy), Image.BILINEAR)
        out[j] = np.asarray(im, dtype=np.uint8)
    return out


def _ycbcr(stack_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T,H,W,4) uint8 RGBA -> (Y, Cb, Cr) each (T,H,W) float32, composited over black."""
    arr = stack_u8.astype(np.float32) / 255.0
    rgb = arr[..., :3] * arr[..., 3:4]          # alpha-composite over black
    y = rgb @ _RGB2Y
    cb = rgb @ _RGB2CB
    cr = rgb @ _RGB2CR
    return y, cb, cr


def _tdelta(x: np.ndarray) -> np.ndarray:
    d = np.zeros_like(x)
    d[1:] = x[1:] - x[:-1]
    return d


def features(ref_u8: np.ndarray, cand_u8: np.ndarray) -> np.ndarray:
    """Aligned proxy stacks ``(T,H,W,4)`` -> model input ``(T, N_CHANNELS, H, W)`` float32."""
    ry, rcb, rcr = _ycbcr(ref_u8)
    cy, ccb, ccr = _ycbcr(cand_u8)
    t, h, w = ry.shape
    feat = np.empty((t, N_CHANNELS, h, w), dtype=np.float32)
    feat[:, 0] = ry
    feat[:, 1] = cy
    feat[:, 2] = np.abs(cy - ry)
    feat[:, 3] = _tdelta(cy)
    feat[:, 4] = _tdelta(ry)
    feat[:, 5] = ccb - rcb
    feat[:, 6] = ccr - rcr
    return feat


def features_from_frames(ref_frames: list[np.ndarray], cand_frames: list[np.ndarray],
                         proxy: int = PROXY, t: int = T_SAMPLES) -> np.ndarray:
    """Convenience: RGBA frame lists -> aligned proxy stacks -> features ``(T,C,H,W)``."""
    return features(to_proxy_stack(ref_frames, proxy, t), to_proxy_stack(cand_frames, proxy, t))
