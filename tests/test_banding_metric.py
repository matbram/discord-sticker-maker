"""The banding-aware color metric: the term that lets the search be trusted on color.

These lock in the fix for the documented blind spot — luma MS-SSIM is unreliable on
*chroma* banding (it barely sees it), so it can prefer a banded result to a dithered
one. The color-aware metric must reliably prefer dither over banding.
"""
from __future__ import annotations

import numpy as np

from encoder.core.frames import frames_from_list
from encoder.metrics import default_metric, get_metric
from encoder.metrics.banding import banding_per_frame
from encoder.metrics.perceptual import ColorAwareMetric, PerceptualMetric

H, W = 96, 256
_BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16 - 0.5


def _frames(rgb: np.ndarray):
    a = np.clip(rgb, 0, 255).astype(np.uint8)
    rgba = np.concatenate([a, np.full(a.shape[:2] + (1,), 255, np.uint8)], -1)
    return frames_from_list([rgba, rgba, rgba], 100)


def _chroma_ramp():
    """A blue ramp with R=G constant: luma barely moves, so banding lives in chroma."""
    x = np.arange(W)[None, :].repeat(H, 0).astype(np.float64)
    src = np.dstack([np.full((H, W), 128.0), np.full((H, W), 128.0), x])
    step = 255 / 3  # 4 levels -> obvious banding
    banded = np.round(src / step) * step
    tile = np.tile(_BAYER, (H // 4, W // 4))
    dith = src.copy()
    dith[..., 2] = np.clip(np.round(src[..., 2] / step + tile) * step, 0, 255)
    return _frames(src), _frames(banded), _frames(dith)


def test_band_term_separates_banding_from_dither():
    src, banded, dith = _chroma_ramp()
    b_banded = banding_per_frame(src.frames, banded.frames).mean()
    b_dith = banding_per_frame(src.frames, dith.frames).mean()
    # Dithering preserves the local mean; banding shifts it -> far larger low-pass ΔE.
    assert b_banded > 5 * b_dith, (b_banded, b_dith)


def test_color_aware_metric_prefers_dither_over_banding():
    src, banded, dith = _chroma_ramp()
    ca = ColorAwareMetric()
    assert ca.distance(src, dith).distance < ca.distance(src, banded).distance


def test_identical_is_zero_distance():
    src, _, _ = _chroma_ramp()
    assert default_metric().distance(src, src).distance == 0.0


def test_threshold_flags_banding_not_dither():
    """Hard banding is flagged not-lossless and scores worse than the dithered version."""
    src, banded, dith = _chroma_ramp()
    ca = ColorAwareMetric()
    d_band = ca.distance(src, banded).distance
    d_dith = ca.distance(src, dith).distance
    assert d_dith < d_band                          # dither is the better approximation
    assert d_band > ca.invisible_threshold          # banding is not marked perceptually lossless


def test_default_metric_is_color_aware():
    assert isinstance(default_metric(), ColorAwareMetric)
    assert isinstance(get_metric("color"), ColorAwareMetric)
    # Pure MS-SSIM stays available by name for comparison.
    assert isinstance(get_metric("msssim"), PerceptualMetric)


def test_band_term_is_exposed_in_extra():
    src, banded, _ = _chroma_ramp()
    res = ColorAwareMetric().distance(src, banded)
    assert res.extra["band_mean"] > 0 and "msssim_mean" in res.extra
