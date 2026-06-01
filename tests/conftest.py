"""Shared pytest fixtures: synthetic frames and binary-availability helpers.

Everything here runs without external binaries or real clips.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from encoder.core.frames import Frames, frames_from_list


def _make_frames(
    n: int = 6, w: int = 64, h: int = 64, *, seed: int = 0,
    motion: bool = False, static: bool = True,
) -> Frames:
    """Build a synthetic RGBA ``Frames``.

    ``static`` repeats one base image; ``motion`` horizontally scrolls it.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    frames: list[np.ndarray] = []
    for i in range(n):
        if motion:
            rgb = np.roll(base, i * 4, axis=1)
        else:  # static
            rgb = base.copy()
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        frames.append(np.concatenate([rgb, alpha], axis=-1))
    return frames_from_list(frames, 100)


@pytest.fixture
def make_frames():
    return _make_frames


def _add_noise(frames: Frames, sigma: float, *, seed: int = 7) -> Frames:
    """Return a copy of ``frames`` with Gaussian RGB noise (alpha preserved)."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for fr in frames.frames:
        rgb = fr[..., :3].astype(np.float64) + rng.normal(0.0, sigma, fr[..., :3].shape)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        out.append(np.concatenate([rgb, fr[..., 3:4]], axis=-1))
    return frames_from_list(out, list(frames.delays_ms))


@pytest.fixture
def add_noise():
    return _add_noise


HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_FFPROBE = shutil.which("ffprobe") is not None
HAVE_GIFSICLE = shutil.which("gifsicle") is not None
HAVE_GIFSKI = shutil.which("gifski") is not None
HAVE_GIF_ENGINE = HAVE_FFMPEG or HAVE_GIFSICLE or HAVE_GIFSKI
