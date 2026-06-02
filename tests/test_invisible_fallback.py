"""Regression guard for the invisible-mode fallback (bridge).

The Fovea encoder never trims frames, so when a clip can't be made perceptually
lossless under the byte ceiling, a single ``mode="invisible"`` encode is stuck on a
low palette (washed out) and is WORSE than the default cap budget-fill. ``_run_fovea``
must therefore fall back to cap when invisible can't reach lossless.

These tests stub the encode primitive, so they need no ffmpeg and are deterministic.
Skipped if the backend (structlog etc.) isn't importable.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
fg = pytest.importorskip("app.pipeline.fovea_gif")


@pytest.fixture(autouse=True)
def _force_legacy_bridge(monkeypatch):
    # The invisible->cap fallthrough under test is the *legacy* bridge path; the native
    # engine handles invisible inside encode(), so disable native here to exercise it.
    monkeypatch.setattr(fg, "_native_available", lambda: False)


def _frames(n=8, hw=32):
    return [np.zeros((hw, hw, 4), np.uint8) for _ in range(n)], [80] * n


def test_invisible_returns_immediately_when_lossless(monkeypatch):
    calls = []

    def fake_encode_once(fitted, delays, budget, seconds, attempts, mode="cap"):
        calls.append(mode)
        return b"x" * 100, 10.0, 64, {"mode": mode, "perceptually_lossless": True}

    monkeypatch.setattr(fg, "_encode_once", fake_encode_once)
    frames, delays = _frames()
    data, n, fps, colors, report = fg._run_fovea(frames, delays, 100_000, "balanced", "invisible", None)

    assert calls == ["invisible"]            # no fallback: returned the smallest-lossless result
    assert report["mode"] == "invisible"
    assert report["perceptually_lossless"] is True
    assert n == len(frames)


def test_invisible_falls_back_to_cap_when_not_lossless(monkeypatch):
    calls = []

    def fake_encode_once(fitted, delays, budget, seconds, attempts, mode="cap"):
        calls.append(mode)
        # invisible can't reach lossless and is stuck on a washed-out 20-color palette;
        # the cap path would reach a richer 200-color palette.
        colors = 20 if mode == "invisible" else 200
        return b"x" * 1000, 10.0, colors, {"mode": mode, "perceptually_lossless": False}

    monkeypatch.setattr(fg, "_encode_once", fake_encode_once)
    frames, delays = _frames()
    # "smooth" keeps every frame and skips trim/frame-fill, so the cap path is a single
    # cap encode — exactly what we want to observe the fallback without ffmpeg.
    data, n, fps, colors, report = fg._run_fovea(frames, delays, 100_000, "smooth", "invisible", None)

    assert "invisible" in calls and "cap" in calls   # fell back to the cap budget-fill path
    assert report["mode"] == "cap"                    # returned the cap result, not the washed one
    assert colors == 200                              # the richer palette, not the stuck 20
