"""Regression tests for the GIF bridge's frame-rate floor, fit-rescue, and the
single-source-of-truth comparison. All deterministic (the encode primitive and the
legacy/metric calls are stubbed), so they need no ffmpeg. Skipped if the backend
package isn't importable.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
fg = pytest.importorskip("app.pipeline.fovea_gif")


@pytest.fixture(autouse=True)
def _force_legacy_bridge(monkeypatch):
    # These tests stub _encode_once to exercise the *legacy* ffmpeg-path dance (fps
    # floor, fit-rescue, frame-fill). The native engine bypasses that dance — it
    # self-guarantees the byte fit in one call — so disable it here.
    monkeypatch.setattr(fg, "_native_available", lambda: False)


def _frames(n, hw=8):
    return [np.zeros((hw, hw, 4), np.uint8) for _ in range(n)], [100] * n  # 100ms -> n/10 s


def test_fps_floor_prevents_slideshow(monkeypatch):
    # 29 frames @ 100ms = 2.9s. sharp floor 5fps -> min_n ~15. Colors stay below the
    # sharp palette floor (160) so the trim is driven all the way to the fps floor.
    def fake(fitted, delays, budget, seconds, attempts, mode="cap"):
        n = len(fitted)
        return (b"x" * int(budget * 0.6), 10.0, max(8, 8 * 29 // n),
                {"mode": mode, "dither": "none", "perceptually_lossless": False,
                 "perceptual_distance": 0.05})

    monkeypatch.setattr(fg, "_encode_once", fake)
    monkeypatch.setattr(fg, "_fill_frames_at_colors", lambda *a, **k: None)
    frames, delays = _frames(29)
    _, n, fps, _, _ = fg._run_fovea(frames, delays, 500_000, "sharp", "cap", None)
    assert n >= 14, f"trimmed into a slideshow: {n} frames"
    assert fps >= 4.5, f"fps floor breached: {fps}"


def test_fit_rescue_trims_below_floor_to_fit(monkeypatch):
    # Everything above 8 frames overshoots the budget; fitting is mandatory, so the
    # rescue must trim below the fps floor.
    def fake(fitted, delays, budget, seconds, attempts, mode="cap"):
        n = len(fitted)
        size = int(budget * 0.8) if n <= 8 else budget * 2
        return (b"x" * size, 10.0, max(8, 8 * 29 // n),
                {"mode": mode, "dither": "none", "perceptually_lossless": False,
                 "perceptual_distance": 0.05})

    monkeypatch.setattr(fg, "_encode_once", fake)
    monkeypatch.setattr(fg, "_fill_frames_at_colors", lambda *a, **k: None)
    frames, delays = _frames(29)
    data, n, _, _, _ = fg._run_fovea(frames, delays, 500_000, "sharp", "cap", None)
    assert len(data) <= 500_000, "fit-rescue failed to get under budget"
    assert n <= 8


def test_comparison_fovea_is_the_report(monkeypatch):
    # The comparison's Fovea side must mirror the encoder report (single source of
    # truth) so the side-by-side badge can't contradict the honesty line.
    rep = {"mode": "cap", "perceptually_lossless": True, "perceptual_distance": 0.0031,
           "dither": "none"}
    monkeypatch.setattr(fg, "_run_fovea", lambda *a, **k: (b"foveadata", 14, 10.0, 256, rep))
    monkeypatch.setattr(fg, "_aligned_distance", lambda *a, **k: 0.25)   # legacy score
    monkeypatch.setattr(fg, "_legacy_get", lambda sig: (b"legacydata", 14))  # skip ffmpeg

    class _M:
        name = "msssim+temporal"
        invisible_threshold = 0.005

    monkeypatch.setattr("encoder.metrics.default_metric", lambda: _M())
    frames, delays = _frames(14)
    out = fg.gif_encode_compare(frames, delays, budget=500_000, notes=[])
    _, _, _, _, comparison, _, report = out
    assert report is rep
    assert comparison["fovea"]["perceptually_lossless"] is True   # from the report
    assert comparison["fovea"]["distance"] == 0.0031              # from the report
    assert comparison["legacy"]["distance"] == 0.25
