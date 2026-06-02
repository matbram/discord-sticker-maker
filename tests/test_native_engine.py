"""Tests for the native Rust engine (``fovea_native`` + ``FoveaNativeEngine``).

Skipped cleanly when the extension isn't built, mirroring the binary-gated tests.
The point of these is the product guarantee: every frame kept, and the global
256-color ceiling broken (per-frame local palettes), with static content reused.
"""
from __future__ import annotations

import numpy as np
import pytest

from encoder.core.frames import frames_from_list

try:
    import fovea_native

    HAVE_NATIVE = True
except Exception:  # noqa: BLE001
    HAVE_NATIVE = False

pytestmark = pytest.mark.skipif(not HAVE_NATIVE, reason="fovea_native extension not built")

W = H = 96


def _multi_hue_frames(n: int) -> list[np.ndarray]:
    """Opaque frames whose hue shifts per frame, so each needs its own rich palette."""
    out = []
    xx = np.arange(W)[None, :]
    yy = np.arange(H)[:, None]
    for k in range(n):
        a = np.zeros((H, W, 4), np.uint8)
        a[..., 0] = ((xx * 255 // W) + k * 9) % 256
        a[..., 1] = ((yy * 255 // H) + k * 5) % 256
        a[..., 2] = ((xx + yy) * 255 // (W + H) + k * 3) % 256
        a[..., 3] = 255
        out.append(np.ascontiguousarray(a))
    return out


def _static_bg_moving_box(n: int) -> list[np.ndarray]:
    """Static gradient background with a small moving opaque box (partial motion)."""
    bg = np.zeros((H, W, 4), np.uint8)
    bg[..., 0] = np.arange(W)[None, :] * 255 // W
    bg[..., 1] = np.arange(H)[:, None] * 255 // H
    bg[..., 2] = 128
    bg[..., 3] = 255
    out = []
    for k in range(n):
        f = bg.copy()
        x, y = (k * 3) % (W - 8), (k * 2) % (H - 8)
        f[y:y + 8, x:x + 8] = (240, 80, 40, 255)
        out.append(np.ascontiguousarray(f))
    return out


def _encode_native(frames, delays_cs, **kw):
    return fovea_native.encode([f.tobytes() for f in frames], W, H, delays_cs, **kw)


def test_engine_available_and_preferred():
    from encoder.core.encode import _select_engine
    from encoder.core.engines import FoveaNativeEngine, available_engines

    assert FoveaNativeEngine.available()
    assert "fovea-native" in [e.name for e in available_engines()]
    eng, _ = _select_engine(frames_from_list(_multi_hue_frames(3), 50), None)
    assert eng.name == "fovea-native"


def test_breaks_global_color_ceiling():
    frames = _multi_hue_frames(8)
    out = _encode_native(frames, [10] * 8, max_colors=256, delta_threshold=0.0)
    assert out["mode"] == "full"
    assert out["gif"][:6] == b"GIF89a"
    # The headline claim: more distinct colors than one global palette can hold.
    assert out["distinct_colors"] > 256, out["distinct_colors"]


def test_delta_reuses_static_pixels():
    frames = _static_bg_moving_box(12)
    out = _encode_native(frames, [5] * 12, max_colors=256, delta_threshold=0.02)
    assert out["mode"] == "delta"
    assert out["reused_pixels"] / out["total_pixels"] > 0.7


def test_transparency_falls_back_to_full_mode():
    frames = _static_bg_moving_box(4)
    for f in frames:
        f[0:H // 4, :, 3] = 0  # transparent band -> alpha matte
    out = _encode_native(frames, [10] * 4, max_colors=128, delta_threshold=0.05)
    assert out["mode"] == "full"


def test_end_to_end_keeps_all_frames_and_fits():
    from encoder import encode

    frames = _static_bg_moving_box(20)
    target = 256 * 1024
    res = encode(
        frames, target_bytes=target, mode="cap", delays_ms=[50] * 20,
        budget_seconds=30, max_attempts=24, out_path="/tmp/_native_e2e.gif",
    )
    assert res.size_bytes <= target
    # All frames preserved by decoding the produced GIF back.
    from encoder.core.frames import load_gif

    assert load_gif("/tmp/_native_e2e.gif").n == 20
