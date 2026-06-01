"""Unit tests for the side-by-side legacy-baseline cache (skip if backend absent)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
fg = pytest.importorskip("app.pipeline.fovea_gif")


def _fitted(seed=0, n=6, hw=32):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (hw, hw, 4), dtype=np.uint8) for _ in range(n)]


def test_sig_stable_and_content_sensitive():
    frames, delays = _fitted(0), [80] * 6
    base = fg._legacy_sig(frames, delays, 500_000, 256, 24)
    assert base == fg._legacy_sig(frames, delays, 500_000, 256, 24)          # deterministic
    assert fg._legacy_sig(frames, delays, 400_000, 256, 24) != base          # budget matters
    assert fg._legacy_sig(frames, delays, 500_000, 128, 24) != base          # colors matter
    assert fg._legacy_sig(_fitted(1), delays, 500_000, 256, 24) != base      # frame content matters


def test_store_hit_and_eviction():
    fg._legacy_cache.clear()
    assert fg._legacy_get("sig-x") is None
    fg._legacy_put("sig-x", b"GIFDATA", 14)
    assert fg._legacy_get("sig-x") == (b"GIFDATA", 14)
    for i in range(fg._LEGACY_MAX + 2):          # overflow evicts the oldest (sig-x)
        fg._legacy_put(f"s{i}", b"x", i)
    assert fg._legacy_get("sig-x") is None
    fg._legacy_cache.clear()
