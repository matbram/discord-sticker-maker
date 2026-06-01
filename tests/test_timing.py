"""Unit tests for the centisecond timing-grid mapping."""
from __future__ import annotations

import pytest

from encoder.core.timing import (
    MIN_DELAY_CS,
    centiseconds_to_ms,
    effective_fps,
    fps_to_centiseconds,
    ms_to_centiseconds,
)


def test_fps_grid_preserves_duration_24fps():
    n = 96
    cs = fps_to_centiseconds(24.0, n)
    assert len(cs) == n
    assert all(c >= MIN_DELAY_CS for c in cs)
    # cumulative duration tracks the source to within a single centisecond
    assert abs(sum(cs) * 10 - n * 1000 / 24) <= 10


def test_fps_grid_error_diffusion_7fps():
    n = 70  # 100/7 = 14.2857 cs/frame — a value naive rounding drifts on
    cs = fps_to_centiseconds(7.0, n)
    assert abs(sum(cs) * 10 - n * 1000 / 7) <= 10


def test_ms_round_trip_preserves_duration():
    delays = [33, 33, 34] * 10  # ~30 fps, non-integer cs
    cs = ms_to_centiseconds(delays)
    assert len(cs) == len(delays)
    assert abs(sum(cs) * 10 - sum(delays)) <= 10


def test_short_delays_are_clamped():
    cs = ms_to_centiseconds([5, 5, 1, 0])  # all sub-2cs
    assert all(c >= MIN_DELAY_CS for c in cs)


def test_effective_fps():
    assert effective_fps([40, 40, 40, 40]) == pytest.approx(25.0, abs=0.01)
    assert effective_fps([]) is None
    assert effective_fps([0]) is None


def test_centiseconds_to_ms():
    assert centiseconds_to_ms([4, 10, 2]) == [40, 100, 20]


def test_fps_validation():
    with pytest.raises(ValueError):
        fps_to_centiseconds(0, 10)
    assert fps_to_centiseconds(24, 0) == []
