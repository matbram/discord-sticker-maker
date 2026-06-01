"""Unit tests for byte-size parsing, presets, and the tolerance window."""
from __future__ import annotations

import pytest

from encoder.core.sizes import (
    PLATFORM_PRESETS,
    Tolerance,
    parse_size_str,
    parse_tolerance,
    preset_bytes,
)


def test_parse_binary_units():
    assert parse_size_str("8MB") == 8 * 1024 * 1024
    assert parse_size_str("512KB") == 512 * 1024
    assert parse_size_str("500KB") == 500 * 1024
    assert parse_size_str("1MiB") == 1024 * 1024
    assert parse_size_str("1KiB") == 1024
    assert parse_size_str("2G") == 2 * 1024 ** 3
    assert parse_size_str("1024") == 1024
    assert parse_size_str("1024B") == 1024


def test_parse_accepts_int_and_float_and_whitespace_and_case():
    assert parse_size_str(1024) == 1024
    assert parse_size_str(1048576.0) == 1048576
    assert parse_size_str("8 mb") == 8 * 1024 * 1024
    assert parse_size_str("  512 KB  ") == 512 * 1024
    assert parse_size_str("1.5MB") == int(1.5 * 1024 * 1024)


def test_parse_errors():
    for bad in ["abc", "-5KB", "", "   ", "KB", "5 furlongs", "5.5.5MB"]:
        with pytest.raises(ValueError):
            parse_size_str(bad)
    for bad_num in [0, -10, 0.0, -3.2]:
        with pytest.raises(ValueError):
            parse_size_str(bad_num)
    with pytest.raises(ValueError):
        parse_size_str(True)  # bool must be rejected despite being an int subclass


def test_presets():
    assert preset_bytes("discord") == 512 * 1024
    assert preset_bytes("Discord") == 512 * 1024  # case-insensitive
    assert "discord" in PLATFORM_PRESETS
    with pytest.raises(ValueError):
        preset_bytes("nonsense-platform")


def test_tolerance_window_is_one_sided():
    t = Tolerance(0.05)
    target = 512 * 1024
    assert t.fits(target, target)
    assert not t.fits(target + 1, target)
    assert t.in_window(500 * 1024, target)
    assert not t.in_window(400 * 1024, target)   # too far under
    assert not t.in_window(target + 1, target)   # over target
    assert t.lo(target) == int(target * 0.95)


def test_parse_tolerance():
    assert parse_tolerance("5%").under_frac == pytest.approx(0.05)
    assert parse_tolerance("0").under_frac == 0.0
    assert parse_tolerance(0.1).under_frac == pytest.approx(0.1)
    assert parse_tolerance(Tolerance(0.2)).under_frac == 0.2
    with pytest.raises(ValueError):
        Tolerance(1.0)   # must be < 1
    with pytest.raises(ValueError):
        Tolerance(-0.1)
