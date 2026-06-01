"""Byte-size parsing, platform presets, and the size tolerance window.

Units are **binary by default** (``KB`` = 1024, ``MB`` = 1024**2). This is the
conservative choice: platform caps such as Discord's "512KB" are themselves
1024-based, and interpreting decimally ("500KB" -> 500_000) risks overshooting a
hard cap. ``KiB``/``MiB`` are accepted as explicit synonyms; a bare number is bytes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Map an upper-cased unit token to its byte factor. Binary throughout.
_UNIT_FACTORS: dict[str, int] = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "KIB": 1024,
    "M": 1024 ** 2,
    "MB": 1024 ** 2,
    "MIB": 1024 ** 2,
    "G": 1024 ** 3,
    "GB": 1024 ** 3,
    "GIB": 1024 ** 3,
}

_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)\s*$")


def parse_size_str(value: str | int | float) -> int:
    """Parse ``"8MB"`` / ``"512KB"`` / ``"1MiB"`` / ``"1024"`` / ``1024`` into bytes.

    Decimal points are allowed (``"1.5MB"``). Raises ``ValueError`` on anything
    that does not name a strictly-positive size.
    """
    # bool is an int subclass — reject it explicitly so ``parse_size_str(True)`` fails.
    if isinstance(value, bool):
        raise ValueError(f"invalid size: {value!r}")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"size must be positive, got {value}")
        return value
    if isinstance(value, float):
        out = int(round(value))
        if out <= 0:
            raise ValueError(f"size must be positive, got {value}")
        return out
    if not isinstance(value, str):
        raise ValueError(f"cannot parse size from {type(value).__name__}")

    m = _SIZE_RE.match(value)
    if not m:
        raise ValueError(f"cannot parse size: {value!r}")
    number = float(m.group(1))
    unit = m.group(2).upper()
    if unit not in _UNIT_FACTORS:
        raise ValueError(f"unknown size unit {m.group(2)!r} in {value!r}")
    out = int(round(number * _UNIT_FACTORS[unit]))
    if out <= 0:
        raise ValueError(f"size must be positive, got {value!r}")
    return out


# Hard platform ceilings we know about, in bytes (all 1024-based).
PLATFORM_PRESETS: dict[str, int] = {
    "discord": 512 * 1024,        # animated sticker / standard upload ceiling we target
    "discord-emoji": 256 * 1024,
    "slack": 128 * 1024,
    "telegram": 512 * 1024,
}


def preset_bytes(name: str) -> int:
    """Resolve a platform preset name to a byte target. ``ValueError`` if unknown."""
    key = name.strip().lower()
    if key not in PLATFORM_PRESETS:
        raise ValueError(
            f"unknown platform preset {name!r}; known: {', '.join(sorted(PLATFORM_PRESETS))}"
        )
    return PLATFORM_PRESETS[key]


@dataclass(frozen=True)
class Tolerance:
    """One-sided acceptance window ``[target*(1-under_frac), target]`` — never over.

    ``fits`` is the hard constraint (``size <= target``); ``in_window`` is the
    "good enough, stop searching" predicate the size search uses for early exit.
    """

    under_frac: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 <= self.under_frac < 1.0):
            raise ValueError(f"under_frac must be in [0, 1), got {self.under_frac}")

    def lo(self, target: int) -> int:
        return int(target * (1.0 - self.under_frac))

    def fits(self, size: int, target: int) -> bool:
        return size <= target

    def in_window(self, size: int, target: int) -> bool:
        return self.lo(target) <= size <= target


def parse_tolerance(spec: str | float | Tolerance) -> Tolerance:
    """Parse ``"5%"`` / ``"0.05"`` / ``"0"`` (or pass a number/Tolerance) -> Tolerance."""
    if isinstance(spec, Tolerance):
        return spec
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return Tolerance(float(spec))
    s = str(spec).strip()
    if s.endswith("%"):
        return Tolerance(float(s[:-1]) / 100.0)
    return Tolerance(float(s))
