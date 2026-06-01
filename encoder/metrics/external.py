"""Optional adapters for stronger reference metrics provided as external binaries.

If a tool like ``ssimulacra2`` or ``butteraugli`` is on PATH it can be used as a
better stand-in judge than MS-SSIM. When none is present (the common case, and
always in CI) ``best_available`` returns ``None`` and the caller falls back to the
pure-numpy ``PerceptualMetric``. These adapters are intentionally never required.
"""
from __future__ import annotations

import shutil

from .base import Metric

# Binaries we know how to talk to, in descending preference.
KNOWN_EXTERNAL = ("ssimulacra2", "butteraugli")


def available_external() -> list[str]:
    return [name for name in KNOWN_EXTERNAL if shutil.which(name)]


def best_available() -> Metric | None:
    """Return an external-metric adapter if a known binary exists, else ``None``.

    Adapters are not yet implemented (no binary is present in current targets);
    this hook keeps the registry honest and gives M2 a place to plug in.
    """
    return None
