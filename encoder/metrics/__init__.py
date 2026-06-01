"""Perceptual metric registry.

``default_metric()`` returns the best available reference judge — an external
binary adapter if present, otherwise the always-available pure-numpy
``PerceptualMetric`` (MS-SSIM + temporal). The learned, motion-aware metric
(milestone M2) will register here.
"""
from __future__ import annotations

from .base import DistanceResult, Metric

__all__ = ["DistanceResult", "Metric", "default_metric", "available_metrics", "get_metric"]


def default_metric() -> Metric:
    from . import external
    from .perceptual import PerceptualMetric

    return external.best_available() or PerceptualMetric()


def available_metrics() -> list[str]:
    from . import external

    return ["msssim", "msssim+temporal"] + external.available_external()


def get_metric(name: str | None) -> Metric:
    """Resolve a metric by name (``"auto"``/``None`` -> default)."""
    from .perceptual import PerceptualMetric

    if name in (None, "auto"):
        return default_metric()
    if name in ("msssim", "msssim+temporal"):
        return PerceptualMetric()
    from . import external

    if name in external.available_external():
        ext = external.best_available()
        if ext is not None:
            return ext
    raise ValueError(f"unknown or unavailable metric: {name!r}")
