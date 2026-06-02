"""Perceptual metric registry.

``default_metric()`` returns the best available reference judge — an external
binary adapter if present, otherwise the always-available pure-numpy
``PerceptualMetric`` (MS-SSIM + temporal). The learned, motion-aware metric
(milestone M2) will register here.
"""
from __future__ import annotations

import os

from .base import DistanceResult, Metric

__all__ = ["DistanceResult", "Metric", "default_metric", "available_metrics", "get_metric"]


def _try_learned() -> Metric | None:
    """The learned judge if its model + onnxruntime are present, else None."""
    from . import learned

    if not learned.model_available():
        return None
    try:
        return learned.LearnedMetric()
    except Exception:  # noqa: BLE001 - a bad model must never break the encoder
        import logging

        logging.getLogger("fovea.metrics").warning("learned metric failed to load; using MS-SSIM")
        return None


def default_metric() -> Metric:
    from . import external
    from .perceptual import ColorAwareMetric

    # The learned judge (opt-in) wins when present; otherwise the color-aware judge
    # (MS-SSIM + temporal + banding-aware OKLab term) is the default so the search
    # can be trusted on color. Pure MS-SSIM stays available by name for comparison.
    if os.getenv("FOVEA_METRIC", "").lower() == "learned":
        lm = _try_learned()
        if lm is not None:
            return lm
    if os.getenv("FOVEA_METRIC", "").lower() in ("msssim", "msssim+temporal"):
        from .perceptual import PerceptualMetric

        return PerceptualMetric()
    return external.best_available() or ColorAwareMetric()


def available_metrics() -> list[str]:
    from . import external, learned

    names = ["msssim", "msssim+temporal", "color", "msssim+temporal+color"]
    if learned.model_available():
        names.append("learned")
    return names + external.available_external()


def get_metric(name: str | None) -> Metric:
    """Resolve a metric by name (``"auto"``/``None`` -> default)."""
    from .perceptual import ColorAwareMetric, PerceptualMetric

    if name in (None, "auto"):
        return default_metric()
    if name in ("msssim", "msssim+temporal"):
        return PerceptualMetric()
    if name in ("color", "msssim+temporal+color"):
        return ColorAwareMetric()
    if name == "learned":
        lm = _try_learned()
        if lm is not None:
            return lm
        raise ValueError("metric 'learned' requested but model/onnxruntime unavailable")
    from . import external

    if name in external.available_external():
        ext = external.best_available()
        if ext is not None:
            return ext
    raise ValueError(f"unknown or unavailable metric: {name!r}")
