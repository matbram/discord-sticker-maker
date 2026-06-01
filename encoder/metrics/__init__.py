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
    from .perceptual import PerceptualMetric

    # Opt-in only: MS-SSIM stays the default until the learned judge is proven out.
    if os.getenv("FOVEA_METRIC", "").lower() == "learned":
        lm = _try_learned()
        if lm is not None:
            return lm
    return external.best_available() or PerceptualMetric()


def available_metrics() -> list[str]:
    from . import external, learned

    names = ["msssim", "msssim+temporal"]
    if learned.model_available():
        names.append("learned")
    return names + external.available_external()


def get_metric(name: str | None) -> Metric:
    """Resolve a metric by name (``"auto"``/``None`` -> default)."""
    from .perceptual import PerceptualMetric

    if name in (None, "auto"):
        return default_metric()
    if name in ("msssim", "msssim+temporal"):
        return PerceptualMetric()
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
