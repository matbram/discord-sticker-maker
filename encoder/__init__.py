"""Fovea — a perceptually-lossless GIF encoder.

Produces the best-looking ``.gif`` that fits under a hard byte cap while keeping
every frame and staying invisible to the eye. ``encode`` and ``EncodeResult`` are
imported lazily so that lightweight submodules (``encoder.core.sizes``,
``encoder.core.timing``) can be used without pulling in numpy/Pillow.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["encode", "EncodeResult", "__version__"]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name == "encode":
        from .core.encode import encode

        return encode
    if name == "EncodeResult":
        from .core.result import EncodeResult

        return EncodeResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
