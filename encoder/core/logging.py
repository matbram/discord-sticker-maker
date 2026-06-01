"""Fovea-local structured logging over the stdlib ``logging`` module.

Deliberately does NOT import the backend's ``structlog`` setup — Fovea is a
standalone package. ``get_logger("frames").info("event", key=value)`` emits
``event key=value`` at INFO. Level is controlled by ``FOVEA_LOG_LEVEL``.
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("FOVEA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    # stdout, not the StreamHandler default (stderr): platforms classify stderr as
    # error level, which would file every INFO encode log as an error.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger("fovea")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def _fmt(event: str, kv: dict) -> str:
    if not kv:
        return event
    return event + " " + " ".join(f"{k}={v}" for k, v in kv.items())


class KVLogger:
    """Thin key-value wrapper so call sites read like ``log.info(event, **kv)``."""

    def __init__(self, logger: logging.Logger) -> None:
        self._log = logger

    def debug(self, event: str, **kv) -> None:
        self._log.debug(_fmt(event, kv))

    def info(self, event: str, **kv) -> None:
        self._log.info(_fmt(event, kv))

    def warning(self, event: str, **kv) -> None:
        self._log.warning(_fmt(event, kv))

    def error(self, event: str, **kv) -> None:
        self._log.error(_fmt(event, kv))


def get_logger(name: str) -> KVLogger:
    _configure()
    return KVLogger(logging.getLogger(f"fovea.{name}"))
