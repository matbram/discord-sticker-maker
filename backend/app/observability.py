"""Structured logging + request tracing.

Every log line is JSON on stdout (Railway-friendly) and carries the active
``request_id`` so all work for one sticker can be traced from logs alone.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
import uuid
from typing import Iterator

import structlog


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def bind_request(request_id: str) -> None:
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request() -> None:
    structlog.contextvars.clear_contextvars()


@contextlib.contextmanager
def stage(name: str, **fields) -> Iterator:
    """Log start / done / error and duration for a pipeline stage."""
    log = get_logger("pipeline").bind(stage=name, **fields)
    start = time.perf_counter()
    log.info("stage.start")
    try:
        yield log
    except Exception as exc:  # noqa: BLE001 - re-raised after logging
        dur = (time.perf_counter() - start) * 1000.0
        log.error("stage.error", duration_ms=round(dur, 1), error=str(exc), exc_info=True)
        raise
    else:
        dur = (time.perf_counter() - start) * 1000.0
        log.info("stage.done", duration_ms=round(dur, 1))
