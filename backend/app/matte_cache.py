"""Tiny in-memory cache for the expensive, reusable part of the pipeline: the
decoded + background-matted frames. Keyed by what actually affects them (source
bytes + trim + background settings), so editing a downstream setting (zoom,
output type, GIF quality) reuses the cutout instead of recomputing it.

Single entry by design — one person editing one source — to keep memory bounded
on small hosts. Replacing the entry frees the previous frames.
"""
from __future__ import annotations

import threading
import time

TTL_SECONDS = 600  # drop after 10 min of inactivity
MAX_BYTES = 250 * 1024 * 1024  # don't cache absurdly large frame sets

_lock = threading.Lock()
_entry: tuple | None = None  # (key, value, timestamp)
stats = {"hits": 0, "misses": 0}


def get(key):
    global _entry
    with _lock:
        if _entry is not None and _entry[0] == key and (time.time() - _entry[2]) < TTL_SECONDS:
            stats["hits"] += 1
            return _entry[1]
        stats["misses"] += 1
        return None


def put(key, value, approx_bytes: int) -> None:
    global _entry
    if approx_bytes > MAX_BYTES:
        return  # too big to keep around safely
    with _lock:
        _entry = (key, value, time.time())


def clear() -> None:
    global _entry
    with _lock:
        _entry = None
