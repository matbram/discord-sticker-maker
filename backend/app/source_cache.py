"""In-memory cache of ingested ``Source`` objects, keyed by content sha1.

Lets the browser upload a large source ONCE and then reference it by ``source_id``
on every subsequent regenerate (slider tweak), instead of re-uploading megabytes
each time — the single biggest chunk of per-edit latency. Small LRU + TTL, byte-
capped so it stays safe on tiny hosts. Stdlib-only (no app imports) so it's cheap
to import and easy to unit-test.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict

TTL_SECONDS = 900                       # drop after 15 min of inactivity
MAX_ENTRIES = 6
MAX_TOTAL_BYTES = 128 * 1024 * 1024     # ~6 typical uploads

_lock = threading.Lock()
_entries: "OrderedDict[str, tuple]" = OrderedDict()   # id -> (source, n_bytes, ts)


def source_id_for(data: bytes) -> str:
    """Stable id for a source's bytes (matches the matte cache's source key)."""
    return hashlib.sha1(data).hexdigest()[:16]


def _evict_locked() -> None:
    now = time.time()
    for k in [k for k, (_, _, ts) in _entries.items() if now - ts > TTL_SECONDS]:
        _entries.pop(k, None)
    total = sum(b for _, b, _ in _entries.values())
    while _entries and (len(_entries) > MAX_ENTRIES or total > MAX_TOTAL_BYTES):
        _, (_, b, _) = _entries.popitem(last=False)   # evict least-recently-used
        total -= b


def put(source) -> str:
    """Store ``source``; return its id (refreshing it if already present)."""
    sid = source_id_for(source.data)
    with _lock:
        _entries[sid] = (source, len(source.data), time.time())
        _entries.move_to_end(sid)
        _evict_locked()
    return sid


def get(sid: str):
    """Return the cached ``Source`` for ``sid`` (refreshing its recency), or None."""
    with _lock:
        e = _entries.get(sid)
        if e is None or (time.time() - e[2]) > TTL_SECONDS:
            if e is not None:
                _entries.pop(sid, None)
            return None
        source, n, _ = e
        _entries[sid] = (source, n, time.time())
        _entries.move_to_end(sid)
        return source


def clear() -> None:
    with _lock:
        _entries.clear()
