"""Unit tests for the server-side source cache. Skipped if the backend isn't importable."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sc = pytest.importorskip("app.source_cache")


class _Src:
    def __init__(self, data):
        self.data = data


def test_put_get_roundtrip():
    sc.clear()
    s = _Src(b"hello-bytes")
    sid = sc.put(s)
    assert sid == sc.source_id_for(b"hello-bytes")
    assert sc.get(sid) is s
    assert sc.get("unknown-id") is None
    sc.clear()


def test_get_drops_expired(monkeypatch):
    sc.clear()
    sid = sc.put(_Src(b"abc"))
    monkeypatch.setattr(sc, "TTL_SECONDS", 0)   # anything older than "now" is stale
    time.sleep(0.01)
    assert sc.get(sid) is None
    sc.clear()


def test_lru_eviction_by_count():
    sc.clear()
    ids = [sc.put(_Src(f"src-{i}".encode())) for i in range(sc.MAX_ENTRIES + 2)]
    assert sc.get(ids[0]) is None        # oldest evicted
    assert sc.get(ids[1]) is None
    assert sc.get(ids[-1]) is not None   # newest kept
    sc.clear()
