"""End-to-end test of the upload-once / reuse-by-id flow on /api/process.

Skipped unless the web stack (fastapi/httpx/python-multipart) is installed, mirroring
how the encoder smoke tests skip without ffmpeg.
"""
import io
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")  # python-multipart, needed for Form/File parsing
Image = pytest.importorskip("PIL.Image")

from fastapi.testclient import TestClient  # noqa: E402

from app import source_cache  # noqa: E402
from app.main import app  # noqa: E402


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


def test_upload_then_reuse_then_expiry():
    source_cache.clear()
    params = '{"outputs":[{"type":"sticker"}]}'
    with TestClient(app) as client:
        # 1) first upload returns a reusable source_id
        r1 = client.post("/api/process", data={"params": params},
                         files={"file": ("x.png", _png(), "image/png")})
        assert r1.status_code == 200, r1.text
        sid = r1.json()["source_id"]
        assert sid and source_cache.get(sid) is not None

        # 2) regenerate by id only (no file re-upload) reuses the cached source
        r2 = client.post("/api/process", data={"params": params, "source_id": sid})
        assert r2.status_code == 200, r2.text
        assert r2.json()["source_id"] == sid

        # 3) an unknown/expired id with no fallback signals the client to resend
        r3 = client.post("/api/process", data={"params": params, "source_id": "deadbeefdeadbeef"})
        assert r3.status_code == 409
        assert r3.json().get("source_expired") is True

        # Let the background jobs finish before the test loop is torn down, so their
        # progress callbacks don't fire into a closed event loop.
        from app.main import store
        jids = [r1.json()["job_id"], r2.json()["job_id"]]
        end = time.time() + 10
        while time.time() < end and any(
                (j := store.get(jid)) is not None and j.status == "running" for jid in jids):
            time.sleep(0.05)
    source_cache.clear()
