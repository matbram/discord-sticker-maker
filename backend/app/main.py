"""FastAPI app: upload → SSE progress → download, plus health and client logging."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import observability as obs
from . import source_cache
from .jobs import JobStore
from .models import ProcessParams
from .pipeline import bg_removal, ingest, orchestrator

log = obs.get_logger("app")


def _sha1(data: bytes) -> str:
    """Short content fingerprint so a downloaded file can be matched to served bytes."""
    return hashlib.sha1(data).hexdigest()[:12]


def _enum(v):
    return v.value if hasattr(v, "value") else v


store = JobStore()
PIPELINE_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("PIPELINE_CONCURRENCY", "2")), thread_name_prefix="pipeline"
)
_ready = {"bg": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    obs.configure_logging()
    log.info("startup.begin", bg_available=bg_removal.available())

    def warm():
        bg_removal.warmup([bg_removal.DEFAULT_MODEL])
        _ready["bg"] = bg_removal.available()
        log.info("startup.warm_done", bg_ready=_ready["bg"])

    threading.Thread(target=warm, daemon=True).start()
    yield
    PIPELINE_EXECUTOR.shutdown(wait=False)


app = FastAPI(title="Discord Sticker Maker", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or obs.new_request_id()
    request.state.request_id = request_id
    obs.bind_request(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        dur = (time.perf_counter() - start) * 1000.0
        log.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            duration_ms=round(dur, 1),
        )
    response.headers["X-Request-ID"] = request_id
    obs.clear_request()
    return response


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/process")
async def process(
    request: Request,
    file: UploadFile | None = File(default=None),
    url: str | None = Form(default=None),
    params: str | None = Form(default=None),
    source_id: str | None = Form(default=None),
):
    request_id = getattr(request.state, "request_id", None) or obs.new_request_id()
    try:
        parsed = ProcessParams.model_validate_json(params) if params else ProcessParams()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"Invalid params: {exc}"}, status_code=400)

    log.info(
        "audit.request",
        request_id=request_id,
        remove_bg=parsed.remove_bg,
        max_fps=parsed.max_fps,
        max_duration_s=parsed.max_duration_s,
        outputs=[
            {"type": _enum(o.type), "max_bytes": o.max_bytes, "max_dim": o.max_dim,
             "gif_quality": _enum(o.gif_quality), "priority": _enum(o.priority)}
            for o in (parsed.outputs or [])
        ],
    )

    try:
        source = source_cache.get(source_id) if source_id else None
        if source is not None:
            log.info("source_cache.hit", source_id=source_id, bytes=len(source.data))
        elif source_id and file is None and not url:
            # Token expired and the client sent nothing to fall back on — ask it to resend.
            return JSONResponse(
                {"error": "Source expired; please re-select your file.",
                 "source_expired": True, "request_id": request_id}, status_code=409)
        elif file is not None:
            data = await file.read()
            source = ingest.from_bytes(data, file.filename)
        elif url:
            source = await asyncio.get_running_loop().run_in_executor(None, ingest.from_url, url)
        else:
            return JSONResponse({"error": "Provide a file or a url"}, status_code=400)
    except ingest.IngestError as exc:
        log.warning("process.ingest_rejected", error=str(exc))
        return JSONResponse({"error": str(exc), "request_id": request_id}, status_code=400)

    sid = source_cache.put(source)
    job_id = uuid.uuid4().hex[:12]
    job = store.create(job_id, request_id)
    job.status = "running"
    loop = asyncio.get_running_loop()

    def emit(stage: str, message: str, *, done=None, total=None, level="info") -> None:
        evt = {"type": "progress", "stage": stage, "message": message, "done": done, "total": total, "level": level}
        loop.call_soon_threadsafe(job.queue.put_nowait, evt)

    # Whole-job wall-clock budget (kept under the client's 120s watchdog) so a
    # pathological clip can't run for minutes; threaded into the Fovea encode loops.
    deadline = time.monotonic() + float(os.getenv("FOVEA_JOB_SECONDS", "100"))

    def run() -> None:
        obs.bind_request(request_id)
        try:
            outs = orchestrator.process(source, parsed, emit, deadline=deadline)
            for otype, data, fmt, meta in outs:
                job.outputs[otype] = {"bytes": data, "fmt": fmt, "meta": meta.model_dump()}
                job.order.append(otype)
                log.info("audit.store", request_id=request_id, job_id=job_id, key=otype,
                         bytes=len(data), fmt=fmt, sha1=_sha1(data),
                         role=("baseline" if "__cmp" in otype else "primary"))
            job.status = "done"
            event = {
                "type": "result",
                "outputs": [{"type": t, "format": job.outputs[t]["fmt"], "meta": job.outputs[t]["meta"]} for t in job.order],
                "meta": job.first["meta"] if job.first else None,
            }
            loop.call_soon_threadsafe(job.queue.put_nowait, event)
        except Exception as exc:  # noqa: BLE001
            job.status, job.error = "error", str(exc)
            log.error("process.failed", error=str(exc), exc_info=True)
            loop.call_soon_threadsafe(
                job.queue.put_nowait, {"type": "error", "error": str(exc), "request_id": request_id}
            )
        finally:
            loop.call_soon_threadsafe(job.queue.put_nowait, {"type": "end"})
            obs.clear_request()

    PIPELINE_EXECUTOR.submit(run)
    return {"job_id": job_id, "request_id": request_id, "source_id": sid}


@app.get("/api/events/{job_id}")
async def events(job_id: str):
    job = store.get(job_id)
    if not job:
        return JSONResponse({"error": "Unknown job"}, status_code=404)

    async def gen():
        while True:
            try:
                # Short keepalive interval so a long, silent encode doesn't look idle
                # to the platform proxy (which resets idle streams -> client watchdog).
                evt = await asyncio.wait_for(job.queue.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if evt.get("type") == "end":
                break
            yield _sse(evt)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


def _serve(out: dict, download: bool, name: str, *, job_id: str | None = None,
           key: str | None = None) -> Response:
    fmt = out["fmt"]
    media = "image/gif" if fmt == "GIF" else "image/png"
    ext = "gif" if fmt == "GIF" else "png"
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{name}.{ext}"'
    log.info("audit.serve", job_id=job_id, key=key, bytes=len(out["bytes"]),
             sha1=_sha1(out["bytes"]), fmt=fmt, download=bool(download))
    return Response(content=out["bytes"], media_type=media, headers=headers)


@app.get("/api/result/{job_id}")
async def result(job_id: str, download: bool = False):
    job = store.get(job_id)
    if not job or not job.first:
        return JSONResponse({"error": "Result not ready"}, status_code=404)
    return _serve(job.first, download, "my-sticker", job_id=job_id,
                  key=(job.order[0] if job.order else None))


@app.get("/api/result/{job_id}/{output}")
async def result_typed(job_id: str, output: str, download: bool = False):
    job = store.get(job_id)
    if not job or not job.order:
        return JSONResponse({"error": "Result not ready"}, status_code=404)
    if output == "all":
        import io as _io
        import zipfile
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for t in job.order:
                if "__cmp" in t:  # comparison baseline is served on demand, not bundled
                    continue
                o = job.outputs[t]
                ext = "gif" if o["fmt"] == "GIF" else "png"
                z.writestr(f"discord-{t}.{ext}", o["bytes"])
        return Response(content=buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="discord-media.zip"', "Cache-Control": "no-store"})
    out = job.outputs.get(output)
    if not out:
        return JSONResponse({"error": "Unknown output"}, status_code=404)
    return _serve(out, download, f"my-{output}", job_id=job_id, key=output)


@app.get("/health")
async def health():
    return {"status": "ok", "bg_available": bg_removal.available(), "bg_ready": _ready["bg"]}


@app.post("/log")
async def client_log(request: Request):
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        payload = {}
    level = str(payload.get("level", "info")).lower()
    getattr(obs.get_logger("client"), "error" if level == "error" else "info")(
        "client.log", **{k: v for k, v in payload.items() if k != "level"}
    )
    return {"ok": True}


# Serve the built frontend (if present) at the root.
_static_dir = os.getenv("STATIC_DIR") or str(Path(__file__).resolve().parents[2] / "frontend" / "dist")
if Path(_static_dir).is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
    log.info("static.mounted", dir=_static_dir)
else:
    @app.get("/")
    async def root():
        return {"service": "discord-sticker-maker", "frontend": "not built", "health": "/health"}
