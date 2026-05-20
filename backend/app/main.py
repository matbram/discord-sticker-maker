"""FastAPI app: upload → SSE progress → download, plus health and client logging."""
from __future__ import annotations

import asyncio
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
from .jobs import JobStore
from .models import ProcessParams
from .pipeline import bg_removal, ingest, orchestrator

log = obs.get_logger("app")

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
):
    request_id = getattr(request.state, "request_id", None) or obs.new_request_id()
    try:
        parsed = ProcessParams.model_validate_json(params) if params else ProcessParams()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"Invalid params: {exc}"}, status_code=400)

    try:
        if file is not None:
            data = await file.read()
            source = ingest.from_bytes(data, file.filename)
        elif url:
            source = await asyncio.get_running_loop().run_in_executor(None, ingest.from_url, url)
        else:
            return JSONResponse({"error": "Provide a file or a url"}, status_code=400)
    except ingest.IngestError as exc:
        log.warning("process.ingest_rejected", error=str(exc))
        return JSONResponse({"error": str(exc), "request_id": request_id}, status_code=400)

    job_id = uuid.uuid4().hex[:12]
    job = store.create(job_id, request_id)
    job.status = "running"
    loop = asyncio.get_running_loop()

    def emit(stage: str, message: str, *, done=None, total=None, level="info") -> None:
        evt = {"type": "progress", "stage": stage, "message": message, "done": done, "total": total, "level": level}
        loop.call_soon_threadsafe(job.queue.put_nowait, evt)

    def run() -> None:
        obs.bind_request(request_id)
        try:
            data, fmt, meta = orchestrator.process(source, parsed, emit)
            job.result, job.fmt, job.meta, job.status = data, fmt, meta.model_dump(), "done"
            loop.call_soon_threadsafe(job.queue.put_nowait, {"type": "result", "meta": job.meta})
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
    return {"job_id": job_id, "request_id": request_id}


@app.get("/api/events/{job_id}")
async def events(job_id: str):
    job = store.get(job_id)
    if not job:
        return JSONResponse({"error": "Unknown job"}, status_code=404)

    async def gen():
        while True:
            try:
                evt = await asyncio.wait_for(job.queue.get(), timeout=60)
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


@app.get("/api/result/{job_id}")
async def result(job_id: str, download: bool = False):
    job = store.get(job_id)
    if not job or job.result is None:
        return JSONResponse({"error": "Result not ready"}, status_code=404)
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = 'attachment; filename="my-sticker.png"'
    return Response(content=job.result, media_type="image/png", headers=headers)


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
