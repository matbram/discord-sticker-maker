"""Run the full pipeline and stream progress through an ``emit`` callback."""
from __future__ import annotations

from typing import Callable

from ..models import STICKER_SIZE, StickerMeta
from ..observability import get_logger, stage
from . import bg_removal, crop_fit, decode, encode, validate
from .ingest import Source

log = get_logger("orchestrator")

# emit(stage, message, *, done=None, total=None, level="info")
EmitFn = Callable[..., None]


def process(source: Source, params, emit: EmitFn) -> tuple[bytes, str, StickerMeta]:
    with stage("decode", kind=source.kind, mime=source.mime):
        emit("decode", "Reading & decoding input")
        frames = decode.decode(source, params)
        emit("decode", f"Decoded {len(frames.frames)} frame(s)", done=len(frames.frames), total=len(frames.frames))

    has_alpha = False
    if params.remove_bg and bg_removal.available():
        model = bg_removal.pick_model(params.bg_model.value if hasattr(params.bg_model, "value") else params.bg_model, frames.frames)
        total = len(frames.frames)
        with stage("bg_removal", model=model, frames=total):
            emit("bg", f"Removing background ({model})", done=0, total=total)
            frames.frames = bg_removal.remove_bg(
                frames.frames,
                model,
                progress=lambda d: emit("bg", f"Removing background {d}/{total}", done=d, total=total),
            )
        has_alpha = True
    elif params.remove_bg:
        emit("bg", "Background removal unavailable - skipping", level="warn")
        log.warning("orchestrator.bg_unavailable")

    with stage("crop_fit"):
        emit("crop", "Cropping & resizing to 320x320")
        fitted = crop_fit.fit_frames(frames.frames, params, has_alpha)

    with stage("encode", animated=frames.animated and len(fitted) > 1):
        emit("encode", "Encoding & optimizing for Discord")
        if frames.animated and len(fitted) > 1:
            data, fmt, n_frames, fps = encode.encode_animated(fitted, frames.delays_ms, params)
        else:
            data, fmt = encode.encode_static(fitted[0], params)
            n_frames, fps = 1, None

    checklist = validate.build_checklist(data, STICKER_SIZE, STICKER_SIZE, fmt, has_alpha)
    meta = StickerMeta(
        width=STICKER_SIZE,
        height=STICKER_SIZE,
        bytes=len(data),
        frames=n_frames,
        fps=fps,
        animated=n_frames > 1,
        format=fmt,
        under_limit=checklist["under_512kb"],
        checklist=checklist,
    )
    log.info("orchestrator.done", **meta.model_dump(exclude={"checklist"}))
    emit("done", "Sticker ready", done=1, total=1)
    return data, fmt, meta
