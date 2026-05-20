"""Run the full pipeline and stream progress through an ``emit`` callback."""
from __future__ import annotations

from typing import Callable

from ..models import MAX_ANIM_FRAMES, STICKER_SIZE, StickerMeta
from ..observability import get_logger, stage
from . import bg_removal, crop_fit, decode, encode, validate
from .ingest import Source

log = get_logger("orchestrator")

# emit(stage, message, *, done=None, total=None, level="info")
EmitFn = Callable[..., None]

# Working resolution cap applied before matting: bounds peak memory (avoids the
# OOM kills seen on Railway) and speeds up inference for resolution-aware models.
MATTING_MAX_SIDE = 512


def process(source: Source, params, emit: EmitFn) -> tuple[bytes, str, StickerMeta]:
    notes: list[str] = []

    with stage("decode", kind=source.kind, mime=source.mime):
        emit("decode", "Reading & decoding input")
        frames = decode.decode(source, params)
        # Cap frames before the expensive stages so bg removal, crop and encode
        # all do proportionally less work.
        if frames.animated and len(frames.frames) > MAX_ANIM_FRAMES:
            frames.frames, frames.delays_ms = encode.even_subsample(
                frames.frames, frames.delays_ms, MAX_ANIM_FRAMES
            )
        emit("decode", f"Decoded {len(frames.frames)} frame(s)", done=len(frames.frames), total=len(frames.frames))

    decoded_frames = len(frames.frames)
    has_alpha = False
    if params.remove_bg and bg_removal.available():
        requested = params.bg_model.value if hasattr(params.bg_model, "value") else params.bg_model
        model = bg_removal.pick_model(requested, frames.frames)
        if requested and requested != "auto" and model != requested:
            notes.append(f"Used the {model} model (the {requested} model needs more memory than this server has).")
        # Shrink to a working resolution before matting to keep memory bounded.
        frames.frames = crop_fit.downscale_max_side(frames.frames, MATTING_MAX_SIDE)
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

    is_animated = frames.animated and len(fitted) > 1
    with stage("encode", animated=is_animated):
        emit("encode", "Encoding & optimizing for Discord")
        if is_animated:
            data, fmt, n_frames, fps = encode.encode_animated(fitted, frames.delays_ms, params)
        else:
            data, fmt = encode.encode_static(fitted[0], params)
            n_frames, fps = 1, None

    requested_fps = float(params.max_fps) if is_animated else None
    if is_animated:
        if n_frames < len(fitted):
            notes.append(f"Trimmed to {n_frames} frames to fit Discord's 512 KB limit.")
        if fps and requested_fps and fps < requested_fps - 0.5:
            notes.append(
                f"Playing at ~{fps} fps — a 320x320 sticker under 512 KB can't hold "
                f"{int(requested_fps)} fps for this length. Shorten the clip for a higher frame rate."
            )

    checklist = validate.build_checklist(data, STICKER_SIZE, STICKER_SIZE, fmt, has_alpha)
    meta = StickerMeta(
        width=STICKER_SIZE,
        height=STICKER_SIZE,
        bytes=len(data),
        frames=n_frames,
        fps=fps,
        requested_fps=requested_fps,
        animated=n_frames > 1,
        format=fmt,
        under_limit=checklist["under_512kb"],
        checklist=checklist,
        notes=notes,
    )
    log.info("orchestrator.done", **meta.model_dump(exclude={"checklist", "notes"}), notes=notes)
    emit("done", "Sticker ready", done=1, total=1)
    return data, fmt, meta
