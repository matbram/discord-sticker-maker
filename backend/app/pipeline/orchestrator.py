"""Run the full pipeline and stream progress through an ``emit`` callback.

decode + background-matte happen ONCE and are cached (keyed by source + trim +
bg settings), then each requested output is cropped/fit/encoded from the shared
frames. Tweaking a downstream setting (zoom, output type, GIF quality) reuses the
cached cutout instead of recomputing the slow part.
"""
from __future__ import annotations

import hashlib
from typing import Callable

from .. import matte_cache
from ..models import StickerMeta, profile_for, resolve_aspect
from ..observability import get_logger, stage
from . import bg_removal, crop_fit, decode, encode, validate
from .ingest import Source

log = get_logger("orchestrator")

EmitFn = Callable[..., None]
MATTING_MAX_SIDE = 512   # matte at <=512 for memory/speed
WORK_MAX_SIDE = 640      # working/cached frames capped here (outputs are <=480)
MATTE_FRAME_CAP = max(profile_for(t)["frame_cap"] for t in ("sticker", "emoji", "gif"))


def _v(x):
    return x.value if hasattr(x, "value") else x


def _matte_key(source: Source, params) -> tuple:
    return (
        hashlib.sha1(source.data).hexdigest()[:16],
        source.kind,
        round(params.trim_start_s, 2), round(params.max_duration_s, 2), int(params.max_fps),
        bool(params.remove_bg), _v(params.bg_model),
    )


def _decode_and_matte(source: Source, params, emit: EmitFn) -> dict:
    """Returns the shared, reusable frame state (cached by caller)."""
    with stage("decode", kind=source.kind, mime=source.mime):
        emit("decode", "Reading & decoding input")
        frames = decode.decode(source, params)
        if frames.animated and len(frames.frames) > MATTE_FRAME_CAP:
            frames.frames, frames.delays_ms = encode.even_subsample(frames.frames, frames.delays_ms, MATTE_FRAME_CAP)
        working = crop_fit.downscale_max_side(frames.frames, WORK_MAX_SIDE)
        emit("decode", f"Decoded {len(working)} frame(s)", done=len(working), total=len(working))

    has_alpha = False
    bg_note = None
    if params.remove_bg and bg_removal.available():
        requested = _v(params.bg_model)
        model = bg_removal.pick_model(requested, working)
        if requested and requested != "auto" and model != requested:
            bg_note = f"Used the {model} model (the {requested} model needs more memory than this server has)."
        working = crop_fit.downscale_max_side(working, MATTING_MAX_SIDE)
        total = len(working)
        with stage("bg_removal", model=model, frames=total):
            emit("bg", f"Removing background ({model})", done=0, total=total)
            working = bg_removal.remove_bg(
                working, model,
                progress=lambda d: emit("bg", f"Removing background {d}/{total}", done=d, total=total),
            )
        has_alpha = True
    elif params.remove_bg:
        emit("bg", "Background removal unavailable - skipping", level="warn")

    return {"frames": working, "delays": frames.delays_ms, "animated": frames.animated,
            "has_alpha": has_alpha, "bg_note": bg_note}


def process(source: Source, params, emit: EmitFn) -> list[tuple[str, bytes, str, StickerMeta]]:
    specs = [(_v(o.type), _v(o.gif_quality), o) for o in (params.outputs or [])]
    profiles = {t: profile_for(t, gq) for t, gq, _ in specs}

    key = _matte_key(source, params)
    shared = matte_cache.get(key)
    if shared is not None:
        log.info("matte_cache.hit", frames=len(shared["frames"]))
        emit("decode", "Reusing cutout", done=1, total=1)
    else:
        shared = _decode_and_matte(source, params, emit)
        approx = sum(int(f.nbytes) for f in shared["frames"])
        matte_cache.put(key, shared, approx)

    base_frames, delays = shared["frames"], shared["delays"]
    animated_src, has_alpha, bg_note = shared["animated"], shared["has_alpha"], shared["bg_note"]

    results: list[tuple[str, bytes, str, StickerMeta]] = []
    for otype, gq, spec in specs:
        prof = profiles[otype]
        eff = params.model_copy(update={
            "priority": spec.priority or params.priority,
            "max_colors": spec.max_colors or params.max_colors,
        })
        fr, de = base_frames, delays
        if animated_src and len(fr) > prof["frame_cap"]:
            fr, de = encode.even_subsample(fr, de, prof["frame_cap"])
        is_anim = animated_src and len(fr) > 1
        notes: list[str] = [bg_note] if bg_note else []

        with stage("output", type=otype, gif_quality=gq):
            emit("encode", f"Making {otype}…")
            if prof["square"]:
                fitted = crop_fit.fit_square(fr, eff, has_alpha, prof["size"])
                w = h = prof["size"]
                if is_anim and prof["animated_format"] == "APNG":
                    data, fmt, n_frames, fps = encode.encode_animated(fitted, de, eff)
                elif is_anim and prof["animated_format"] == "GIF":
                    data, fmt, n_frames, fps = encode.encode_gif(
                        fitted, de, budget=prof["budget"], max_colors=eff.max_colors, fps_cap=prof.get("fps_cap", 30))
                else:
                    data, fmt = encode.encode_static(fitted[0], eff)
                    n_frames, fps = 1, None
            else:
                sh, sw = fr[0].shape[:2]
                aw, ah = resolve_aspect(spec.aspect, sw, sh)
                fitted = crop_fit.fit_to_canvas(fr, eff, has_alpha, aw, ah, prof["max_dim"])
                h, w = fitted[0].shape[:2]
                src_de = de if is_anim else [100]
                data, fmt, n_frames, fps = encode.encode_gif(
                    fitted, src_de, budget=prof["budget"], max_colors=eff.max_colors, fps_cap=prof.get("fps_cap", 24))

        animated = n_frames > 1
        if animated and n_frames < len(fitted):
            notes.append(f"Trimmed to {n_frames} frames to fit the {otype}'s size limit.")
        if animated and fps and fps < params.max_fps - 0.5:
            notes.append(f"Playing at ~{fps} fps — for more frames, lower colors or shorten the clip.")

        checklist = validate.build_checklist_for(otype, data, w, h, fmt, has_alpha, prof)
        meta = StickerMeta(
            output_type=otype, width=w, height=h, bytes=len(data), frames=n_frames, fps=fps,
            requested_fps=float(params.max_fps) if animated else None, animated=animated,
            format=fmt, under_limit=len(data) <= prof["hard_limit"], checklist=checklist, notes=notes,
        )
        log.info("orchestrator.output", **meta.model_dump(exclude={"checklist", "notes"}))
        results.append((otype, data, fmt, meta))
        emit("output_done", f"{otype} ready", done=len(results), total=len(specs))

    emit("done", "All set", done=1, total=1)
    return results
