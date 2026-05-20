"""Run the full pipeline (decode -> matte once -> per-output encode) and stream
progress through an ``emit`` callback. Produces one or many Discord outputs."""
from __future__ import annotations

from typing import Callable

from ..models import MAX_ANIM_FRAMES, StickerMeta, profile_for
from ..observability import get_logger, stage
from . import bg_removal, crop_fit, decode, encode, validate
from .ingest import Source

log = get_logger("orchestrator")

EmitFn = Callable[..., None]
MATTING_MAX_SIDE = 512


def _v(x):
    return x.value if hasattr(x, "value") else x


def process(source: Source, params, emit: EmitFn) -> list[tuple[str, bytes, str, StickerMeta]]:
    outputs = params.outputs or []
    specs = [(_v(o.type), _v(o.gif_quality), o) for o in outputs]
    profiles = {t: profile_for(t, gq) for t, gq, _ in specs}
    max_cap = max((p["frame_cap"] for p in profiles.values()), default=MAX_ANIM_FRAMES)

    with stage("decode", kind=source.kind, mime=source.mime):
        emit("decode", "Reading & decoding input")
        frames = decode.decode(source, params)
        if frames.animated and len(frames.frames) > max_cap:
            frames.frames, frames.delays_ms = encode.even_subsample(frames.frames, frames.delays_ms, max_cap)
        emit("decode", f"Decoded {len(frames.frames)} frame(s)", done=len(frames.frames), total=len(frames.frames))

    # Matte ONCE; reused by every output.
    has_alpha = False
    bg_note = None
    if params.remove_bg and bg_removal.available():
        requested = _v(params.bg_model)
        model = bg_removal.pick_model(requested, frames.frames)
        if requested and requested != "auto" and model != requested:
            bg_note = f"Used the {model} model (the {requested} model needs more memory than this server has)."
        frames.frames = crop_fit.downscale_max_side(frames.frames, MATTING_MAX_SIDE)
        total = len(frames.frames)
        with stage("bg_removal", model=model, frames=total):
            emit("bg", f"Removing background ({model})", done=0, total=total)
            frames.frames = bg_removal.remove_bg(
                frames.frames, model,
                progress=lambda d: emit("bg", f"Removing background {d}/{total}", done=d, total=total),
            )
        has_alpha = True
    elif params.remove_bg:
        emit("bg", "Background removal unavailable - skipping", level="warn")

    results: list[tuple[str, bytes, str, StickerMeta]] = []
    for otype, gq, spec in specs:
        prof = profiles[otype]
        eff = params.model_copy(update={
            "priority": spec.priority or params.priority,
            "max_colors": spec.max_colors or params.max_colors,
        })
        # per-output frame cap (matte cap may be higher)
        fr, de = frames.frames, frames.delays_ms
        if frames.animated and len(fr) > prof["frame_cap"]:
            fr, de = encode.even_subsample(fr, de, prof["frame_cap"])
        animated_src = frames.animated and len(fr) > 1
        notes: list[str] = []
        if bg_note:
            notes.append(bg_note)

        with stage("output", type=otype, gif_quality=gq):
            emit("encode", f"Making {otype}…")
            if prof["square"]:
                fitted = crop_fit.fit_square(fr, eff, has_alpha, prof["size"])
                w = h = prof["size"]
                if animated_src and prof["animated_format"] == "APNG":
                    data, fmt, n_frames, fps = encode.encode_animated(fitted, de, eff)
                elif animated_src and prof["animated_format"] == "GIF":
                    data, fmt, n_frames, fps = encode.encode_gif(
                        fitted, de, budget=prof["budget"], max_colors=eff.max_colors, fps_cap=prof.get("fps_cap", 30))
                else:
                    data, fmt = encode.encode_static(fitted[0], eff)
                    n_frames, fps = 1, None
            else:  # gif keep-aspect
                fitted = crop_fit.fit_aspect(fr, eff, prof["max_dim"])
                h, w = fitted[0].shape[:2]
                if animated_src:
                    data, fmt, n_frames, fps = encode.encode_gif(
                        fitted, de, budget=prof["budget"], max_colors=eff.max_colors, fps_cap=prof.get("fps_cap", 24))
                else:
                    data, fmt, n_frames, fps = encode.encode_gif(
                        fitted, [100], budget=prof["budget"], max_colors=eff.max_colors, fps_cap=prof.get("fps_cap", 24))

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
