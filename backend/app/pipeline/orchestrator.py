"""Run the full pipeline and stream progress through an ``emit`` callback.

decode + background-matte happen ONCE and are cached (keyed by source + trim +
bg settings), then each requested output is cropped/fit/encoded from the shared
frames. Tweaking a downstream setting (zoom, output type, GIF quality) reuses the
cached cutout instead of recomputing the slow part.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Callable

from .. import matte_cache
from ..models import StickerMeta, profile_for, resolve_aspect
from ..observability import get_logger, stage
from . import bg_removal, crop_fit, decode, encode, fovea_gif, validate
from .ingest import Source

log = get_logger("orchestrator")

EmitFn = Callable[..., None]
MATTING_MAX_SIDE = 512   # matte at <=512 for memory/speed
# Working/cached frames are capped to this longest side — and it's the ceiling for a GIF's
# "Source" dimensions, so it must be >= common source sizes (a 720x1280 phone clip needs
# 1280, not the old 640 which halved it to 360x640). Env-tunable: raise for crisper
# source-res GIFs (costs memory + encode time), lower if the box is memory-constrained.
WORK_MAX_SIDE = int(os.getenv("FOVEA_WORK_MAX_SIDE", "1280"))
MATTE_FRAME_CAP = max(profile_for(t)["frame_cap"] for t in ("sticker", "emoji", "gif"))


def _v(x):
    return x.value if hasattr(x, "value") else x


def _gif_dims(data: bytes) -> tuple[int, int] | None:
    """Actual canvas W×H from a GIF's logical screen descriptor (little-endian u16 at
    bytes 6–9). Fovea may descend resolution to fit the budget, so the encoded GIF can
    be smaller than the pre-encode `fitted` frames — read the truth from the bytes."""
    if len(data) >= 10 and data[:3] == b"GIF":
        return (data[6] | (data[7] << 8), data[8] | (data[9] << 8))
    return None


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:12]


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


def process(source: Source, params, emit: EmitFn,
            deadline: float | None = None) -> list[tuple[str, bytes, str, StickerMeta]]:
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
    extras: list[tuple[str, bytes, str, StickerMeta]] = []  # comparison baselines (served, not shown)
    completed = 0
    for otype, gq, spec in specs:
        prof = profiles[otype]
        # Effective size limit + dimensions: per-output overrides from the UI, else profile.
        budget = spec.max_bytes or prof["budget"]
        hard_limit = spec.max_bytes or prof["hard_limit"]
        out_size = spec.max_dim or prof.get("size")            # square side (sticker/emoji)
        eff = params.model_copy(update={
            "priority": spec.priority or params.priority,
            "max_colors": spec.max_colors or params.max_colors,
            "max_bytes": budget,
            "zoom": spec.zoom if spec.zoom is not None else params.zoom,
            "offset_x": spec.offset_x if spec.offset_x is not None else params.offset_x,
            "offset_y": spec.offset_y if spec.offset_y is not None else params.offset_y,
            "fit_mode": spec.fit_mode if spec.fit_mode is not None else params.fit_mode,
        })
        log.info("audit.output.budget", type=otype, requested_max_bytes=spec.max_bytes,
                 budget=budget, hard_limit=hard_limit, square_size=out_size,
                 requested_w=spec.width, requested_h=spec.height, requested_max_dim=spec.max_dim)
        fr, de = base_frames, delays
        if animated_src and len(fr) > prof["frame_cap"]:
            fr, de = encode.even_subsample(fr, de, prof["frame_cap"])
        is_anim = animated_src and len(fr) > 1
        notes: list[str] = [bg_note] if bg_note else []
        mode = _v(spec.mode)
        # Per-output wall-clock cap: one pathological output (e.g. 512px at a tight
        # budget, where the color-floor trim loop does many high-res re-encodes) must
        # not blow the whole job or stall the client's SSE connection. The Fovea search
        # is anytime, so hitting this just returns the best candidate so far.
        out_deadline = time.monotonic() + float(os.getenv("FOVEA_OUTPUT_SECONDS", "45"))
        if deadline is not None:
            out_deadline = min(deadline, out_deadline)
        comparison = None
        baseline_data = None
        report = None

        with stage("output", type=otype, gif_quality=gq):
            emit("encode", f"Making {otype}…")
            if prof["square"]:
                fitted = crop_fit.fit_square(fr, eff, has_alpha, out_size)
                w = h = out_size
                if is_anim and prof["animated_format"] == "APNG":
                    data, fmt, n_frames, fps = encode.encode_animated(fitted, de, eff)
                elif is_anim and prof["animated_format"] == "GIF":
                    # Square emoji: its dimensions are fixed (Discord requires 128×128), so the
                    # encoder must NOT trade resolution for color here.
                    data, fmt, n_frames, fps, report = fovea_gif.gif_encode(
                        fitted, de, budget=budget, max_colors=eff.max_colors,
                        fps_cap=prof.get("fps_cap", 30), priority=_v(eff.priority),
                        mode=mode, deadline=out_deadline, notes=notes, allow_descent=False)
                else:
                    data, fmt = encode.encode_static(fitted[0], eff)
                    n_frames, fps = 1, None
            else:
                # The animated output, keeping the SOURCE aspect ratio. Two formats:
                #   * WebP (anim_format="webp"): VP8 holds rich color at the SOURCE resolution
                #     (or an exact Custom W×H) within budget — perceptually lossless, all frames.
                #   * GIF  (default): GIF/LZW can't be lossless at source res in a small budget
                #     (0.157 bits/px => ~2 colors), so we emit the LARGEST perceptually-lossless
                #     GIF — cap the longest side, then the resolution-for-color descent finds the
                #     biggest palette-rich size, keeping all frames.
                sh, sw = fr[0].shape[:2]
                anim_fmt = _v(spec.anim_format)
                if anim_fmt == "webp":
                    if spec.width and spec.height:
                        aw, ah, long_side = spec.width, spec.height, max(spec.width, spec.height)
                    else:
                        aw, ah, long_side = sw, sh, max(sw, sh)
                    fitted = crop_fit.fit_to_canvas(fr, eff, has_alpha, aw, ah, long_side)
                    h, w = fitted[0].shape[:2]
                    if is_anim:
                        data, fmt, n_frames, fps, report = fovea_gif.webp_encode(
                            fitted, de, budget=budget, deadline=out_deadline, notes=notes)
                    else:
                        data, fmt = encode.encode_static(fitted[0], eff)
                        n_frames, fps = 1, None
                else:  # gif — largest perceptually-lossless GIF, sized to the byte budget
                    # A lossless GIF holds ~1 bit/pixel of real detail (LZW on a per-frame
                    # palette), so the largest losslessly-encodable longest side scales with
                    # sqrt(budget / frames). Start there (<= source) and let the descent refine
                    # to true losslessness. The upshot the user feels: a BIGGER file-size limit
                    # yields a BIGGER GIF, up to the source resolution (~8MB for 720x1280x29).
                    n_est = max(1, len(fr))
                    ar = max(sw, sh) / max(1, min(sw, sh))
                    cap = max(128, int(round((budget * 8.0 / n_est * ar) ** 0.5)))  # ~1 bit/px
                    if spec.width and spec.height:
                        aw, ah, long_side = spec.width, spec.height, min(max(spec.width, spec.height), cap)
                    else:
                        aw, ah, long_side = sw, sh, min(max(sw, sh), cap)
                    fitted = crop_fit.fit_to_canvas(fr, eff, has_alpha, aw, ah, long_side)
                    h, w = fitted[0].shape[:2]
                    if is_anim:
                        data, fmt, n_frames, fps, report = fovea_gif.gif_encode(
                            fitted, de, budget=budget, max_colors=eff.max_colors,
                            fps_cap=prof.get("fps_cap", 24), priority=_v(eff.priority),
                            mode=mode, deadline=out_deadline, notes=notes, allow_descent=True)
                    else:
                        data, fmt = encode.encode_static(fitted[0], eff)
                        n_frames, fps = 1, None

        # Report the animated output's REAL dimensions (the encoder may shrink to fit the
        # budget, keeping the source aspect): read GIF from its header, WebP from the report.
        if fmt == "GIF":
            actual = _gif_dims(data)
            if actual:
                w, h = actual
        elif fmt == "WEBP" and isinstance(report, dict) and report.get("scaled_dim"):
            try:
                w, h = (int(v) for v in report["scaled_dim"].split("x"))
            except Exception:  # noqa: BLE001
                pass

        animated = n_frames > 1
        if animated and n_frames < len(fitted):
            notes.append(f"Trimmed to {n_frames} frames to fit the {otype}'s size limit.")
        if animated and fps and fps < params.max_fps - 0.5:
            notes.append(f"Playing at ~{fps} fps — for more frames, lower colors or shorten the clip.")

        checklist = validate.build_checklist_for(otype, data, w, h, fmt, has_alpha, prof)
        meta = StickerMeta(
            output_type=otype, width=w, height=h, bytes=len(data), frames=n_frames, fps=fps,
            requested_fps=float(params.max_fps) if animated else None, animated=animated,
            format=fmt, under_limit=len(data) <= hard_limit, checklist=checklist, notes=notes,
            comparison=comparison, report=report,
        )
        log.info("orchestrator.output", **meta.model_dump(exclude={"checklist", "notes", "comparison"}))
        results.append((otype, data, fmt, meta))
        if baseline_data is not None and comparison is not None:
            bframes = comparison["legacy"]["frames"]
            extras.append((f"{otype}__cmp", baseline_data, "GIF", StickerMeta(
                output_type=f"{otype}__cmp", width=w, height=h, bytes=len(baseline_data),
                frames=bframes, fps=None, animated=bframes > 1, format="GIF",
                under_limit=len(baseline_data) <= hard_limit, checklist={}, notes=[])))
        completed += 1
        emit("output_done", f"{otype} ready", done=completed, total=len(specs))

    results.extend(extras)
    emit("done", "All set", done=1, total=1)
    return results
