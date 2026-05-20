"""Encode to Discord-ready PNG / APNG and optimize under the size budget.

APNG constraint: every frame shares the default image's IHDR (and PLTE/tRNS),
so we cannot give frames independent palettes. Two valid strategies:
  * RGBA  - true-color + full alpha (best quality); shrink via frame/fps cuts.
  * palette - quantize ALL frames against ONE shared palette via a vertical
    strip through pngquant (8-bit with per-index alpha), then split + assemble.

Performance: frames are capped/even-subsampled before we get here, per-frame PNG
compression is parallelized, and we pick the next reduction step from the measured
size instead of brute-forcing every combination.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from ..models import MAX_ANIM_FRAMES
from ..observability import get_logger

log = get_logger("encode")

WORKERS = min(4, os.cpu_count() or 1)


def _pngquant_available() -> bool:
    return shutil.which("pngquant") is not None


def _pngquant(png_bytes: bytes, colors: int) -> bytes | None:
    try:
        proc = subprocess.run(
            ["pngquant", "--force", "--strip", "--quality=0-100", str(colors), "-"],
            input=png_bytes,
            capture_output=True,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        log.warning("encode.pngquant_failed", returncode=proc.returncode)
    except Exception:  # noqa: BLE001
        log.warning("encode.pngquant_error", exc_info=True)
    return None


def even_subsample(frames, delays, max_n):
    """Keep <= max_n frames spread evenly across the timeline, preserving total duration."""
    n = len(frames)
    if n <= max_n:
        return frames, delays
    bounds = [round(k * n / max_n) for k in range(max_n + 1)]
    nf, nd = [], []
    for k in range(max_n):
        a = bounds[k]
        b = min(max(bounds[k] + 1, bounds[k + 1]), n)
        nf.append(frames[a])
        nd.append(max(1, int(sum(delays[a:b]))))
    return nf, nd


def _avg_fps(delays) -> float | None:
    if not delays:
        return None
    mean_ms = sum(delays) / len(delays)
    return round(1000.0 / mean_ms, 2) if mean_ms > 0 else None


def _rgba_png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def _parallel_rgba_pngs(frames) -> list[bytes]:
    if len(frames) == 1:
        return [_rgba_png(frames[0])]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(_rgba_png, frames))


def _palette_frame_pngs(frames, colors) -> list[bytes] | None:
    """One shared palette for all frames (required for valid APNG)."""
    h, w = frames[0].shape[:2]
    strip = np.concatenate(frames, axis=0)  # (h*n, w, 4)
    buf = io.BytesIO()
    Image.fromarray(strip, "RGBA").save(buf, "PNG", compress_level=1)  # pngquant re-reads pixels
    quantized = _pngquant(buf.getvalue(), colors)
    if not quantized:
        return None
    pal = Image.open(io.BytesIO(quantized))
    pal.load()
    transparency = pal.info.get("transparency")
    out = []
    for i in range(len(frames)):
        frame = pal.crop((0, i * h, w, (i + 1) * h))
        b = io.BytesIO()
        if transparency is not None:
            frame.save(b, "PNG", transparency=transparency)
        else:
            frame.save(b, "PNG")
        out.append(b.getvalue())
    return out


def _assemble_apng(frame_pngs: list[bytes], delays) -> bytes:
    from apng import APNG, PNG

    anim = APNG()
    for data, delay in zip(frame_pngs, delays):
        anim.append(PNG.from_bytes(data), delay=int(delay), delay_den=1000)
    anim.num_plays = 0  # loop forever
    return anim.to_bytes()


def encode_static(arr: np.ndarray, params) -> tuple[bytes, str]:
    buf = io.BytesIO()
    Image.fromarray(arr, "RGBA").save(buf, "PNG", optimize=True)
    data = buf.getvalue()
    if len(data) > params.max_bytes and _pngquant_available():
        smaller = _pngquant(data, params.max_colors)
        if smaller and len(smaller) < len(data):
            data = smaller
    log.info("encode.static", bytes=len(data))
    return data, "PNG"


def encode_animated(frames, delays, params) -> tuple[bytes, str, int, float | None]:
    frames, delays = even_subsample(frames, delays, MAX_ANIM_FRAMES)
    budget = params.max_bytes
    have_pq = _pngquant_available()
    priority = getattr(params, "priority", "balanced")
    priority = priority.value if hasattr(priority, "value") else priority
    best: tuple[bytes, int, list] | None = None

    def consider(data, nf, de):
        nonlocal best
        if best is None or len(data) < len(best[0]):
            best = (data, nf, de)

    def attempt(mode, fr, de, colors=None):
        pngs = _parallel_rgba_pngs(fr) if mode == "rgba" else _palette_frame_pngs(fr, colors)
        if pngs is None:
            return None
        data = _assemble_apng(pngs, de)
        log.info("encode.attempt", mode=mode, colors=colors, frames=len(fr), bytes=len(data), priority=priority)
        consider(data, len(fr), de)
        return data

    def done(data, fr, de):
        return data, "APNG", len(fr), _avg_fps(de)

    # 1. Lossless RGBA at full frames — best quality if it happens to fit.
    data = attempt("rgba", frames, delays)
    if data is not None and len(data) <= budget:
        return done(data, frames, delays)

    if not have_pq:
        # No pngquant: only lever is dropping frames (RGBA).
        for divisor in (2, 3, 4):
            target = max(8, len(frames) // divisor)
            if target >= len(frames):
                continue
            f2, d2 = even_subsample(frames, delays, target)
            data = attempt("rgba", f2, d2)
            if data is not None and len(data) <= budget:
                return done(data, f2, d2)
        assert best is not None
        log.warning("encode.over_budget", bytes=len(best[0]), budget=budget)
        return best[0], "APNG", best[1], _avg_fps(best[2])

    # Palette ladder, bounded above by the user's max_colors and below by the
    # priority's color floor. Fewer colors -> smaller frames -> more frames fit.
    floor = {"smooth": 16, "balanced": 32, "sharp": 64}.get(priority, 32)
    ladder = [c for c in (128, 96, 64, 48, 32, 24, 16) if floor <= c <= params.max_colors]
    if not ladder:
        ladder = [max(16, min(params.max_colors, 64))]

    if priority == "sharp":
        # Color-first: keep colors high, drop frames to fit.
        data = attempt("pal", frames, delays, ladder[0])
        if data is not None and len(data) <= budget:
            return done(data, frames, delays)
        per_frame = len(best[0]) / max(len(best[2]), 1)
        target = max(8, int(budget / per_frame * 0.9))
        if target < len(frames):
            f2, d2 = even_subsample(frames, delays, target)
            for colors in ladder:
                data = attempt("pal", f2, d2, colors)
                if data is not None and len(data) <= budget:
                    return done(data, f2, d2)
    else:
        # Frame-first (smooth / balanced): keep ALL frames, lower colors until it
        # fits — this maximizes smoothness, which is what most people want.
        for colors in ladder:
            data = attempt("pal", frames, delays, colors)
            if data is not None and len(data) <= budget:
                return done(data, frames, delays)
        # Even the fewest colors at full frames won't fit — drop frames at the floor.
        per_frame = len(best[0]) / max(len(best[2]), 1)
        target = max(8, int(budget / per_frame * 0.9))
        if target < len(frames):
            f2, d2 = even_subsample(frames, delays, target)
            data = attempt("pal", f2, d2, ladder[-1])
            if data is not None and len(data) <= budget:
                return done(data, f2, d2)

    assert best is not None
    log.warning("encode.over_budget", bytes=len(best[0]), budget=budget)
    return best[0], "APNG", best[1], _avg_fps(best[2])
