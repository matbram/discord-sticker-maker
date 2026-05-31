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
import tempfile
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


def _apngasm_available() -> bool:
    return shutil.which("apngasm") is not None


def _assemble_apngasm(frame_pngs: list[bytes], delays, zlevel: int = 0) -> bytes | None:
    """Assemble with apngasm: inter-frame delta + compression => much smaller files
    (often 3-5x) so far more frames fit under 512KB. apngasm 2.x applies one global
    delay, so we use the mean (video and most GIFs are uniform). None on any failure.

    zlevel selects the deflater: 0=zlib (fast, ~3s/72 frames), 1=7zip (slower, ~17s,
    but noticeably smaller). "smooth" uses 7zip so the freed bytes buy more colours
    at the same frame count."""
    if not _apngasm_available():
        return None
    delay_ms = max(10, int(round(sum(delays) / len(delays)))) if delays else 100
    try:
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, data in enumerate(frame_pngs):
                p = os.path.join(td, f"f{i:04d}.png")
                with open(p, "wb") as fh:
                    fh.write(data)
                paths.append(p)
            out = os.path.join(td, "out.png")
            # -i1: the inter-frame delta gives the size win; the deflater is per zlevel.
            cmd = ["apngasm", out, *paths, str(delay_ms), "1000", f"-z{zlevel}", "-i1"]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0 or not os.path.exists(out):
                log.warning("encode.apngasm_failed", returncode=proc.returncode,
                            stderr=proc.stderr.decode("utf-8", "replace")[:300])
                return None
            with open(out, "rb") as fh:
                return fh.read()
    except Exception:  # noqa: BLE001
        log.warning("encode.apngasm_error", exc_info=True)
        return None


def _assemble_apng(frame_pngs: list[bytes], delays, zlevel: int = 0) -> bytes:
    data = _assemble_apngasm(frame_pngs, delays, zlevel)
    if data is not None:
        return data
    # Fallback: pure-Python apng lib (full frames, preserves per-frame delays).
    from apng import APNG, PNG

    anim = APNG()
    for png, delay in zip(frame_pngs, delays):
        anim.append(PNG.from_bytes(png), delay=int(delay), delay_den=1000)
    anim.num_plays = 0  # loop forever
    return anim.to_bytes()


LADDER_WORKERS = min(2, os.cpu_count() or 1)  # bounded for memory on small hosts


def _run_parallel(funcs):
    if len(funcs) == 1:
        return [funcs[0]()]
    with ThreadPoolExecutor(max_workers=LADDER_WORKERS) as ex:
        return list(ex.map(lambda fn: fn(), funcs))


def _rgba_apng(frames, delays, zlevel: int = 0):
    return _assemble_apng(_parallel_rgba_pngs(frames), delays, zlevel)


def _palette_apng(frames, delays, colors, zlevel: int = 0):
    pngs = _palette_frame_pngs(frames, colors)
    return _assemble_apng(pngs, delays, zlevel) if pngs is not None else None


RGBA_FRAME_FLOOR = 6  # don't drop a truecolor sticker below this many frames


def _rgba_fit_by_frames(frames, delays, budget, zlevel: int = 1, floor: int = RGBA_FRAME_FLOOR):
    """Largest frame count whose *truecolor* RGBA APNG fits ``budget``.

    Truecolor (PNG color type 6) has no shared palette, so colors are never
    banded/washed out — the only lever is the frame count. We measure the full
    encode, estimate how many frames fit, and keep the most that do. Returns
    ``(data, frames, delays)`` or ``None`` if even ``floor`` frames won't fit
    (caller then falls back to a palette)."""
    full = _rgba_apng(frames, delays, zlevel)
    if len(full) <= budget:
        return full, frames, delays
    n = len(frames)
    if n <= floor:
        return None
    per = len(full) / max(n, 1)
    est = int(budget / per * 0.95) if per > 0 else floor
    seen: set[int] = set()
    for target in (est, int(n * 0.66), n // 2, n // 3, floor):
        target = max(floor, min(n - 1, int(target)))
        if target in seen:
            continue
        seen.add(target)
        f2, d2 = even_subsample(frames, delays, target)
        data = _rgba_apng(f2, d2, zlevel)
        if len(data) <= budget:
            return data, f2, d2
    return None


def _rgba_smallest(frames, delays, zlevel: int = 1):
    """Last-resort truecolor encode at the frame floor (used when pngquant is
    unavailable and the full clip is over budget)."""
    f2, d2 = even_subsample(frames, delays, min(len(frames), RGBA_FRAME_FLOOR))
    return _rgba_apng(f2, d2, zlevel), f2, d2


def _write_frames(frames, td) -> None:
    for i, arr in enumerate(frames):
        Image.fromarray(arr, "RGBA").save(os.path.join(td, f"f{i:05d}.png"), "PNG")


def _gif_render(td, fps_v, colors):
    """Run palettegen+paletteuse against frames already written in `td`."""
    pattern = os.path.join(td, "f%05d.png")
    pal = os.path.join(td, f"pal{colors}.png")
    out = os.path.join(td, f"o{colors}.gif")
    fr_arg = f"{fps_v:.3f}"
    r1 = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", fr_arg,
                         "-i", pattern, "-vf", f"palettegen=max_colors={colors}:reserve_transparent=1", pal],
                        capture_output=True)
    if r1.returncode != 0:
        log.warning("gif.palettegen_failed", stderr=r1.stderr.decode("utf-8", "replace")[:200]); return None
    r2 = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", fr_arg,
                         "-i", pattern, "-i", pal, "-lavfi", "paletteuse=alpha_threshold=128", "-loop", "0", out],
                        capture_output=True)
    if r2.returncode != 0 or not os.path.exists(out):
        log.warning("gif.paletteuse_failed", stderr=r2.stderr.decode("utf-8", "replace")[:200]); return None
    with open(out, "rb") as fh:
        data = fh.read()
    log.info("gif.attempt", colors=colors, fps=round(fps_v, 2), bytes=len(data))
    return data


def encode_gif(frames, delays, *, budget, max_colors=256, fps_cap=24) -> tuple[bytes, str, int, float | None]:
    """GIF via ffmpeg palettegen/paletteuse (1-bit alpha). Frames are written once;
    the colour levels are tried in parallel and we keep the highest that fits."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for GIF encoding")
    mean_ms = (sum(delays) / len(delays)) if delays else 100.0
    base_fps = min(fps_cap, max(1.0, 1000.0 / mean_ms))
    color_steps = [c for c in (256, 128, 64, 32) if c <= max_colors] or [max(2, min(max_colors, 256))]

    def run_set(fr, fps_v):
        td = tempfile.mkdtemp()
        try:
            _write_frames(fr, td)
            res = _run_parallel([(lambda c=c: _gif_render(td, fps_v, c)) for c in color_steps])
            return list(zip(color_steps, res))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    full = run_set(frames, base_fps)
    fits = [(c, d) for c, d in full if d is not None and len(d) <= budget]
    if fits:
        c, d = max(fits, key=lambda x: x[0])
        return d, "GIF", len(frames), round(base_fps, 2)

    best = min((d for _, d in full if d), key=len, default=None)
    best_n = len(frames)
    n = len(frames)
    for divisor in (2, 3, 4):
        target = max(6, n // divisor)
        if target >= n:
            continue
        f2, d2 = even_subsample(frames, delays, target)
        fps2 = min(fps_cap, max(1.0, 1000.0 / ((sum(d2) / len(d2)) or 100.0)))
        d = _gif_render_once(f2, fps2, color_steps[-1])
        if d is not None and len(d) <= budget:
            return d, "GIF", len(f2), round(fps2, 2)
        if d is not None and (best is None or len(d) < len(best)):
            best, best_n = d, len(f2)

    if best is None:
        raise RuntimeError("gif encode produced nothing")
    log.warning("gif.over_budget", bytes=len(best), budget=budget)
    return best, "GIF", best_n, round(base_fps, 2)


def _gif_render_once(fr, fps_v, colors):
    td = tempfile.mkdtemp()
    try:
        _write_frames(fr, td)
        return _gif_render(td, fps_v, colors)
    finally:
        shutil.rmtree(td, ignore_errors=True)


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

    def done(data, fr, de):
        return data, "APNG", len(fr), _avg_fps(de)

    # "Richer"/sharp (the sticker default) and the no-pngquant case keep colors:
    # encode truecolor RGBA — which has no shared palette and so can never band or
    # wash out — and drop frames to fit. 7zip (-z1) squeezes more frames under
    # budget before any are cut.
    if priority == "sharp" or not have_pq:
        fit = _rgba_fit_by_frames(frames, delays, budget, zlevel=1)
        if fit is not None:
            return done(*fit)
        if not have_pq:  # no quantizer available -> ship the smallest truecolor clip
            return done(*_rgba_smallest(frames, delays))
        # Truecolor won't fit even at the frame floor (very high-entropy clip):
        # fall through to the shared-palette ladder below as a last resort.

    # Palette path: one shared palette, now up to 256 colors (the old top of 128
    # is what left dense GIFs looking flat). Used by "smooth" (keep frames, spend
    # the budget on colors) and "balanced", plus sharp's extreme fallback.
    floor = {"smooth": 24, "balanced": 32, "sharp": 64}.get(priority, 32)
    ladder = [c for c in (256, 128, 64, 32, 16, 8) if floor <= c <= params.max_colors] \
        or [max(8, min(params.max_colors, 256))]
    # smooth/sharp pay for 7zip so the smaller files buy more colors or frames.
    zlevel = 1 if priority in ("smooth", "sharp") else 0

    # One probe at the floor color, full frames, tells us how big things are.
    probe = _palette_apng(frames, delays, floor, zlevel)
    if probe is None:  # pngquant hiccup -> truecolor with frame drop
        fit = _rgba_fit_by_frames(frames, delays, budget, zlevel=1)
        return done(*(fit or _rgba_smallest(frames, delays)))
    log.info("encode.probe", colors=floor, frames=len(frames), bytes=len(probe), priority=priority)
    BIG = 1_000_000  # rgba sentinel for "max quality"

    if len(probe) <= budget and priority != "sharp":
        # Full frames fit -> spend headroom on quality (more colors / rgba), in parallel.
        cand = [(BIG, lambda: _rgba_apng(frames, delays, zlevel))] + \
               [(c, (lambda c=c: _palette_apng(frames, delays, c, zlevel))) for c in ladder if c > floor]
        res = _run_parallel([fn for _, fn in cand])
        scored = [(q, d) for (q, _), d in zip(cand, res) if d is not None and len(d) <= budget]
        if scored:
            return done(max(scored, key=lambda x: x[0])[1], frames, delays)
        return done(probe, frames, delays)

    # Too big at full frames (or sharp fallback): pick a frame count from the probe,
    # then try colors at that count in parallel and keep the highest that fits.
    per = len(probe) / max(len(frames), 1)
    target = min(len(frames), max(6, int(budget / per * 0.9))) if len(probe) > budget else len(frames)
    f2, d2 = even_subsample(frames, delays, target)
    res = _run_parallel([(lambda c=c: _palette_apng(f2, d2, c, zlevel)) for c in ladder])
    for c, d in zip(ladder, res):
        log.info("encode.attempt", colors=c, frames=len(f2), bytes=(len(d) if d else -1), priority=priority)
    scored = [(c, d) for c, d in zip(ladder, res) if d is not None and len(d) <= budget]
    if scored:
        return done(max(scored, key=lambda x: x[0])[1], f2, d2)

    # Still over -> halve frames at the floor color.
    f3, d3 = even_subsample(frames, delays, max(6, target // 2))
    d = _palette_apng(f3, d3, floor, zlevel)
    if d is not None:
        log.warning("encode.over_budget", bytes=len(d), budget=budget, frames=len(f3))
        return done(d, f3, d3)
    return done(probe, frames, delays)
