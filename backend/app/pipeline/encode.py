"""Encode to Discord-ready PNG / APNG and optimize under the size budget.

APNG constraint: every frame shares the default image's IHDR (and PLTE/tRNS),
so we cannot give frames independent palettes. Two valid strategies:
  * RGBA  - true-color + full alpha (best quality); shrink via frame/fps cuts.
  * palette - quantize ALL frames against ONE shared palette via a vertical
    strip through pngquant (8-bit with per-index alpha), then split + assemble.
We escalate from best quality toward smaller until we fit ``max_bytes``.
"""
from __future__ import annotations

import io
import shutil
import subprocess

import numpy as np
from PIL import Image

from ..observability import get_logger

log = get_logger("encode")


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


def _apply_skip(frames, delays, skip):
    if skip <= 1:
        return frames, delays
    nf, nd = [], []
    for i in range(0, len(frames), skip):
        nf.append(frames[i])
        nd.append(int(sum(delays[i : i + skip])) or 1)
    return nf, nd


def _avg_fps(delays) -> float | None:
    if not delays:
        return None
    mean_ms = sum(delays) / len(delays)
    return round(1000.0 / mean_ms, 2) if mean_ms > 0 else None


def _rgba_frame_pngs(frames) -> list[bytes]:
    out = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f, "RGBA").save(buf, "PNG", optimize=True)
        out.append(buf.getvalue())
    return out


def _palette_frame_pngs(frames, colors) -> list[bytes] | None:
    """One shared palette for all frames (required for valid APNG)."""
    h, w = frames[0].shape[:2]
    strip = np.concatenate(frames, axis=0)  # (h*n, w, 4)
    buf = io.BytesIO()
    Image.fromarray(strip, "RGBA").save(buf, "PNG")
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
    have_pq = _pngquant_available()

    plans: list[tuple[str, int, int | None]] = [("rgba", 1, None), ("rgba", 2, None)]
    if have_pq:
        plans += [
            ("pal", 1, 128), ("pal", 1, 64), ("rgba", 3, None),
            ("pal", 2, 64), ("pal", 2, 32), ("rgba", 4, None), ("pal", 3, 32),
        ]
    else:
        plans += [("rgba", 3, None), ("rgba", 4, None), ("rgba", 6, None)]

    best: tuple[bytes, int, list] | None = None
    for mode, skip, colors in plans:
        fr, de = _apply_skip(frames, delays, skip)
        if len(fr) < 1:
            continue
        frame_pngs = _rgba_frame_pngs(fr) if mode == "rgba" else _palette_frame_pngs(fr, colors)
        if frame_pngs is None:
            continue
        data = _assemble_apng(frame_pngs, de)
        log.info("encode.attempt", mode=mode, skip=skip, colors=colors, frames=len(fr), bytes=len(data))
        if best is None or len(data) < len(best[0]):
            best = (data, len(fr), de)
        if len(data) <= params.max_bytes:
            return data, "APNG", len(fr), _avg_fps(de)

    assert best is not None
    log.warning("encode.over_budget", bytes=len(best[0]), budget=params.max_bytes)
    return best[0], "APNG", best[1], _avg_fps(best[2])
