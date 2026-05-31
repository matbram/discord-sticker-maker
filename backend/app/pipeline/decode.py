"""Decode any source into a list of RGBA frames + per-frame delays.

- Static / animated images (PNG, APNG, JPEG, WebP, GIF, HEIC, BMP, TIFF) via Pillow.
  Iterating an animated image sequentially and converting each frame to RGBA lets
  Pillow apply GIF/APNG disposal, so we get fully composited frames (not deltas).
- Video (MP4/MOV/WebM/...) via ffmpeg, sampled at a target fps and trimmed.
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HEIF = True
except Exception:  # noqa: BLE001
    _HEIF = False

from ..models import MAX_ANIM_FRAMES
from ..observability import get_logger
from .encode import even_subsample
from .ingest import IngestError, InputKind, Source

log = get_logger("decode")

DEFAULT_FRAME_DELAY_MS = 100
MAX_FRAMES = 80  # safety cap; orchestrator subsamples to MAX_ANIM_FRAMES


@dataclass
class Frames:
    frames: list[np.ndarray]  # each HxWx4 uint8 (RGBA)
    delays_ms: list[int]
    animated: bool
    src_fps: float | None = None


def _to_rgba(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGBA"), dtype=np.uint8)


def _decode_image(data: bytes, max_duration_s: float | None = None, trim_start_s: float = 0.0) -> Frames:
    """Decode a static or animated image to RGBA frames + per-frame delays.

    For animated images we seek to the trim window *first*, then evenly sample it
    down to ``MAX_ANIM_FRAMES`` before converting. This means:
      - a trim window anywhere in a long GIF is honored (the old code only ever
        looked at the first ~80 source frames, so a later window froze to one
        frame), and
      - we never build more than ``MAX_ANIM_FRAMES`` RGBA arrays, so memory is
        bounded no matter how long the source is.
    """
    import io

    img = Image.open(io.BytesIO(data))
    animated = bool(getattr(img, "is_animated", False)) and getattr(img, "n_frames", 1) > 1

    if not animated:
        return Frames(frames=[_to_rgba(img)], delays_ms=[0], animated=False)

    n = int(getattr(img, "n_frames", 1))
    start_ms = max(0.0, float(trim_start_s) * 1000.0)
    end_ms = start_ms + max(0.0, float(max_duration_s)) * 1000.0 if max_duration_s else float("inf")

    # Pass 1: walk frame durations only (cheap — no RGBA), collecting the indices
    # whose playback window intersects [start, end]. Stop once we're past the
    # window so trimming a 4s clip out of a 5-minute GIF costs ~4s of seeking.
    t = 0.0
    win_idx: list[int] = []
    win_delays: list[int] = []
    last_d = DEFAULT_FRAME_DELAY_MS
    for i in range(n):
        img.seek(i)
        last_d = int(img.info.get("duration", DEFAULT_FRAME_DELAY_MS)) or DEFAULT_FRAME_DELAY_MS
        seg_start, seg_end = t, t + last_d
        if seg_start < end_ms and seg_end > start_ms:
            clipped = min(seg_end, end_ms) - max(seg_start, start_ms)
            win_idx.append(i)
            win_delays.append(max(1, int(round(clipped))))
        t = seg_end
        if t >= end_ms:
            break

    if not win_idx:  # window starts past the end of the clip -> last frame, static
        img.seek(n - 1)
        return Frames(frames=[_to_rgba(img)], delays_ms=[max(1, last_d)], animated=False)

    # Sample down to MAX_ANIM_FRAMES (duration-preserving) BEFORE converting, so
    # only the kept indices are ever turned into RGBA arrays.
    if len(win_idx) > MAX_ANIM_FRAMES:
        win_idx, win_delays = even_subsample(win_idx, win_delays, MAX_ANIM_FRAMES)

    frames: list[np.ndarray] = []
    for i in win_idx:
        img.seek(i)
        frames.append(_to_rgba(img))

    log.info("decode.image_animated", source_frames=n, kept=len(frames),
             trim_start_s=trim_start_s, max_duration_s=max_duration_s)
    return Frames(frames=frames, delays_ms=list(win_delays), animated=len(frames) > 1)


def _decode_video(data: bytes, max_fps: int, max_duration_s: float, trim_start_s: float) -> Frames:
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "input")
        with open(inp, "wb") as fh:
            fh.write(data)
        # Sample at an fps that lands ~MAX_ANIM_FRAMES across the clip rather than
        # extracting hundreds of frames we'd only throw away. Covers the full
        # duration evenly and keeps every downstream stage cheap.
        eff_fps = min(max_fps, max(1.0, MAX_ANIM_FRAMES / max_duration_s))
        pattern = os.path.join(td, "f_%05d.png")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(trim_start_s),
            "-t", str(max_duration_s),
            "-i", inp,
            "-vf", f"fps={eff_fps:.3f}",
            "-frames:v", str(MAX_FRAMES),
            pattern,
        ]
        log.info("decode.ffmpeg", cmd=" ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", "replace")[:1000]
            log.error("decode.ffmpeg_failed", returncode=proc.returncode, stderr=stderr)
            raise IngestError("Could not decode video")
        files = sorted(glob.glob(os.path.join(td, "f_*.png")))
        if not files:
            raise IngestError("No frames extracted from video")
        frames = [np.asarray(Image.open(f).convert("RGBA"), dtype=np.uint8) for f in files]
        delay = int(round(1000 / eff_fps))
        log.info("decode.video", frames=len(frames), fps=round(eff_fps, 2))
        return Frames(frames=frames, delays_ms=[delay] * len(frames), animated=len(frames) > 1, src_fps=eff_fps)


def decode(source: Source, params) -> Frames:
    if source.kind == InputKind.VIDEO:
        return _decode_video(source.data, params.max_fps, params.max_duration_s, params.trim_start_s)
    return _decode_image(source.data, params.max_duration_s, params.trim_start_s)
