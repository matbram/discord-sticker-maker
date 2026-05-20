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
from PIL import Image, ImageSequence

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HEIF = True
except Exception:  # noqa: BLE001
    _HEIF = False

from ..models import MAX_ANIM_FRAMES
from ..observability import get_logger
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


def _decode_image(data: bytes) -> Frames:
    import io

    img = Image.open(io.BytesIO(data))
    animated = bool(getattr(img, "is_animated", False)) and getattr(img, "n_frames", 1) > 1

    if not animated:
        arr = _to_rgba(img)
        return Frames(frames=[arr], delays_ms=[0], animated=False)

    frames: list[np.ndarray] = []
    delays: list[int] = []
    for frame in ImageSequence.Iterator(img):
        frames.append(_to_rgba(frame))
        delays.append(int(frame.info.get("duration", DEFAULT_FRAME_DELAY_MS)) or DEFAULT_FRAME_DELAY_MS)
        if len(frames) >= MAX_FRAMES:
            break
    log.info("decode.image_animated", frames=len(frames))
    return Frames(frames=frames, delays_ms=delays, animated=True)


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
    return _decode_image(source.data)
