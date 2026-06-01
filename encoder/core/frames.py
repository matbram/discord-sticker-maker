"""Standalone frame I/O for Fovea.

A ``Frames`` is a list of composited HxWx4 uint8 RGBA arrays plus per-frame
millisecond delays. We decode to *composited* frames (not deltas) so disposal
quirks are resolved up front and every downstream stage sees full frames.

  - Images (GIF/APNG/WebP/PNG/JPEG...) via Pillow's ``ImageSequence`` — Pillow
    applies GIF/APNG disposal, so each iterated frame is already composited.
  - Video via ffmpeg, sampled at a chosen fps (default = source fps capped at the
    GIF ~50 fps ceiling).

This mirrors the idioms in ``backend/app/pipeline/decode.py`` but shares no code:
Fovea is standalone, and — unlike the backend — it never subsamples frames
(spec: keep every frame). Input caps bound memory/time instead.
"""
from __future__ import annotations

import glob
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageSequence

from . import ffmpeg
from .logging import get_logger

log = get_logger("frames")

DEFAULT_FRAME_DELAY_MS = 100


@dataclass
class Frames:
    frames: list[np.ndarray]      # each HxWx4 uint8 (RGBA), composited
    delays_ms: list[int]          # per-frame display time, milliseconds
    src_fps: float | None = None
    loop: int = 0                 # 0 = loop forever

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def height(self) -> int:
        return int(self.frames[0].shape[0])

    @property
    def width(self) -> int:
        return int(self.frames[0].shape[1])

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def total_ms(self) -> int:
        return int(sum(self.delays_ms))


@dataclass
class InputCaps:
    max_file_bytes: int = 200 * 1024 * 1024
    max_pixels: int = 1920 * 1080          # per-frame pixel budget
    max_frames: int = 1200
    max_duration_s: float = 60.0


class InputTooLarge(ValueError):
    pass


class DecodeError(RuntimeError):
    pass


def sniff_kind(path: str) -> str:
    """Classify a file as ``"gif"``, ``"image"``, or ``"video"`` from magic bytes."""
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head[:4] == b"GIF8":
        return "gif"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image"            # PNG / APNG (Pillow handles APNG animation)
    if head[:2] == b"\xff\xd8":
        return "image"            # JPEG
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image"            # WebP (possibly animated)
    if head[:2] in (b"BM",) or head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image"            # BMP / TIFF
    return "video"                # let ffmpeg/ffprobe handle everything else


def _check_file_cap(path: str, caps: InputCaps) -> None:
    size = os.path.getsize(path)
    if size > caps.max_file_bytes:
        raise InputTooLarge(f"file is {size} bytes > cap {caps.max_file_bytes}")


def _enforce_frame_caps(frames: list[np.ndarray], caps: InputCaps) -> None:
    if len(frames) > caps.max_frames:
        raise InputTooLarge(f"{len(frames)} frames > cap {caps.max_frames}")
    if frames:
        h, w = frames[0].shape[:2]
        if h * w > caps.max_pixels:
            raise InputTooLarge(f"{w}x{h} = {w * h} px/frame > cap {caps.max_pixels}")


def _slice_by_time(
    frames: list[np.ndarray], delays: list[int], trim_start_s: float, max_duration_s: float | None
) -> tuple[list[np.ndarray], list[int]]:
    """Keep frames whose playback window intersects [trim_start, trim_start+dur]."""
    if not frames or max_duration_s is None:
        return frames, delays
    start_ms = max(0.0, trim_start_s * 1000.0)
    end_ms = start_ms + max(0.0, max_duration_s) * 1000.0
    out_f: list[np.ndarray] = []
    out_d: list[int] = []
    t = 0.0
    for f, d in zip(frames, delays):
        d = max(1, int(d))
        if t < end_ms and t + d > start_ms:
            out_f.append(f)
            out_d.append(d)
        t += d
        if t >= end_ms:
            break
    if not out_f:
        return [frames[-1]], [max(1, int(delays[-1]))]
    return out_f, out_d


def frames_from_list(
    frames: list[np.ndarray], delays_ms: list[int] | int | None = None, *, loop: int = 0
) -> Frames:
    """Wrap an in-memory list of RGBA arrays into ``Frames`` (library entry point)."""
    if not frames:
        raise DecodeError("no frames provided")
    norm: list[np.ndarray] = []
    for fr in frames:
        arr = np.asarray(fr)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3 + [np.full_like(arr, 255)], axis=-1)
        elif arr.shape[-1] == 3:
            alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=-1)
        norm.append(np.ascontiguousarray(arr.astype(np.uint8)))
    n = len(norm)
    if delays_ms is None:
        delays = [DEFAULT_FRAME_DELAY_MS] * n
    elif isinstance(delays_ms, int):
        delays = [int(delays_ms)] * n
    else:
        delays = [int(d) for d in delays_ms]
        if len(delays) != n:
            raise DecodeError(f"delays length {len(delays)} != frame count {n}")
    return Frames(frames=norm, delays_ms=delays, loop=loop)


def frames_from_image(
    path: str, *, trim_start_s: float = 0.0, max_duration_s: float | None = None,
    caps: InputCaps | None = None,
) -> Frames:
    """Decode an (animated) image via Pillow into composited RGBA frames."""
    caps = caps or InputCaps()
    _check_file_cap(path, caps)
    img = Image.open(path)
    animated = bool(getattr(img, "is_animated", False)) and getattr(img, "n_frames", 1) > 1
    frames: list[np.ndarray] = []
    delays: list[int] = []
    for frame in ImageSequence.Iterator(img):
        frames.append(np.asarray(frame.convert("RGBA"), dtype=np.uint8))
        if animated:
            d = int(frame.info.get("duration", DEFAULT_FRAME_DELAY_MS)) or DEFAULT_FRAME_DELAY_MS
        else:
            d = 0
        delays.append(d)
        if len(frames) > caps.max_frames:
            raise InputTooLarge(f">{caps.max_frames} frames")
    loop = int(img.info.get("loop", 0)) if animated else 0
    if animated:
        frames, delays = _slice_by_time(frames, delays, trim_start_s, max_duration_s)
    _enforce_frame_caps(frames, caps)
    log.info("decode.image", frames=len(frames), animated=animated, size=f"{frames[0].shape[1]}x{frames[0].shape[0]}")
    return Frames(frames=frames, delays_ms=delays, loop=loop)


def frames_from_video(
    path: str, *, fps: float | None = None, max_fps: float = 50.0,
    trim_start_s: float = 0.0, max_duration_s: float | None = None,
    caps: InputCaps | None = None,
) -> Frames:
    """Decode video to all frames at ``fps`` (default = min(source fps, max_fps))."""
    caps = caps or InputCaps()
    _check_file_cap(path, caps)
    if not ffmpeg.have_ffmpeg():
        raise DecodeError("ffmpeg is required to decode video")

    src_fps: float | None = None
    duration_s = max_duration_s
    try:
        info = ffmpeg.probe_source(path)
        src_fps = info.fps
        if info.width and info.height and info.width * info.height > caps.max_pixels:
            raise InputTooLarge(f"{info.width}x{info.height} > cap {caps.max_pixels} px")
        if max_duration_s is None and info.duration_s:
            duration_s = min(info.duration_s, caps.max_duration_s)
    except ffmpeg.FfmpegError:
        log.warning("probe.failed", path=os.path.basename(path))

    use_fps = float(fps) if fps else (min(src_fps, max_fps) if src_fps else max_fps)
    use_fps = max(1.0, min(use_fps, max_fps))

    with tempfile.TemporaryDirectory() as td:
        pattern = os.path.join(td, "f_%05d.png")
        cmd = [ffmpeg.ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error"]
        if trim_start_s:
            cmd += ["-ss", f"{trim_start_s}"]
        if duration_s:
            cmd += ["-t", f"{duration_s}"]
        cmd += ["-i", path, "-vf", f"fps={use_fps:.4f}", "-frames:v", str(caps.max_frames + 1), pattern]
        proc = ffmpeg.run(cmd)
        if proc.returncode != 0:
            raise DecodeError(f"ffmpeg decode failed: {(proc.stderr or b'')[:400]!r}")
        files = sorted(glob.glob(os.path.join(td, "f_*.png")))
        if not files:
            raise DecodeError("ffmpeg produced no frames")
        frames = [np.asarray(Image.open(f).convert("RGBA"), dtype=np.uint8) for f in files]
    _enforce_frame_caps(frames, caps)
    delay = int(round(1000.0 / use_fps))
    log.info("decode.video", frames=len(frames), fps=round(use_fps, 3))
    return Frames(frames=frames, delays_ms=[delay] * len(frames), src_fps=use_fps, loop=0)


def frames_from_source(
    source: str, *, fps: float | None = None, max_fps: float = 50.0,
    trim_start_s: float = 0.0, max_duration_s: float | None = None,
    caps: InputCaps | None = None,
) -> Frames:
    """Decode a path (video / gif / image) into ``Frames``, dispatching on content."""
    kind = sniff_kind(source)
    if kind == "video":
        return frames_from_video(
            source, fps=fps, max_fps=max_fps, trim_start_s=trim_start_s,
            max_duration_s=max_duration_s, caps=caps,
        )
    return frames_from_image(
        source, trim_start_s=trim_start_s, max_duration_s=max_duration_s, caps=caps
    )


def load_gif(path: str) -> Frames:
    """Decode a produced GIF back to composited frames (for metric scoring)."""
    return frames_from_image(path)
