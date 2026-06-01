"""ffmpeg / ffprobe discovery and probing.

Fovea shells out to ffmpeg for video decode and (in the ffmpeg engine) for GIF
palette generation. This module centralizes binary discovery, a logged
``run`` wrapper, and source probing. It is intentionally dependency-free beyond
the stdlib so it can be imported without numpy/Pillow.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

from .logging import get_logger

log = get_logger("ffmpeg")


class FfmpegError(RuntimeError):
    pass


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def have_ffmpeg() -> bool:
    return ffmpeg_path() is not None


def run(cmd: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command capturing stdout/stderr. Does not raise on non-zero exit."""
    log.debug("run", cmd=" ".join(cmd))
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def _stderr(proc: subprocess.CompletedProcess, limit: int = 600) -> str:
    return (proc.stderr or b"").decode("utf-8", "replace")[:limit]


def _parse_rational(value: str | None) -> float | None:
    """Parse ffprobe rationals like ``"30000/1001"`` or ``"25/1"`` to a float fps."""
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


@dataclass
class ProbeInfo:
    width: int | None
    height: int | None
    fps: float | None
    nb_frames: int | None
    duration_s: float | None
    has_video: bool


def probe_source(path: str) -> ProbeInfo:
    """Probe a media file for video stream dimensions, fps, frame count, duration."""
    pp = ffprobe_path()
    if not pp:
        raise FfmpegError("ffprobe not found on PATH")
    cmd = [
        pp, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,width,height,nb_frames:format=duration",
        "-of", "json", str(path),
    ]
    proc = run(cmd)
    if proc.returncode != 0:
        raise FfmpegError(f"ffprobe failed: {_stderr(proc)}")
    data = json.loads(proc.stdout or b"{}")
    fmt = data.get("format", {})
    duration = None
    try:
        duration = float(fmt["duration"]) if "duration" in fmt else None
    except (ValueError, TypeError):
        duration = None
    streams = data.get("streams", [])
    if not streams:
        return ProbeInfo(None, None, None, None, duration, False)
    s = streams[0]
    nb = s.get("nb_frames")
    try:
        nb_frames = int(nb) if nb not in (None, "N/A") else None
    except (ValueError, TypeError):
        nb_frames = None
    return ProbeInfo(
        width=s.get("width"),
        height=s.get("height"),
        fps=_parse_rational(s.get("avg_frame_rate")),
        nb_frames=nb_frames,
        duration_s=duration,
        has_video=True,
    )
