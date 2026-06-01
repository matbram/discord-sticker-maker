"""End-to-end smoke test, gated on the external binaries being present.

Synthesizes an *ephemeral* ffmpeg ``lavfi testsrc`` clip in ``tmp_path`` (never
written into the corpus) and runs a real encode + one baseline runner. Skips
entirely when ffmpeg / a GIF engine is unavailable, so the unit suite stays green
with zero binaries installed.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest
from PIL import Image

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
HAVE_GIF_ENGINE = any(shutil.which(b) for b in ("ffmpeg", "gifski", "gifsicle"))

pytestmark = pytest.mark.skipif(
    not (HAVE_FFMPEG and HAVE_GIF_ENGINE),
    reason="needs ffmpeg + a GIF engine for an end-to-end encode",
)


def _make_testsrc(path, *, dur: int = 1, size: str = "128x128", rate: int = 12) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=duration={dur}:size={size}:rate={rate}", str(path)],
        check=True,
    )


def test_encode_video_hits_cap_and_reports(tmp_path):
    from encoder import encode
    from encoder.core.sizes import parse_size_str

    src = tmp_path / "t.mp4"
    _make_testsrc(src)
    out = tmp_path / "o.gif"
    rep = tmp_path / "o.gif.json"

    res = encode(str(src), parse_size_str("512KB"), "cap",
                 budget_seconds=60, max_attempts=20, out_path=str(out), report_path=str(rep))

    assert out.exists()
    assert res.size_bytes <= 512 * 1024
    assert res.size_bytes == out.stat().st_size

    im = Image.open(str(out))
    assert im.format == "GIF"
    assert getattr(im, "n_frames", 1) >= 2

    report = json.loads(rep.read_text())
    assert report["engine_used"] in ("ffmpeg-palette", "gifski", "gifsicle-lossy")
    assert report["under_target"] is True
    assert report["n_frames"] >= 2
    assert isinstance(report["perceptually_lossless"], bool)


def test_encode_preserves_all_frames(tmp_path):
    from encoder import encode
    from encoder.core.frames import frames_from_source, load_gif
    from encoder.core.sizes import parse_size_str

    src = tmp_path / "t.mp4"
    _make_testsrc(src, dur=1, rate=12)
    src_frames = frames_from_source(str(src))
    out = tmp_path / "o.gif"
    encode(str(src), parse_size_str("2MB"), "cap", out_path=str(out), max_attempts=12)
    assert load_gif(str(out)).n == src_frames.n        # the no-frame-dropping guarantee


def test_invisible_mode_runs(tmp_path):
    from encoder import encode
    from encoder.core.sizes import parse_size_str

    src = tmp_path / "t.mp4"
    _make_testsrc(src)
    out = tmp_path / "o.gif"
    res = encode(str(src), parse_size_str("8MB"), "invisible", out_path=str(out), max_attempts=20)
    assert out.exists()
    assert isinstance(res.perceptually_lossless, bool)


def test_bench_runner_one_cell(tmp_path):
    from encoder.core.engines import available_engines
    from encoder.metrics import default_metric

    from bench.manifest import ClipEntry
    from bench.runners import run_clip_target

    engines = available_engines()
    if not engines:
        pytest.skip("no GIF engine available")
    _make_testsrc(tmp_path / "clip.mp4")
    clip = ClipEntry(id="c", path="clip.mp4", category="video_clip")
    rec = run_clip_target(clip, 1024 * 1024, engines[0], default_metric(), str(tmp_path),
                          max_attempts=8)
    assert rec.skipped_reason is None
    assert rec.achieved_bytes is not None and rec.achieved_bytes <= 1024 * 1024
    assert rec.distance is not None
